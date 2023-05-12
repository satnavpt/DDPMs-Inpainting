import copy
import functools
from functools import partial
import os
import math
import shutil
from tqdm.auto import tqdm

import blobfile as bf
import numpy as np
import torch as th
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torchvision.utils import save_image as _save_image
from torch.optim import AdamW
from torch.utils.data.distributed import DistributedSampler

from .utils import update_ema, UniformSampler
from core.eval_util import Evaluator
from core.utils import ConditionType


def zero_grad(network_params):
    for param in network_params:
        if param.grad is not None:
            param.grad.detach_()
            param.grad.zero_()


class TrainLoop:
    def __init__(
        self,
        *,
        dh,
        logger,
        network,
        model,
        epochs,
        schedule_sampler,
        condition,
        trainloader,
        trainsampler,
        sampleloader,
        samplesampler,
        evalloader,
        evalsampler,
        ode,
        dpmsolver,
        batch_size,
        lr,
        weight_decay,
        ema_rate,
        save_interval,
        chkpt_dir,
        sample_dir,
        sample_intv,
        sample_size,
        eval_intv,
        backup_dir,
    ):
        self.dh = dh
        self.logger = logger

        self.network = network
        self.model = model
        self.epochs = epochs

        self.schedule_sampler = schedule_sampler or UniformSampler(model)

        self.condition = condition
        self.trainloader = trainloader
        self.trainsampler = trainsampler
        self.sampleloader = sampleloader
        self.samplesampler = samplesampler
        self.evalloader = evalloader
        self.evalsampler = evalsampler

        self.ode = ode
        self.dpmsolver = dpmsolver

        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )

        self.save_interval = save_interval
        self.chkpt_dir = chkpt_dir

        self.sample_dir = sample_dir
        self.sample_intv = sample_intv
        self.sample_size = sample_size

        self.eval_intv = eval_intv

        self.backup_dir = backup_dir

        self.resume_epoch = 0
        self.network_params = list(self.network.parameters())
        self.master_params = self.network_params
        self._load_and_sync_parameters()

        self.opt = AdamW(self.master_params, lr=self.lr, weight_decay=self.weight_decay)

        if self.resume_epoch:
            try:
                self._load_optimizer_state()
                self.ema_params = [
                    self._load_ema_parameters(rate) for rate in self.ema_rate
                ]
            except Exception as e:
                logger.log(f"{e}: failed to load optimzer state / ema params")
                self.ema_params = [
                    copy.deepcopy(self.master_params) for _ in range(len(self.ema_rate))
                ]
        else:
            self.ema_params = [
                copy.deepcopy(self.master_params) for _ in range(len(self.ema_rate))
            ]

        if th.cuda.is_available() and self.dh.distributed:
            self.use_ddp = True
            self.ddp_network = DDP(
                self.network,
                device_ids=[self.dh.dev()],
                output_device=self.dh.dev(),
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=False,
            )
        else:
            self.use_ddp = False
            self.ddp_network = self.network

        self.save_image = partial(_save_image, normalize=True, value_range=(-1.0, 1.0))

    def train(self):
        # trunk-ignore(ruff/B020)
        for self.epoch in range(self.resume_epoch + 1, self.epochs):
            self.step()
            self.save()
            self.sample()
            self.eval()

            if self.dh.distributed:
                dist.barrier()

    def step(self):
        self.network.train()
        if isinstance(self.trainsampler, DistributedSampler):
            self.trainsampler.set_epoch(self.epoch)
        if self.dh.distributed:
            dist.barrier()
        with tqdm(
            self.trainloader,
            desc=f"{self.epoch}/{self.epochs} epochs",
            disable=self.dh.rank != 0,
        ) as pb:
            for batch in pb:
                if self.condition == ConditionType.none:
                    x0 = batch["gt"].to(self.dh.dev())
                    y = None
                    mask = None
                elif self.condition == ConditionType.colourisation:
                    x0 = batch["gt"].to(self.dh.dev())
                    y = batch["cond"].to(self.dh.dev())
                    mask = None
                elif self.condition == ConditionType.inpainting:
                    x0 = batch["gt"].to(self.dh.dev())
                    y = batch["cond"].to(self.dh.dev())
                    mask = batch["mask"].to(self.dh.dev())

                zero_grad(self.network_params)

                t, weights = self.schedule_sampler.sample(x0.shape[0], self.dh.dev())

                compute_losses = functools.partial(
                    self.model.training_losses,
                    self.ddp_network,
                    x0,
                    t,
                    y=y,
                    mask=mask,
                )

                losses = compute_losses()
                if type(losses) != dict:
                    losses = {"loss": losses}

                loss = (losses["loss"] * weights).mean()
                if self.dh.distributed:
                    dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                    loss.div_(self.dh.world_size)

                for k, v in losses.items():
                    w_v = v * weights
                    self.logger.updateMetric(k, w_v.mean().item(), pb)

                loss.backward()

                sqsum = 0.0
                for p in self.master_params:
                    sqsum += (p.grad**2).sum().item()
                self.logger.updateMetric("grad_norm", np.sqrt(sqsum), pb)

                self.optimize()

            self.logger.endMetricEpoch(self.epoch)

    def optimize(self):
        self.opt.step()
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.master_params, rate=rate)

    def eval(self):
        self.network.eval()
        if (self.epoch % self.eval_intv == 0) or (self.epoch == self.epochs - 1):
            self.logger.log("loading ema")
            master_params = self._master_params_to_state_dict(self.master_params)
            ema_params = self._master_params_to_state_dict(self.ema_params[0])
            self.network.load_state_dict(ema_params)

            if hasattr(self.model, "eps"):
                eps = self.model.eps
            else:
                eps = None

            evaluator = Evaluator(
                self.dh, self.logger, self.evalloader, self.evalsampler
            )
            evaluator.run(
                self.model,
                self.epoch,
                self.ddp_network,
                False,
                self.ode,
                self.dpmsolver,
                eps,
            )
            self.logger.log("loading master params")
            self.network.load_state_dict(master_params)

    def process(self, xin):
        x = xin.clone()
        del xin
        x = x.clamp(min=-1.0, max=1.0)
        x = x.sub(-1.0)
        x = x.div(2.0)
        x = x.mul(255)
        x = x.add(0.5)
        x = x.clamp(0, 255)
        x = x.to(th.uint8)
        return x

    def sample(self):
        self.network.eval()
        if isinstance(self.samplesampler, DistributedSampler):
            self.samplesampler.set_epoch(self.epoch)
        if self.dh.distributed:
            dist.barrier()
        if (self.epoch % self.sample_intv == 0) or (self.epoch == self.epochs - 1):
            self.logger.log("loading ema")
            master_params = self._master_params_to_state_dict(self.master_params)
            ema_params = self._master_params_to_state_dict(self.ema_params[0])
            self.network.load_state_dict(ema_params)
            self.network.eval()

            x = next(iter(self.sampleloader))
            if self.condition == ConditionType.colourisation:
                x0 = x["gt"].to(self.dh.dev())
                y = x["cond"].to(self.dh.dev())
                mask = None
            elif self.condition == ConditionType.inpainting:
                x0 = x["gt"].to(self.dh.dev())
                y = x["cond"].to(self.dh.dev())
                mask = x["mask"].to(self.dh.dev())
            else:
                x0 = None
                y = None
                mask = None

            image_size = self.sampleloader.dataset.resolution[0]
            nrow = math.floor(math.sqrt(self.sample_size))
            shape = (self.sample_size // self.dh.world_size, 3, image_size, image_size)

            xsamp = self.model.sample_fn(
                self.network,
                shape,
                clip_denoised=True,
                x0=x0,
                y=y,
                mask=mask,
                progress=self.dh.rank == 0,
                device=self.dh.dev(),
            )

            if self.dh.distributed:
                x_list = [
                    th.zeros(shape, device=self.dh.dev())
                    for _ in range(self.dh.world_size)
                ]
                dist.all_gather(x_list, xsamp)
                xsamp = th.cat(x_list, dim=0)

                if y is not None:
                    y_list = [
                        th.zeros(shape, device=self.dh.dev())
                        for _ in range(self.dh.world_size)
                    ]
                    dist.all_gather(y_list, y)
                    y = th.cat(y_list, dim=0)

                if x0 is not None:
                    x0_list = [
                        th.zeros(shape, device=self.dh.dev())
                        for _ in range(self.dh.world_size)
                    ]
                    dist.all_gather(x0_list, x0)
                    x0 = th.cat(x0_list, dim=0)

            if self.dh.rank == 0:
                if y is not None:
                    out = th.cat([y.cpu(), xsamp.cpu(), x0.cpu()], dim=0)
                else:
                    out = xsamp.cpu()

                self.save_image(
                    out,
                    os.path.join(self.sample_dir, f"{self.epoch:04d}.jpg"),
                    nrow=nrow,
                )
                self.logger.tbImage("sample", self.process(out), self.epoch)

            self.logger.log("loading master params")
            self.network.load_state_dict(master_params)

    def save(self):
        if (self.dh.rank) == 0 and (
            (self.epoch % self.save_interval == 0) or (self.epoch == self.epochs - 1)
        ):

            def save_checkpoint(rate, params):
                state_dict = self._master_params_to_state_dict(params)
                self.logger.log(f"saving model {rate}...")
                if not rate:
                    filename = f"{(self.epoch):04d}/model_{(self.epoch):04d}.pt"
                else:
                    filename = f"{(self.epoch):04d}/model_ema_{(self.epoch):04d}.pt"
                with bf.BlobFile(bf.join(self.chkpt_dir, filename), "wb") as f:
                    th.save(state_dict, f)

            save_checkpoint(0, self.master_params)
            for rate, params in zip(self.ema_rate, self.ema_params):
                save_checkpoint(rate, params)

            with bf.BlobFile(
                bf.join(
                    self.chkpt_dir, f"{(self.epoch):04d}/opt_{(self.epoch):04d}.pt"
                ),
                "wb",
            ) as f:
                th.save(self.opt.state_dict(), f)

            try:
                if (self.epoch - 2 >= 0) and ((self.epoch - 2) % 100 != 0):
                    self.logger.log("deleting old checkpoint")
                    p = os.path.join(self.chkpt_dir, f"{(self.epoch-2):04d}")
                    shutil.rmtree(p)
            except Exception as e:
                self.logger.log(e)
                self.logger.log("no previous checkpoint to delete")

    def _master_params_to_state_dict(self, master_params):
        state_dict = self.network.state_dict()
        for i, (name, _value) in enumerate(self.network.named_parameters()):
            assert name in state_dict
            state_dict[name] = master_params[i]
        return state_dict

    def _state_dict_to_master_params(self, state_dict):
        params = [state_dict[name] for name, _ in self.network.named_parameters()]
        return params

    def _load_and_sync_parameters(self):
        resume_checkpoint = self.find_resume_checkpoint()

        if resume_checkpoint:
            self.epoch = self.resume_epoch = self.parse_resume_epoch_from_foldername(
                resume_checkpoint
            )
            self.logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
            self.network.load_state_dict(
                self.dh.load_state_dict(
                    os.path.join(resume_checkpoint, f"model_{(self.epoch):04d}.pt"),
                    map_location=self.dh.dev(),
                )
            )

    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.master_params)

        resume_checkpoint = self.find_resume_checkpoint()
        if resume_checkpoint:
            if self.dh.rank == 0:
                self.logger.log(f"loading EMA from checkpoint: {resume_checkpoint}...")
                state_dict = self.dh.load_state_dict(
                    os.path.join(resume_checkpoint, f"model_ema_{(self.epoch):04d}.pt"),
                    map_location=self.dh.dev(),
                )
                ema_params = self._state_dict_to_master_params(state_dict)

        self.dh.sync_params(ema_params)
        return ema_params

    def _load_optimizer_state(self):
        resume_checkpoint = self.find_resume_checkpoint()
        if resume_checkpoint:
            self.logger.log(
                f"loading optimizer state from checkpoint: {resume_checkpoint}"
            )
            state_dict = self.dh.load_state_dict(
                os.path.join(resume_checkpoint, f"opt_{(self.epoch):04d}.pt"),
                map_location=self.dh.dev(),
            )
            self.opt.load_state_dict(state_dict)

    def find_resume_checkpoint(self):
        try:
            dirs = list(reversed(sorted(os.listdir(self.chkpt_dir))))
            last = dirs[0]
            chkpt_path = os.path.join(self.chkpt_dir, last)
            return chkpt_path
        except:
            return None

    def parse_resume_epoch_from_foldername(self, foldername):
        split = foldername.split("/")
        try:
            r = int(split[len(split) - 1])
            return r
        except:
            return 0
