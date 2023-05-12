from tqdm.auto import tqdm
from functools import partial
import numpy as np

import torch
import torch.distributed as dist
import torchmetrics.image as Metrics
from torch.utils.data.distributed import DistributedSampler
from core.utils import ConditionType


class Evaluator:
    def __init__(self, dh, logger, evalloader, evalsampler):
        self.batch_size = evalloader.batch_size
        self.image_size = evalloader.dataset.resolution[0]
        self.shape = (self.batch_size, 3, self.image_size, self.image_size)
        self.condition = evalloader.dataset.condition
        self.evalloader = evalloader
        self.evalsampler = evalsampler

        self.logger = logger

        self.fid = Metrics.FrechetInceptionDistance(
            dist_sync_on_step=True, reset_real_features=False
        ).to(dh.dev())
        self.fid.set_dtype(torch.float64)
        self.iScore = Metrics.InceptionScore(dist_sync_on_step=True).to(dh.dev())
        self.dh = dh
        self.computed_real_features = False

    def process(self, xin):
        x = xin.clone()
        del xin
        x = x.clamp(min=-1.0, max=1.0)
        x = x.sub(-1.0)
        x = x.div(2.0)
        x = x.mul(255)
        x = x.add(0.5)
        x = x.clamp(0, 255)
        x = x.to(torch.uint8)
        return x

    def run(
        self,
        model,
        epoch,
        network,
        ddim=False,
        ode=False,
        dpmsolver=False,
        eps=1e-3,
        tol=1e-3,
    ):
        all_metrics = {"vb": [], "mse": [], "xstart_mse": []}
        all_bpd = []
        self.fid.reset()
        self.iScore.reset()
        if isinstance(self.evalsampler, DistributedSampler):
            self.evalsampler.set_epoch(epoch)
        if self.dh.distributed:
            dist.barrier()
        batches_completed = 0

        if ddim:
            sf = model.ddim_sample_fn
        elif ode[0]:
            sf = partial(
                model.ode_sample_fn, method=ode[1], eps=eps, rtol=tol, atol=tol
            )
        elif dpmsolver[0]:
            sf = partial(
                model.dpmsolver_sample_fn,
                method=dpmsolver[2],
                steps=dpmsolver[1],
                order=3,
                eps=eps,
                rtol=tol,
                atol=tol,
            )
        else:
            sf = model.sample_fn

        for batch in tqdm(
            self.evalloader, desc="Evaluating: ", disable=self.dh.rank != 0
        ):
            if self.condition == ConditionType.inpainting:
                xreal = batch["gt"].to(self.dh.dev())
                xcond = batch["cond"].to(self.dh.dev())
                mask = batch["mask"].to(self.dh.dev())
            elif self.condition == ConditionType.colourisation:
                xreal = batch["gt"].to(self.dh.dev())
                xcond = batch["cond"].to(self.dh.dev())
                mask = None
            elif self.condition == ConditionType.none:
                xreal = batch["gt"].to(self.dh.dev())
                xcond = None
                mask = None

            xsamp = sf(
                network,
                self.shape,
                clip_denoised=True,
                x0=xreal,
                y=xcond,
                mask=mask,
                progress=False,
                device=self.dh.dev(),
            )

            if (batches_completed < 10) and hasattr(model, "calc_bpd_loop"):
                batch_metrics = model.calc_bpd_loop(
                    network=network, x0=xreal, clip_denoised=True, y=xcond
                )

                for key, term_list in all_metrics.items():
                    terms = batch_metrics[key].mean(dim=0) / dist.get_world_size()
                    dist.all_reduce(terms)
                    term_list.append(terms.detach().cpu().numpy())

                total_bpd = batch_metrics["total_bpd"]

                total_bpd = total_bpd.mean() / dist.get_world_size()
                dist.all_reduce(total_bpd)
                all_bpd.append(total_bpd.item())
                print(f"bpd: {np.mean(all_bpd)}")
            elif hasattr(model, "likelihood_fn"):
                bpd, _, _ = model.likelihood_fn(network, torch.clone(xsamp).detach())
                bpd = bpd.mean() / dist.get_world_size()
                dist.all_reduce(bpd)
                all_bpd.append(bpd.item())
                print(f"bpd: {np.mean(all_bpd)}")

            xsamp = self.process(xsamp)
            xreal = self.process(xreal)

            self.fid.update(xsamp, False)
            if not self.computed_real_features:
                self.fid.update(xreal, True)
            self.iScore.update(xsamp)

            print(f"fid ({self.dh.dev()}): {self.fid.compute()}")
            print(f"iscore ({self.dh.dev()}): {self.iScore.compute()}")

            if self.dh.distributed:
                dist.barrier()

            batches_completed += 1

        self.computed_real_features = True

        if self.dh.distributed:
            dist.barrier()

        fid = self.fid.compute()
        iScoreMean, iScoreStd = self.iScore.compute()

        dist.reduce(fid, 0, dist.ReduceOp.SUM)
        fid /= self.dh.world_size
        dist.reduce(iScoreMean, 0, dist.ReduceOp.SUM)
        iScoreMean /= self.dh.world_size
        dist.reduce(iScoreStd, 0, dist.ReduceOp.SUM)
        iScoreStd /= self.dh.world_size

        if self.dh.rank == 0:
            self.logger.tbStat("fid", fid.item(), epoch)
            self.logger.tbStat("iscore mean", iScoreMean.item(), epoch)
            self.logger.tbStat("iscore std", iScoreStd.item(), epoch)
            self.logger.tbStat("nll", np.mean(all_bpd), epoch)
