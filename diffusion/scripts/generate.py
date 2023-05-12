import os
import copy
from functools import partial

import torch as th
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torchvision.utils import save_image as _save_image

from core.dist_util import DistributedHelper as DH
from core.logger import Logger
from core.dataset import get_dataloader
from core.networks.unet import Network as UNet
from core.networks.ncsnpp import Network as NCSN
from core.models.model_utils import (
    Respacer,
    space_timesteps,
    get_named_beta_schedule,
    VarType,
    NetworkType,
    ModelType,
)
from core.models.sdes.vpsdeModel import Model as VPSDE
from core.utils import ConditionType


def main(args):
    dh = DH()
    os.makedirs(os.path.expanduser(args.chkpt_dir), exist_ok=True)
    os.makedirs(os.path.expanduser(args.log_dir), exist_ok=True)
    os.makedirs(os.path.expanduser(args.sample_dir), exist_ok=True)

    logger = Logger(dh, args.log_dir)

    logger.log("creating data loaders...")
    evalloader, evalsampler = get_dataloader(
        dataset=args.dataset.dataset,
        batch_size=args.batch_size,
        split="eval",
        pin_memory=args.dataset.pin_memory,
        drop_last=args.dataset.drop_last,
        num_workers=dh.num_dev(),
        distributed=dh.distributed,
    )

    logger.log("creating network...")
    network = create_network(cf=args.network).to(dh.dev())

    logger.log("creating diffusion...")
    model = create_model(cf=args.model)

    logger.log("load params...")
    network_params = list(network.parameters())
    master_params = network_params
    epoch = load_and_sync_parameters(dh, logger, network, args.chkpt_dir)

    ema_params = [
        load_ema_parameters(dh, logger, master_params, epoch, network, args.chkpt_dir)
        for rate in args.ema_rate
    ]

    if th.cuda.is_available() and dh.distributed:
        ddp_network = DDP(
            network,
            device_ids=[dh.dev()],
            output_device=dh.dev(),
            broadcast_buffers=False,
            bucket_cap_mb=128,
            find_unused_parameters=False,
        )
    else:
        ddp_network = network

    network.eval()
    logger.log("loading ema")
    master_params = master_params_to_state_dict(network, master_params)
    ema_params = master_params_to_state_dict(network, ema_params[0])
    network.load_state_dict(ema_params)
    samples = args.batch_size
    generate_samples(
        dh=dh,
        loader=evalloader,
        model=model,
        network=ddp_network,
        ddim=args.model.timestep_respacing.startswith("ddim"),
        ode=args.model.ode,
        dpmsolver=args.model.dpmsolver,
        samples=samples,
        condition=args.network.condition,
        image_size=args.network.image_size,
        dir=args.sample_dir,
        eps=args.model.sampling_eps,
        tol=args.model.tol,
    )


def generate_samples(
    dh,
    loader,
    model,
    network,
    ddim,
    ode,
    dpmsolver,
    samples,
    condition,
    image_size,
    dir,
    eps,
    tol,
):
    x = next(iter(loader))
    if condition == ConditionType.colourisation:
        x0 = x["gt"].to(dh.dev())
        y = x["cond"].to(dh.dev())
        mask = None
    elif condition == ConditionType.inpainting:
        x0 = x["gt"].to(dh.dev())
        y = x["cond"].to(dh.dev())
        mask = x["mask"].to(dh.dev())
    else:
        x0 = None
        y = None
        mask = None

    shape = (samples, 3, image_size, image_size)

    if ddim:
        sf = model.ddim_sample_fn
    elif ode[0]:
        sf = partial(model.ode_sample_fn, method=ode[1], eps=eps, atol=tol, rtol=tol)
    elif dpmsolver[0]:
        sf = partial(
            model.dpmsolver_sample_fn,
            method=dpmsolver[2],
            steps=dpmsolver[1],
            order=3,
            eps=eps,
            atol=tol,
            rtol=tol,
        )
    else:
        sf = model.sample_fn

    xsamp = sf(
        network,
        shape,
        clip_denoised=True,
        x0=x0,
        y=y,
        mask=mask,
        progress=dh.rank == 0,
        device=dh.dev(),
    )

    if dh.rank == 0:
        save_image(
            xsamp.cpu(), os.path.join(dir, "xsamp.jpg"), nrow=int(samples**0.5)
        )
        if y is not None:
            save_image(y.cpu(), os.path.join(dir, "y.jpg"), nrow=int(samples**0.5))
        if x0 is not None:
            save_image(x0.cpu(), os.path.join(dir, "x0.jpg"), nrow=int(samples**0.5))


save_image = partial(_save_image, normalize=True, value_range=(-1.0, 1.0))


def create_network(cf):
    if cf.network_type == NetworkType.UNET:
        attention_ds = []
        for res in cf.attention_resolutions.split(","):
            attention_ds.append(cf.image_size // int(res))
        return UNet(
            in_channels=(3 if cf.condition == ConditionType.none else 6),
            hid_channels=cf.num_channels,
            out_channels=(
                3 if cf.var_type not in [VarType.LEARNED, VarType.LEARNED_RANGE] else 6
            ),
            num_res_blocks=cf.num_res_blocks,
            attention_resolutions=tuple(attention_ds),
            dropout=cf.dropout,
            channel_mult=cf.channel_mult,
            num_heads=cf.num_heads,
            num_heads_upsample=cf.num_heads_upsample,
            use_scale_shift_norm=cf.use_scale_shift_norm,
            conv_resample=cf.conv_resample,
        )
    elif cf.network_type == NetworkType.NCSN:
        return NCSN(
            nonlinearity="swish",
            sigma_max=cf.sigma_max,
            sigma_min=cf.sigma_min,
            num_scales=cf.num_scales,
            nf=cf.num_channels,
            num_res_blocks=cf.num_res_blocks,
            attn_resolutions=cf.attention_resolutions,
            dropout=cf.dropout,
            resamp_with_conv=cf.conv_resample,
            ch_mult=cf.channel_mult,
            image_size=cf.image_size,
            conditional=True,
            skip_rescale=cf.skip_rescale,
            init_scale=cf.init_scale,
            continuous=cf.continuous,
            fourier_scale=cf.fourier_scale,
            num_channels=3,
            scale_by_sigma=cf.scale_by_sigma,
        )


def create_model(cf):
    if cf.model_type == ModelType.DDPM:
        betas = get_named_beta_schedule(cf.noise_schedule, cf.diffusion_steps)
        if not cf.timestep_respacing:
            timestep_respacing = [cf.diffusion_steps]
        else:
            timestep_respacing = cf.timestep_respacing
        return Respacer(
            use_timesteps=space_timesteps(cf.diffusion_steps, timestep_respacing),
            betas=betas,
            mean_type=cf.mean_type,
            var_type=cf.var_type,
            loss_type=cf.loss_type,
            rescale_timesteps=cf.rescale_timesteps,
        )
    elif cf.model_type == ModelType.VPSDE:
        return VPSDE(
            num_timesteps=cf.diffusion_steps,
            noise_schedule=cf.noise_schedule,
            continuous=cf.continuous,
            likelihood_weighting=cf.likelihood_weighting,
            eps=cf.sampling_eps,
            reduce_mean=cf.reduce_mean,
            ode=cf.ode,
            predictor=cf.predictor,
            corrector=cf.corrector,
            probability_flow=cf.probability_flow,
            denoise=cf.denoise,
            n_steps=cf.n_steps,
            snr=cf.snr,
            condition=cf.condition,
        )


def load_and_sync_parameters(dh, logger, network, chkpt_dir):
    resume_checkpoint = find_resume_checkpoint(chkpt_dir)
    epoch = parse_resume_epoch_from_foldername(resume_checkpoint)
    logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
    network.load_state_dict(
        dh.load_state_dict(
            os.path.join(resume_checkpoint, f"model_{(epoch):04d}.pt"),
            map_location=dh.dev(),
        )
    )
    return epoch


def load_ema_parameters(dh, logger, master_params, epoch, network, chkpt_dir):
    ema_params = copy.deepcopy(master_params)

    resume_checkpoint = find_resume_checkpoint(chkpt_dir)
    if dh.rank == 0:
        logger.log(f"loading EMA from checkpoint: {resume_checkpoint}...")
        state_dict = dh.load_state_dict(
            os.path.join(resume_checkpoint, f"model_ema_{(epoch):04d}.pt"),
            map_location=dh.dev(),
        )
        ema_params = state_dict_to_master_params(state_dict, network)

    dh.sync_params(ema_params)
    return ema_params


def find_resume_checkpoint(chkpt_dir):
    dirs = list(reversed(sorted(os.listdir(chkpt_dir))))
    last = dirs[0]
    chkpt_path = os.path.join(chkpt_dir, last)
    return chkpt_path


def parse_resume_epoch_from_foldername(foldername):
    split = foldername.split("/")
    r = int(split[len(split) - 1])
    return r


def state_dict_to_master_params(state_dict, network):
    params = [state_dict[name] for name, _ in network.named_parameters()]
    return params


def master_params_to_state_dict(network, master_params):
    state_dict = network.state_dict()
    for i, (name, _value) in enumerate(network.named_parameters()):
        assert name in state_dict
        state_dict[name] = master_params[i]
    return state_dict
