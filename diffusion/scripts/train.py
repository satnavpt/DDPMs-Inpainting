import os

from core.dist_util import DistributedHelper as DH
from core.logger import Logger
from core.dataset import get_dataloader
from core.utils import create_uniform_sampler, ConditionType
from core.train_util import TrainLoop
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


def main(args):
    dh = DH()

    os.makedirs(os.path.expanduser(args.sample_dir), exist_ok=True)
    os.makedirs(os.path.expanduser(args.chkpt_dir), exist_ok=True)
    os.makedirs(os.path.expanduser(args.log_dir), exist_ok=True)

    logger = Logger(dh, args.log_dir)

    logger.log("creating data loaders...")
    trainloader, trainsampler = get_dataloader(
        dataset=args.dataset.dataset,
        batch_size=args.train.batch_size,
        split="train",
        pin_memory=args.dataset.pin_memory,
        drop_last=args.dataset.drop_last,
        num_workers=dh.num_dev(),
        distributed=dh.distributed,
    )
    sampleloader, samplesampler = get_dataloader(
        dataset=args.dataset.dataset,
        batch_size=args.sample_size,
        split="sample",
        pin_memory=args.dataset.pin_memory,
        drop_last=args.dataset.drop_last,
        num_workers=dh.num_dev(),
        distributed=dh.distributed,
    )
    evalloader, evalsampler = get_dataloader(
        dataset=args.dataset.dataset,
        batch_size=args.train.batch_size,
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

    logger.log("creating schedule sampler...")
    schedule_sampler = create_uniform_sampler(model)

    logger.log("training...")
    TrainLoop(
        dh=dh,
        logger=logger,
        network=network,
        model=model,
        epochs=args.train.epochs,
        schedule_sampler=schedule_sampler,
        condition=args.network.condition,
        trainloader=trainloader,
        trainsampler=trainsampler,
        sampleloader=sampleloader,
        samplesampler=samplesampler,
        evalloader=evalloader,
        evalsampler=evalsampler,
        ode=args.model.ode,
        dpmsolver=args.model.dpmsolver,
        batch_size=args.train.batch_size,
        lr=args.train.lr,
        weight_decay=args.train.weight_decay,
        ema_rate=args.train.ema_rate,
        save_interval=args.save_interval,
        chkpt_dir=args.chkpt_dir,
        sample_dir=args.sample_dir,
        sample_intv=args.sample_intv,
        sample_size=args.sample_size,
        eval_intv=args.eval_intv,
        backup_dir=args.backup_dir,
    ).train()


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
