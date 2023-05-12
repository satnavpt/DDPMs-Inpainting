from scripts.evaluate import main as fseval
from core.dataset import CelebAHQInpainting
from core.utils import ConditionType
from core.models.model_utils import (
    VarType,
    MeanType,
    LossType,
    NoiseSchedule,
    NetworkType,
    ModelType,
)


class dotdict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def main():
    run_name = "lv_inpaint_celebahq_50"
    chkpt_name = "lv_inpaint_celebahq"
    var_type = VarType.LEARNED_RANGE

    dataset = dotdict(
        dict(
            dataset=CelebAHQInpainting,
            data_dir="./datasets",
            pin_memory=False,
            drop_last=True,
        )
    )

    network = dotdict(
        dict(
            network_type=NetworkType.UNET,
            image_size=256,
            channel_mult=(1, 1, 2, 2, 4, 4),
            num_channels=128,
            num_res_blocks=2,
            attention_resolutions="16",
            var_type=var_type,
            num_heads=1,
            num_heads_upsample=-1,
            use_scale_shift_norm=False,
            dropout=0.0,
            conv_resample=True,
            condition=ConditionType.inpainting,
        )
    )

    model = dotdict(
        dict(
            model_type=ModelType.DDPM,
            diffusion_steps=1000,
            noise_schedule=NoiseSchedule.LINEAR,
            loss_type=LossType.RESCALED_MSE,
            var_type=var_type,
            mean_type=MeanType.EPSILON,
            ode=(False, 0),
            dpmsolver=(False, "", "adaptive"),
            rescale_timesteps=False,
            timestep_respacing="50",
        )
    )

    args = dotdict(
        dict(
            run_name=run_name,
            dataset=dataset,
            network=network,
            model=model,
            ema_rate="0.9999",
            batch_size=64,
            log_dir="/rds/user/pt442/hpc-work/backups/" + run_name + "/logs",
            save_interval=1,
            chkpt_dir="/rds/user/pt442/hpc-work/backups/" + chkpt_name + "/chkpts",
        )
    )

    fseval(args)


if __name__ == "__main__":
    main()