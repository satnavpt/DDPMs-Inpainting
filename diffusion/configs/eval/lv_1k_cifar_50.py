from scripts.evaluate import main as fseval
from core.dataset import CIFAR10
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
    run_name = "lv_1k_cifar_50"
    chkpt_name = "lv_1k_cifar"
    var_type = VarType.LEARNED_RANGE

    dataset = dotdict(
        dict(
            dataset=CIFAR10,
            data_dir="./datasets",
            pin_memory=False,
            drop_last=True,
        )
    )

    network = dotdict(
        dict(
            network_type=NetworkType.UNET,
            image_size=32,
            channel_mult=(1, 2, 2, 2),
            num_channels=128,
            num_res_blocks=3,
            attention_resolutions="16,8",
            var_type=var_type,
            num_heads=4,
            num_heads_upsample=-1,
            use_scale_shift_norm=True,
            dropout=0.3,
            conv_resample=True,
            condition=ConditionType.none,
        )
    )

    model = dotdict(
        dict(
            model_type=ModelType.DDPM,
            diffusion_steps=1000,
            noise_schedule=NoiseSchedule.COSINE,
            loss_type=LossType.RESCALED_MSE,
            var_type=var_type,
            mean_type=MeanType.EPSILON,
            rescale_timesteps=True,
            timestep_respacing="50",
        )
    )

    args = dotdict(
        dict(
            run_name=run_name,
            dataset=dataset,
            network=network,
            model=model,
            batch_size=128,
            ema_rate="0.9999",
            log_dir="/rds/user/pt442/hpc-work/backups/" + run_name + "/logs",
            save_interval=1,
            chkpt_dir="/rds/user/pt442/hpc-work/backups/" + chkpt_name + "/chkpts",
        )
    )

    fseval(args)


if __name__ == "__main__":
    main()
