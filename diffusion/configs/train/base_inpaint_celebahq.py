from scripts.train import main as trainScript
from core.dataset import CelebAHQInpainting
from core.utils import ConditionType
from core.models.model_utils import VarType, MeanType, LossType, NoiseSchedule, NetworkType, ModelType


class dotdict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def main():
    run_name = "base_inpaint_celebahq"
    var_type = VarType.FIXED_SMALL
    batch_size = 64

    dataset = dotdict(dict(
        dataset=CelebAHQInpainting,
        data_dir="./datasets",
        pin_memory=False,
        drop_last=True,
    ))

    network = dotdict(dict(
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
        conv_resample=False,
        condition=ConditionType.inpainting,
    ))

    model = dotdict(dict(
        model_type=ModelType.DDPM,
        diffusion_steps=1000,
        noise_schedule=NoiseSchedule.LINEAR,
        loss_type=LossType.MSE,
        var_type=var_type,
        mean_type=MeanType.EPSILON,
        rescale_timesteps=False,
        timestep_respacing="",
    ))

    train = dotdict(dict(
        epochs=1900,
        batch_size=batch_size,
        lr=2e-5,
        weight_decay=0.0,
        ema_rate="0.9999",
    ))

    args = dotdict(dict(
        run_name=run_name,

        dataset=dataset,
        network=network,
        model=model,
        train=train,

        log_dir="/rds/user/pt442/hpc-work/backups/" + run_name + "/logs",

        save_interval=1,
        chkpt_dir="/rds/user/pt442/hpc-work/backups/" + run_name + "/chkpts",

        sample_dir="/rds/user/pt442/hpc-work/backups/" + run_name + "/samples",
        sample_intv=25,
        sample_size=16,

        eval_intv=450,

        backup_dir="/rds/user/pt442/hpc-work/backups/" + run_name + "/chkpts",

    ))

    trainScript(args)


if __name__ == "__main__":
    main()
