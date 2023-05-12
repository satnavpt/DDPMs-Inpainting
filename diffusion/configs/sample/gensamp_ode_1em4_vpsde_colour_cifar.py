from scripts.generate import main as gs
from core.dataset import CIFAR10Colourisation
from core.utils import ConditionType
from core.models.model_utils import NoiseSchedule, NetworkType, ModelType
from core.models.sdes.predictors.predictor import NonePredictor
from core.models.sdes.correctors.corrector import NoneCorrector


class dotdict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def main():
    run_name = "ode_1em4_vpsde_colour_cifar"
    chkpt_name = "vpsde_cifar"
    condition = ConditionType.colourisation

    dataset = dotdict(
        dict(
            dataset=CIFAR10Colourisation,
            data_dir="./datasets",
            pin_memory=False,
            drop_last=True,
        )
    )

    network = dotdict(
        dict(
            network_type=NetworkType.NCSN,
            channel_mult=(1, 2, 2, 2),
            num_channels=128,
            num_res_blocks=8,
            attention_resolutions=(16,),
            dropout=0.1,
            conv_resample=True,
            image_size=32,
            condition=condition,
            skip_rescale=True,
            init_scale=0.0,
            continuous=True,
            fourier_scale=16,
            nonlinearity="swish",
            sigma_max=0.01,
            sigma_min=50,
            num_scales=1000,
            scale_by_sigma=False,
        )
    )

    model = dotdict(
        dict(
            model_type=ModelType.VPSDE,
            diffusion_steps=1000,
            noise_schedule=NoiseSchedule.LINEAR,
            continuous=True,
            likelihood_weighting=False,
            sampling_eps=1e-5,
            tol=1e-5,
            reduce_mean=True,
            probability_flow=False,
            denoise=True,
            n_steps=1,
            snr=0.16,
            rescale_timesteps=False,
            timestep_respacing="",
            ode=(True, "RK45"),
            dpmsolver=(False, 0, "singlestep"),
            predictor=NonePredictor,
            corrector=NoneCorrector,
            condition=condition,
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
            sample_dir="/rds/user/pt442/hpc-work/backups/" + run_name + "/samples",
        )
    )

    gs(args)


if __name__ == "__main__":
    main()
