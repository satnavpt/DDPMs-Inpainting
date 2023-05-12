import math
import numpy as np
import enum
import torch
from ddpm import Model as DDPMModel


class MeanType(enum.Enum):
    PREVIOUS_X = enum.auto()
    START_X = enum.auto()
    EPSILON = enum.auto()


class VarType(enum.Enum):
    LEARNED = enum.auto()
    FIXED_SMALL = enum.auto()
    FIXED_LARGE = enum.auto()
    LEARNED_RANGE = enum.auto()


class LossType(enum.Enum):
    MSE = enum.auto()
    RESCALED_MSE = enum.auto()
    KL = enum.auto()
    RESCALED_KL = enum.auto()


class ModelType(enum.Enum):
    DDPM = enum.auto()
    VPSDE = enum.auto()
    SUBVPSDE = enum.auto()


class NetworkType(enum.Enum):
    UNET = enum.auto()
    NCSN = enum.auto()


class NoiseSchedule(enum.Enum):
    LINEAR = enum.auto()
    COSINE = enum.auto()


def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    if schedule_name == NoiseSchedule.LINEAR:
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == NoiseSchedule.COSINE:
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


def from_flattened_numpy(x, shape):
    return torch.from_numpy(x.reshape(shape))


def to_flattened_numpy(x):
    return x.detach().cpu().numpy().reshape((-1,))


M = torch.tensor(
    [
        [5.7735014e-01, -8.1649649e-01, 4.7008697e-08],
        [5.7735026e-01, 4.0824834e-01, 7.0710671e-01],
        [5.7735026e-01, 4.0824822e-01, -7.0710683e-01],
    ]
)

invM = torch.inverse(M)


def decouple(inputs):
    return torch.einsum("bihw,ij->bjhw", inputs, M.to(inputs.device))


def couple(inputs):
    return torch.einsum("bihw,ij->bjhw", inputs, invM.to(inputs.device))


def get_colourisation_mask(image):
    mask = torch.cat(
        [torch.ones_like(image[:, :1, ...]), torch.zeros_like(image[:, 1:, ...])], dim=1
    )
    return mask


def space_timesteps(num_timesteps, section_counts):
    if isinstance(section_counts, str):
        if section_counts.startswith("ddim"):
            desired_count = int(section_counts[len("ddim") :])
            for i in range(1, num_timesteps):
                if len(range(0, num_timesteps, i)) == desired_count:
                    return set(range(0, num_timesteps, i))
            raise ValueError(
                f"cannot create exactly {num_timesteps} steps with an integer stride"
            )
        section_counts = [int(x) for x in section_counts.split(",")]
    size_per = num_timesteps // len(section_counts)
    extra = num_timesteps % len(section_counts)
    start_idx = 0
    all_steps = []
    for i, section_count in enumerate(section_counts):
        size = size_per + (1 if i < extra else 0)
        if size < section_count:
            raise ValueError(
                f"cannot divide section of {size} steps into {section_count}"
            )
        if section_count <= 1:
            frac_stride = 1
        else:
            frac_stride = (size - 1) / (section_count - 1)
        cur_idx = 0.0
        taken_steps = []
        for _ in range(section_count):
            taken_steps.append(start_idx + round(cur_idx))
            cur_idx += frac_stride
        all_steps += taken_steps
        start_idx += size
    return set(all_steps)


class Respacer(DDPMModel):
    def __init__(self, use_timesteps, **kwargs):
        self.use_timesteps = set(use_timesteps)
        self.timestep_map = []
        self.original_num_steps = len(kwargs["betas"])

        base_diffusion = DDPMModel(**kwargs)
        last_alpha_cumprod = 1.0
        new_betas = []
        for i, alpha_cumprod in enumerate(base_diffusion.alpha_bar):
            if i in self.use_timesteps:
                new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
                last_alpha_cumprod = alpha_cumprod
                self.timestep_map.append(i)
        kwargs["betas"] = np.array(new_betas)
        super().__init__(**kwargs)

    def p_xtm1_xt(self, model, *args, **kwargs):
        return super().p_xtm1_xt(self._wrap_model(model), *args, **kwargs)

    def training_losses(self, model, *args, **kwargs):
        return super().training_losses(self._wrap_model(model), *args, **kwargs)

    def _wrap_model(self, model):
        if isinstance(model, _WrappedModel):
            return model
        return _WrappedModel(
            model, self.timestep_map, self.rescale_timesteps, self.original_num_steps
        )

    def _scale_timesteps(self, t):
        return t


class _WrappedModel:
    def __init__(self, model, timestep_map, rescale_timesteps, original_num_steps):
        self.model = model
        self.timestep_map = timestep_map
        self.rescale_timesteps = rescale_timesteps
        self.original_num_steps = original_num_steps

    def __call__(self, x, ts, **kwargs):
        map_tensor = torch.tensor(self.timestep_map, device=ts.device, dtype=ts.dtype)
        new_ts = map_tensor[ts]
        if self.rescale_timesteps:
            new_ts = new_ts.float() * (1000.0 / self.original_num_steps)
        return self.model(x, new_ts, **kwargs)


def normal_kl(mean1, logvar1, mean2, logvar2):
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, torch.Tensor):
            tensor = obj
            break

    logvar1, logvar2 = [
        x if isinstance(x, torch.Tensor) else torch.tensor(x).to(tensor)
        for x in (logvar1, logvar2)
    ]

    return 0.5 * (
        -1.0
        + logvar2
        - logvar1
        + torch.exp(logvar1 - logvar2)
        + ((mean1 - mean2) ** 2) * torch.exp(-logvar2)
    )


def approx_standard_normal_cdf(x):
    return 0.5 * (
        1.0 + torch.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * torch.pow(x, 3)))
    )


def discretized_gaussian_log_likelihood(x, *, means, log_scales):
    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = approx_standard_normal_cdf(min_in)
    log_cdf_plus = torch.log(cdf_plus.clamp(min=1e-12))
    log_one_minus_cdf_min = torch.log((1.0 - cdf_min).clamp(min=1e-12))
    cdf_delta = cdf_plus - cdf_min
    log_probs = torch.where(
        x < -0.999,
        log_cdf_plus,
        torch.where(
            x > 0.999, log_one_minus_cdf_min, torch.log(cdf_delta.clamp(min=1e-12))
        ),
    )
    return log_probs
