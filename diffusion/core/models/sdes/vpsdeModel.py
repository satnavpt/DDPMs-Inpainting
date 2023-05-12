from tqdm.auto import tqdm

import numpy as np
from scipy import integrate

import torch

from core.models.sdes.sde import SDE
from core.models.model_utils import (
    get_named_beta_schedule,
    from_flattened_numpy,
    to_flattened_numpy,
    couple,
    decouple,
    get_colourisation_mask,
)
from core.utils import ConditionType
from core.models.sdes.RK45Colour import RK45Colour
from core.models.sdes.RK45Inpaint import RK45Inpaint
from core.models.sdes.dpmsolver import NoiseScheduleVP, DPM_Solver
from core.dataset import inverse_scaler


class Model(SDE):
    def __init__(
        self,
        num_timesteps,
        noise_schedule,
        continuous,
        likelihood_weighting,
        eps,
        reduce_mean,
        ode,
        predictor,
        corrector,
        probability_flow,
        denoise,
        n_steps,
        snr,
        condition,
    ):
        super().__init__(num_timesteps)
        self.num_timesteps = num_timesteps
        self.discrete_beta = torch.Tensor(
            get_named_beta_schedule(noise_schedule, num_timesteps)
        )
        self.alpha = 1.0 - self.discrete_beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        self.root_alpha_bar = torch.sqrt(self.alpha_bar)
        self.root_1m_alpha_bar = torch.sqrt(1.0 - self.alpha_bar)
        self.beta_start = 0.1
        self.beta_end = 20
        self.continuous = continuous
        self.likelihood_weighting = likelihood_weighting
        self.eps = eps
        self.denoise = denoise
        self.n_steps = n_steps
        self.reduce_op = (
            torch.mean
            if reduce_mean
            else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)
        )
        self.ode = ode

        if not ode[0]:
            self.predictor = predictor(self, probability_flow)
            self.corrector = corrector(self, snr, self.n_steps)
        self.condition = condition

    @property
    def T(self):
        return 1

    def sde(self, x, t):
        beta_t = self.beta_start + t * (self.beta_end - self.beta_start)
        x = x
        drift = (-0.5 * beta_t[:, None, None, None] * x).to(dtype=torch.float32)
        diffusion = torch.sqrt(beta_t)
        return drift, diffusion

    def marginal_prob(self, x, t):
        log_mean_coeff = (
            -0.25 * t**2 * (self.beta_end - self.beta_start)
            - 0.5 * t * self.beta_start
        )
        mean = torch.exp(log_mean_coeff[:, None, None, None]) * x
        std = torch.sqrt(1.0 - torch.exp(2.0 * log_mean_coeff))
        return mean, std

    def prior_sampling(self, shape, dtype=torch.float32):
        return torch.randn(*shape, dtype=dtype)

    def prior_logp(self, z):
        shape = z.shape
        N = np.prod(shape[1:])
        logps = -N / 2.0 * np.log(2 * np.pi) - torch.sum(z**2, dim=(1, 2, 3)) / 2.0
        return logps

    def discretize(self, x, t):
        timestep = (t * (self.num_timesteps - 1) / self.T).long()
        beta = self.discrete_beta.to(x.device)[timestep]
        alpha = self.alpha.to(x.device)[timestep]
        sqrt_beta = torch.sqrt(beta)
        f = torch.sqrt(alpha)[:, None, None, None] * x - x
        G = sqrt_beta
        return f, G

    def noise_fn(self, network, x, t):
        if self.continuous:
            labels = t * 999
            noise = network(x, labels)
            return noise
        else:
            raise NotImplementedError()

    def score_fn(self, network, x, t):
        if self.continuous:
            score = self.noise_fn(network, x, t)
            std = self.marginal_prob(torch.zeros_like(x), t)[1]
        else:
            labels = t * (self.num_timesteps - 1)
            score = network(x, labels)
            std = self.root_1m_alpha_bar.to(labels.device)[labels.long()]

        score = -score / std[:, None, None, None]
        return score

    def training_losses(self, network, x0, t=None, y=None, mask=None):
        t = torch.rand(x0.shape[0], device=x0.device) * (self.T - self.eps) + self.eps
        z = torch.randn_like(x0)
        mean, std = self.marginal_prob(x0, t)
        perturbed_data = mean + std[:, None, None, None] * z
        score = self.score_fn(network=network, x=perturbed_data, t=t)

        if not self.likelihood_weighting:
            losses = torch.square(score * std[:, None, None, None] + z)
            losses = self.reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)
        else:
            g2 = self.sde(torch.zeros_like(x0), t)[1] ** 2
            losses = torch.square(score + z / std[:, None, None, None])
            losses = self.reduce_op(losses.reshape(losses.shape[0], -1), dim=-1) * g2

        return losses

    def likelihood_fn(
        self,
        network,
        x,
        y=None,
        hutchinson_type="Rademacher",
        rtol=1e-5,
        atol=1e-5,
        method="RK45",
        eps=1e-5,
    ):
        with torch.no_grad():
            shape = x.shape
            if hutchinson_type == "Gaussian":
                epsilon = torch.randn_like(x)
            elif hutchinson_type == "Rademacher":
                epsilon = torch.randint_like(x, low=0, high=2).float() * 2 - 1.0
            else:
                raise NotImplementedError(f"Hutchinson type {hutchinson_type} unknown.")

            def drift_fn(network, x, t):
                return self.reverse(probability_flow=True).sde(network, x, t)[0]

            def div_fn(network, x, t, noise):
                return get_div_fn(lambda xx, tt: drift_fn(network, xx, tt))(x, t, noise)

            def ode_func(t, xt):
                sample = (
                    from_flattened_numpy(xt[: -shape[0]], shape)
                    .to(x.device)
                    .type(torch.float32)
                )
                vec_t = torch.ones(sample.shape[0], device=sample.device) * t
                drift = to_flattened_numpy(drift_fn(network, sample, vec_t))
                logp_grad = to_flattened_numpy(div_fn(network, sample, vec_t, epsilon))
                return np.concatenate([drift, logp_grad], axis=0)

            init = np.concatenate(
                [to_flattened_numpy(x), np.zeros((shape[0],))], axis=0
            )
            solution = integrate.solve_ivp(
                ode_func, (eps, self.T), init, rtol=rtol, atol=atol, method=method
            )
            nfe = solution.nfev
            zp = solution.y[:, -1]
            z = (
                from_flattened_numpy(zp[: -shape[0]], shape)
                .to(x.device)
                .type(torch.float32)
            )
            delta_logp = (
                from_flattened_numpy(zp[-shape[0] :], (shape[0],))
                .to(x.device)
                .type(torch.float32)
            )
            prior_logp = self.prior_logp(z)
            bpd = -(prior_logp + delta_logp) / np.log(2)
            N = np.prod(shape[1:])
            bpd = bpd / N
            offset = 7.0
            bpd = bpd + offset
            return bpd, z, nfe

    def sample_fn(
        self,
        network=None,
        shape=None,
        clip_denoised=False,
        x0=None,
        y=None,
        mask=None,
        progress=False,
        device=None,
    ):
        if self.condition == ConditionType.none:
            with torch.no_grad():
                x = self.prior_sampling(shape).to(device)

                timesteps = torch.linspace(
                    self.T, self.eps, self.num_timesteps, device=device
                )

                for i in tqdm(
                    range(self.num_timesteps), desc="Sampling", disable=not progress
                ):
                    t = timesteps[i]
                    vec_t = torch.ones(shape[0], device=t.device) * t
                    x, x_mean = self.corrector.update_fn(network, x, vec_t)
                    x, x_mean = self.predictor.update_fn(network, x, vec_t)

                return inverse_scaler(x_mean if self.denoise else x)
        elif self.condition == ConditionType.colourisation:
            with torch.no_grad():
                shape = y.shape
                mask = get_colourisation_mask(y)
                x = couple(
                    decouple(y) * mask
                    + decouple(self.prior_sampling(shape).to(y.device) * (1.0 - mask))
                )

                timesteps = torch.linspace(self.T, self.eps, self.num_timesteps)
                for i in tqdm(
                    range(self.num_timesteps), desc="Sampling", disable=not progress
                ):
                    t = timesteps[i]
                    x, x_mean = self.corrector.colorization_update_fn(network, y, x, t)
                    x, x_mean = self.predictor.colorization_update_fn(network, y, x, t)

                return inverse_scaler(x_mean if self.denoise else x)
        elif self.condition == ConditionType.inpainting:
            with torch.no_grad():
                x = y
                timesteps = torch.linspace(self.T, self.eps, self.num_timesteps)
                for i in tqdm(
                    range(self.num_timesteps), desc="Sampling", disable=not progress
                ):
                    t = timesteps[i]
                    x, x_mean = self.corrector.inpainting_update_fn(
                        network, y, mask, x, t
                    )
                    x, x_mean = self.predictor.inpainting_update_fn(
                        network, y, mask, x, t
                    )
                return inverse_scaler(x_mean if self.denoise else x)

    def ode_sample_fn(
        self,
        network=None,
        shape=None,
        clip_denoised=None,
        x0=None,
        y=None,
        mask=None,
        device=None,
        rtol=1e-2,
        atol=1e-2,
        method="RK45",
        denoise=False,
        eps=1e-4,
        progress=None,
    ):
        if self.condition == ConditionType.none:

            def ode_func(t, x):
                x = from_flattened_numpy(x, shape).to(device).type(torch.float32)
                vec_t = torch.ones(shape[0], device=x.device) * t
                drift = self.drift_fn(network, x, vec_t)
                return to_flattened_numpy(drift)

            with torch.no_grad():
                x = self.prior_sampling(shape).to(device)
                solution = integrate.solve_ivp(
                    ode_func,
                    (self.T, eps),
                    to_flattened_numpy(x),
                    rtol=rtol,
                    atol=atol,
                    method=method,
                )
                nfe = solution.nfev
                x = (
                    torch.tensor(solution.y[:, -1])
                    .reshape(shape)
                    .to(device)
                    .type(torch.float32)
                )
                if denoise:
                    x = self.denoise_update_fn(x)
                print(f"nfe: {nfe}")
                return inverse_scaler(x)
        elif self.condition == ConditionType.colourisation:

            def ode_func(t, x, grey):
                x = from_flattened_numpy(x, shape).to(device).type(torch.float32)

                vec_t = torch.ones(shape[0], device=x.device) * t
                drift_x = self.drift_fn(network, x, vec_t)
                drift = couple(decouple(drift_x) * (1.0 - mask))

                return to_flattened_numpy(drift)

            with torch.no_grad():
                shape = y.shape
                mask = get_colourisation_mask(y)
                x = couple(
                    decouple(self.prior_sampling(shape).to(y.device) * (1.0 - mask))
                    + decouple(y) * mask
                )

                solution = integrate.solve_ivp(
                    ode_func,
                    (self.T, eps),
                    to_flattened_numpy(x),
                    rtol=rtol,
                    atol=atol,
                    grey=y,
                    shape=shape,
                    device=device,
                    sde=self,
                    method=RK45Colour,
                )

                nfe = solution.nfev
                x = (
                    torch.tensor(solution.y[:, -1])
                    .reshape(shape)
                    .to(device)
                    .type(torch.float32)
                )

                if denoise:
                    x = self.denoise_update_fn(x)
                return inverse_scaler(x)
        elif self.condition == ConditionType.inpainting:

            def ode_func(t, x, cond, mask):
                x = from_flattened_numpy(x, shape).to(device).type(torch.float32)

                vec_t = torch.ones(shape[0], device=x.device) * t
                drift_x = self.drift_fn(network, x, vec_t)
                drift = (drift_x) * (mask)

                return to_flattened_numpy(drift)

            with torch.no_grad():
                shape = y.shape
                x = y
                solution = integrate.solve_ivp(
                    ode_func,
                    (self.T, eps),
                    to_flattened_numpy(x),
                    rtol=rtol,
                    atol=atol,
                    cond=y,
                    mask=mask,
                    shape=shape,
                    device=device,
                    sde=self,
                    method=RK45Inpaint,
                )

                nfe = solution.nfev
                x = (
                    torch.tensor(solution.y[:, -1])
                    .reshape(shape)
                    .to(device)
                    .type(torch.float32)
                )

                if denoise:
                    x = self.denoise_update_fn(x)
                return inverse_scaler(x)

    def dpmsolver_sample_fn(
        self,
        network=None,
        shape=None,
        clip_denoised=None,
        x0=None,
        y=None,
        mask=None,
        device=None,
        rtol=1e-5,
        atol=1e-5,
        method="adaptive",
        denoise=False,
        eps=1e-3,
        progress=None,
        steps=10,
        order=3,
    ):
        network.eval()
        ns = NoiseScheduleVP(
            "linear", continuous_beta_0=self.beta_start, continuous_beta_1=self.beta_end
        )
        with torch.no_grad():
            if self.condition == ConditionType.none:
                dpm_solver = DPM_Solver(
                    network, self.noise_fn, ns, algorithm_type="dpmsolver"
                )
                x = self.prior_sampling(shape).to(device)
                x = dpm_solver.sample(
                    x,
                    steps=steps - 1 if denoise else steps,
                    t_start=1.0,
                    t_end=eps,
                    order=order,
                    skip_type="log_SNR",
                    method=method,
                    denoise_to_zero=denoise,
                    atol=atol,
                    rtol=rtol,
                    lower_order_final=False,
                )
                return inverse_scaler(x)
            elif self.condition == ConditionType.colourisation:

                def correcting_xt_fn(x, t, step):
                    mask = get_colourisation_mask(x)
                    vec_t = torch.ones(x.shape[0], device=x.device) * t

                    masked_data_mean, std = self.marginal_prob(decouple(y), vec_t)
                    masked_data = (
                        masked_data_mean
                        + torch.randn_like(x) * std[:, None, None, None]
                    )
                    x = couple(decouple(x) * (1.0 - mask) + masked_data * mask)
                    return x

                dpm_solver = DPM_Solver(
                    network,
                    self.noise_fn,
                    ns,
                    algorithm_type="dpmsolver",
                    correcting_xt_fn=correcting_xt_fn,
                )
                shape = y.shape
                mask = get_colourisation_mask(y)
                x = couple(
                    decouple(self.prior_sampling(shape).to(y.device) * (1.0 - mask))
                    + decouple(y) * mask
                )

                x = dpm_solver.sample(
                    x,
                    steps=steps - 1 if denoise else steps,
                    t_start=self.T,
                    t_end=eps,
                    order=order,
                    skip_type="logSNR",
                    method=method,
                    denoise_to_zero=denoise,
                    atol=atol,
                    rtol=rtol,
                    lower_order_final=False,
                )
                return inverse_scaler(x)
            elif self.condition == ConditionType.inpainting:

                def correcting_xt_fn(x, t, step):
                    vec_t = torch.ones(x.shape[0], device=x.device) * t
                    masked_data_mean, std = self.marginal_prob(y, vec_t)
                    masked_data = (
                        masked_data_mean
                        + torch.randn_like(x) * std[:, None, None, None]
                    )
                    x = (x) * (mask) + masked_data * (1.0 - mask)
                    return x

                dpm_solver = DPM_Solver(
                    network,
                    self.noise_fn,
                    ns,
                    algorithm_type="dpmsolver",
                    correcting_xt_fn=correcting_xt_fn,
                )
                shape = y.shape
                x = y

                x = dpm_solver.sample(
                    x,
                    steps=steps - 1 if denoise else steps,
                    t_start=self.T,
                    t_end=eps,
                    order=order,
                    skip_type="time_uniform",
                    method=method,
                    denoise_to_zero=denoise,
                    atol=atol,
                    rtol=rtol,
                    lower_order_final=False,
                )

                return inverse_scaler(x)


def get_div_fn(fn):
    def div_fn(x, t, eps):
        with torch.enable_grad():
            x.requires_grad_(True)
            fn_eps = torch.sum(fn(x, t) * eps)
            grad_fn_eps = torch.autograd.grad(fn_eps, x)[0]
        x.requires_grad_(False)
        return torch.sum(grad_fn_eps * eps, dim=tuple(range(1, len(x.shape))))

    return div_fn
