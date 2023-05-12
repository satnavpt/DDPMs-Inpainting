import numpy as np
import torch

from core.networks.blocks import mean_flat
from core.models.model_utils import (
    MeanType,
    VarType,
    LossType,
    normal_kl,
    discretized_gaussian_log_likelihood,
)


class Model:
    def __init__(
        self,
        *,
        beta,
        mean_type,
        var_type,
        loss_type,
        rescale_timesteps,
    ):
        self.mean_type = mean_type
        self.var_type = var_type
        self.loss_type = loss_type
        self.rescale_timesteps = rescale_timesteps

        beta = np.array(beta, dtype=np.float64)
        self.beta = beta
        self.num_timesteps = int(beta.shape[0])

        alpha = 1.0 - beta
        self.alpha_bar = np.cumprod(alpha, axis=0)
        self.alpha_bar_prev = np.append(1.0, self.alpha_bar[:-1])
        self.alpha_bar_next = np.append(self.alpha_bar[1:], 0.0)

        self.root_alpha_bar = np.sqrt(self.alpha_bar)
        self.root_1m_alpha_bar = np.sqrt(1.0 - self.alpha_bar)
        self.log_1m_alpha_bar = np.log(1.0 - self.alpha_bar)
        self.root_1d_alpha_bar = np.sqrt(1.0 / self.alpha_bar)
        self.root_1d1m_alpha_bar = np.sqrt(1.0 / self.alpha_bar - 1)

        self.posterior_variance = (
            beta * (1.0 - self.alpha_bar_prev) / (1.0 - self.alpha_bar)
        )
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
            beta * np.sqrt(self.alpha_bar_prev) / (1.0 - self.alpha_bar)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alpha_bar_prev) * np.sqrt(alpha) / (1.0 - self.alpha_bar)
        )

    def q_xt_x0(self, x0, t):
        mean = _extract(self.root_alpha_bar, t, x0.shape) * x0
        variance = _extract(1.0 - self.alpha_bar, t, x0.shape)
        log_variance = _extract(self.log_1m_alpha_bar, t, x0.shape)
        return mean, variance, log_variance

    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        return (
            _extract(self.root_alpha_bar, t, x0.shape) * x0
            + _extract(self.root_1m_alpha_bar, t, x0.shape) * noise
        )

    def q_xtm1_xt_x0(self, x0, xt, t):
        posterior_mean = (
            _extract(self.posterior_mean_coef1, t, xt.shape) * x0
            + _extract(self.posterior_mean_coef2, t, xt.shape) * xt
        )
        posterior_variance = _extract(self.posterior_variance, t, xt.shape)
        posterior_log_variance_clipped = _extract(
            self.posterior_log_variance_clipped, t, xt.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_xtm1_xt(self, network, xt, t, clip_denoised=True, denoised_fn=None, y=None):
        B, C = xt.shape[:2]

        if y is not None:
            inp = torch.cat([xt, y], dim=1)
        else:
            inp = xt

        network_output = network(inp, self._scale_timesteps(t))

        if self.var_type in [VarType.LEARNED, VarType.LEARNED_RANGE]:
            network_output, network_var_values = torch.split(network_output, C, dim=1)
            if self.var_type == VarType.LEARNED:
                network_log_variance = network_var_values
                network_variance = torch.exp(network_log_variance)
            else:
                min_log = _extract(self.posterior_log_variance_clipped, t, xt.shape)
                max_log = _extract(np.log(self.beta), t, xt.shape)
                frac = (network_var_values + 1) / 2
                network_log_variance = frac * max_log + (1 - frac) * min_log
                network_variance = torch.exp(network_log_variance)
        else:
            network_variance, network_log_variance = {
                VarType.FIXED_LARGE: (
                    np.append(self.posterior_variance[1], self.beta[1:]),
                    np.log(np.append(self.posterior_variance[1], self.beta[1:])),
                ),
                VarType.FIXED_SMALL: (
                    self.posterior_variance,
                    self.posterior_log_variance_clipped,
                ),
            }[self.var_type]
            network_variance = _extract(network_variance, t, xt.shape)
            network_log_variance = _extract(network_log_variance, t, xt.shape)

        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        if self.mean_type == MeanType.PREVIOUS_X:
            pred_xstart = process_xstart(
                self._predict_x0_from_xtm1(xt=xt, t=t, xtm1=network_output)
            )
            network_mean = network_output
        elif self.mean_type in [MeanType.START_X, MeanType.EPSILON]:
            if self.mean_type == MeanType.START_X:
                pred_xstart = process_xstart(network_output)
            else:
                pred_xstart = process_xstart(
                    self._predict_x0_from_eps(xt=xt, t=t, eps=network_output)
                )
            network_mean, _, _ = self.q_xtm1_xt_x0(x0=pred_xstart, xt=xt, t=t)
        else:
            raise NotImplementedError(self.mean_type)

        return {
            "mean": network_mean,
            "variance": network_variance,
            "log_variance": network_log_variance,
            "pred_xstart": pred_xstart,
        }

    def _predict_x0_from_eps(self, xt, t, eps):
        return (
            _extract(self.root_1d_alpha_bar, t, xt.shape) * xt
            - _extract(self.root_1d1m_alpha_bar, t, xt.shape) * eps
        )

    def _predict_x0_from_xtm1(self, xt, t, xtm1):
        return (
            _extract(1.0 / self.posterior_mean_coef1, t, xt.shape) * xtm1
            - _extract(
                self.posterior_mean_coef2 / self.posterior_mean_coef1, t, xt.shape
            )
            * xt
        )

    def _predict_eps_from_x0(self, xt, t, pred_xstart):
        return (
            _extract(self.root_1d_alpha_bar, t, xt.shape) * xt - pred_xstart
        ) / _extract(self.root_1d1m_alpha_bar, t, xt.shape)

    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t

    def p_sample(self, network, xt, t, clip_denoised=True, denoised_fn=None, y=None):
        out = self.p_xtm1_xt(
            network,
            xt,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            y=y,
        )
        noise = torch.randn_like(xt)
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(xt.shape) - 1)))
        sample = (
            out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise
        )
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    @torch.inference_mode()
    def sample_fn(
        self,
        network,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        x0=None,
        y=None,
        mask=None,
        device=None,
        progress=False,
    ):
        final = None
        for sample in self.p_sample_loop_progressive(
            network,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            x0=x0,
            y=y,
            mask=mask,
            device=device,
            progress=progress,
        ):
            final = sample
        s = final["sample"]
        return s

    def p_sample_loop_progressive(
        self,
        network,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        x0=None,
        y=None,
        mask=None,
        device=None,
        progress=False,
    ):
        if device is None:
            device = next(network.parameters()).device
        if noise is not None:
            img = noise
        else:
            img = torch.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            from tqdm.auto import tqdm

            indices = tqdm(indices, desc="Sampling: ")

        for i in indices:
            t = torch.tensor([i] * shape[0], device=device)
            with torch.no_grad():
                out = self.p_sample(
                    network,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    y=y,
                )
                if mask is not None:
                    out["sample"] = (x0 * (1.0 - mask)) + (mask * out["sample"])
                yield out
                img = out["sample"]

    def ddim_sample(
        self,
        network,
        xt,
        t,
        clip_denoised=True,
        denoised_fn=None,
        y=None,
        eta=0.0,
    ):
        out = self.p_xtm1_xt(
            network,
            xt,
            t,
            clip_denoised=clip_denoised,
            y=y,
            denoised_fn=denoised_fn,
        )
        eps = self._predict_eps_from_x0(xt, t, out["pred_xstart"])
        alpha_bar = _extract(self.alpha_bar, t, xt.shape)
        alpha_bar_prev = _extract(self.alpha_bar_prev, t, xt.shape)
        sigma = (
            eta
            * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
            * torch.sqrt(1 - alpha_bar / alpha_bar_prev)
        )
        noise = torch.randn_like(xt)
        mean_pred = (
            out["pred_xstart"] * torch.sqrt(alpha_bar_prev)
            + torch.sqrt(1 - alpha_bar_prev - sigma**2) * eps
        )
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(xt.shape) - 1)))
        sample = mean_pred + nonzero_mask * sigma * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def ddim_sample_fn(
        self,
        network,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        x0=None,
        y=None,
        mask=None,
        device=None,
        progress=False,
        eta=0.0,
    ):
        final = None
        for sample in self.ddim_sample_loop_progressive(
            network,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            x0=x0,
            y=y,
            mask=mask,
            device=device,
            progress=progress,
            eta=eta,
        ):
            final = sample
        return final["sample"]

    def ddim_sample_loop_progressive(
        self,
        network,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        x0=None,
        y=None,
        mask=None,
        device=None,
        progress=False,
        eta=0.0,
    ):
        if device is None:
            device = next(network.parameters()).device
        if noise is not None:
            img = noise
        else:
            img = torch.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:
            t = torch.tensor([i] * shape[0], device=device)
            with torch.no_grad():
                out = self.ddim_sample(
                    network,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    y=y,
                    eta=eta,
                )
                if mask is not None:
                    out["sample"] = (x0 * (1.0 - mask)) + (mask * out["sample"])
                yield out
                img = out["sample"]

    def _vb_terms_bpd(self, network, x0, xt, t, clip_denoised=True, y=None):
        true_mean, _, true_log_variance_clipped = self.q_xtm1_xt_x0(x0=x0, xt=xt, t=t)
        out = self.p_xtm1_xt(network, xt, t, clip_denoised=clip_denoised, y=y)
        kl = normal_kl(
            true_mean, true_log_variance_clipped, out["mean"], out["log_variance"]
        )
        kl = mean_flat(kl) / np.log(2.0)

        decoder_nll = -discretized_gaussian_log_likelihood(
            x0, means=out["mean"], log_scales=0.5 * out["log_variance"]
        )
        decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

        output = torch.where((t == 0), decoder_nll, kl)
        return {"output": output, "pred_xstart": out["pred_xstart"]}

    def training_losses(self, network, x0, t, y=None, mask=None, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        xt = self.q_sample(x0, t, noise=noise)

        terms = {}

        if self.loss_type == LossType.KL or self.loss_type == LossType.RESCALED_KL:
            terms["loss"] = self._vb_terms_bpd(
                network=network,
                x0=x0,
                xt=xt,
                t=t,
                clip_denoised=False,
                y=y,
            )["output"]
            if self.loss_type == LossType.RESCALED_KL:
                terms["loss"] *= self.num_timesteps
        elif self.loss_type == LossType.MSE or self.loss_type == LossType.RESCALED_MSE:
            if y is not None:
                if mask is not None:
                    inp = torch.cat([xt * mask + (1.0 - mask) * x0, y], dim=1)
                else:
                    inp = torch.cat([xt, y], dim=1)
            else:
                inp = xt

            network_output = network(inp, self._scale_timesteps(t))

            if self.var_type in [
                VarType.LEARNED,
                VarType.LEARNED_RANGE,
            ]:
                B, C = xt.shape[:2]
                network_output, network_var_values = torch.split(
                    network_output, C, dim=1
                )
                # Learn the variance using the variational bound, but don't let
                # it affect our mean prediction.
                frozen_out = torch.cat(
                    [network_output.detach(), network_var_values], dim=1
                )
                terms["vb"] = self._vb_terms_bpd(
                    network=lambda *args, r=frozen_out: r,
                    x0=x0,
                    xt=xt,
                    t=t,
                    clip_denoised=False,
                    y=y,
                )["output"]
                if self.loss_type == LossType.RESCALED_MSE:
                    terms["vb"] *= self.num_timesteps / 1000.0

            target = {
                MeanType.PREVIOUS_X: self.q_xtm1_xt_x0(x0=x0, xt=xt, t=t)[0],
                MeanType.START_X: x0,
                MeanType.EPSILON: noise,
            }[self.mean_type]
            if mask is not None:
                terms["mse"] = mean_flat(
                    ((mask * target) - (mask * network_output)) ** 2
                )
            else:
                terms["mse"] = mean_flat((target - network_output) ** 2)
            if "vb" in terms:
                terms["loss"] = terms["mse"] + terms["vb"]
            else:
                terms["loss"] = terms["mse"]
        else:
            raise NotImplementedError(self.loss_type)

        return terms

    def _prior_bpd(self, x0):
        batch_size = x0.shape[0]
        t = torch.tensor([self.num_timesteps - 1] * batch_size, device=x0.device)
        qt_mean, _, qt_log_variance = self.q_xt_x0(x0, t)
        kl_prior = normal_kl(
            mean1=qt_mean, logvar1=qt_log_variance, mean2=0.0, logvar2=0.0
        )
        return mean_flat(kl_prior) / np.log(2.0)

    def calc_bpd_loop(self, network, x0, clip_denoised=True, y=None):
        device = x0.device
        batch_size = x0.shape[0]

        vb = []
        xstart_mse = []
        mse = []
        for t in list(range(self.num_timesteps))[::-1]:
            t_batch = torch.tensor([t] * batch_size, device=device)
            noise = torch.randn_like(x0)
            x_t = self.q_sample(x0=x0, t=t_batch, noise=noise)
            with torch.no_grad():
                out = self._vb_terms_bpd(
                    network,
                    x0=x0,
                    xt=x_t,
                    t=t_batch,
                    clip_denoised=clip_denoised,
                    y=y,
                )
            vb.append(out["output"])
            xstart_mse.append(mean_flat((out["pred_xstart"] - x0) ** 2))
            eps = self._predict_eps_from_x0(x_t, t_batch, out["pred_xstart"])
            mse.append(mean_flat((eps - noise) ** 2))

        vb = torch.stack(vb, dim=1)
        xstart_mse = torch.stack(xstart_mse, dim=1)
        mse = torch.stack(mse, dim=1)

        prior_bpd = self._prior_bpd(x0)
        total_bpd = vb.sum(dim=1) + prior_bpd
        return {
            "total_bpd": total_bpd,
            "prior_bpd": prior_bpd,
            "vb": vb,
            "xstart_mse": xstart_mse,
            "mse": mse,
        }


def _extract(arr, timesteps, broadcast_shape):
    res = torch.from_numpy(arr).float().to(device=timesteps.device)[timesteps]
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)
