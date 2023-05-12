import torch

from core.models.sdes.correctors.corrector import Corrector


class AnnealedLangevinDynamics(Corrector):
    def __init__(self, sde, snr, n_steps):
        super().__init__(sde, snr, n_steps)

    def update_fn(self, network, x, t):
        sde = self.sde
        score_fn = sde.score_fn
        n_steps = self.n_steps
        target_snr = self.snr
        timestep = (t * (sde.timesteps - 1) / sde.T).long()
        alpha = sde.alpha.to(t.device)[timestep]

        std = self.sde.marginal_prob(x, t)[1]

        for _ in range(n_steps):
            grad = score_fn(network, x, t)
            noise = torch.randn_like(x)
            step_size = (target_snr * std) ** 2 * 2 * alpha
            x_mean = x + step_size[:, None, None, None] * grad
            x = x_mean + noise * torch.sqrt(step_size * 2)[:, None, None, None]

        return x, x_mean
