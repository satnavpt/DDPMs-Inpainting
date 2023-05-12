import torch

from core.models.sdes.correctors.corrector import Corrector


class LangevinCorrector(Corrector):
    def __init__(self, sde, snr, n_steps):
        super().__init__(sde, snr, n_steps)

    def update_fn(self, network, x, t):
        sde = self.sde
        n_steps = self.n_steps
        target_snr = self.snr
        timestep = (t * (sde.timesteps - 1) / sde.T).long()
        alpha = sde.alpha.to(t.device)[timestep]

        for _ in range(n_steps):
            grad = self.sde.score_fn(network, x, t)
            noise = torch.randn_like(x)
            grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
            noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
            step_size = (target_snr * noise_norm / grad_norm) ** 2 * 2 * alpha
            x_mean = x + step_size[:, None, None, None] * grad
            x = x_mean + torch.sqrt(step_size * 2)[:, None, None, None] * noise

        return x, x_mean
