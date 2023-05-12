import torch

from core.models.sdes.predictors.predictor import Predictor


class AncestralSamplingPredictor(Predictor):
    """The ancestral sampling predictor. Currently only supports VE/VP SDEs."""

    def __init__(self, sde, score_fn, probability_flow=False):
        super().__init__(sde, score_fn, probability_flow)

    def vesde_update_fn(self, network, x, t):
        sde = self.sde
        timestep = (t * (sde.N - 1) / sde.T).long()
        sigma = sde.discrete_sigmas[timestep]
        adjacent_sigma = torch.where(
            timestep == 0,
            torch.zeros_like(t),
            sde.discrete_sigmas.to(t.device)[timestep - 1],
        )
        score = self.score_fn(network, x, t)
        x_mean = x + score * (sigma**2 - adjacent_sigma**2)[:, None, None, None]
        std = torch.sqrt(
            (adjacent_sigma**2 * (sigma**2 - adjacent_sigma**2)) / (sigma**2)
        )
        noise = torch.randn_like(x)
        x = x_mean + std[:, None, None, None] * noise
        return x, x_mean

    def vpsde_update_fn(self, network, x, t):
        sde = self.sde
        timestep = (t * (sde.N - 1) / sde.T).long()
        beta = sde.discrete_betas.to(t.device)[timestep]
        score = self.score_fn(network, x, t)
        x_mean = (x + beta[:, None, None, None] * score) / torch.sqrt(1.0 - beta)[
            :, None, None, None
        ]
        noise = torch.randn_like(x)
        x = x_mean + torch.sqrt(beta)[:, None, None, None] * noise
        return x, x_mean

    def update_fn(self, network, x, t):
        return self.vpsde_update_fn(network, x, t)
