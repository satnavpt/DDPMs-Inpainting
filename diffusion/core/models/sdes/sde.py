import abc
import torch

from core.models.sdes.predictors.reverse_diffusion import ReverseDiffusionPredictor


class SDE(abc.ABC):
    def __init__(self, timesteps):
        super().__init__()
        self.timesteps = timesteps

    @property
    @abc.abstractmethod
    def T(self):
        pass

    @abc.abstractmethod
    def sde(self, x, t):
        pass

    @abc.abstractmethod
    def marginal_prob(self, x, t):
        pass

    @abc.abstractmethod
    def prior_sampling(self, shape):
        pass

    @abc.abstractmethod
    def prior_logp(self, z):
        pass

    def discretize(self, x, t):
        dt = 1 / self.timesteps
        drift, diffusion = self.sde(x, t)
        F = drift * dt
        G = diffusion * torch.sqrt(torch.tensor(dt, device=t.device))
        return F, G

    @abc.abstractmethod
    def score_fn(self, network, x, t):
        pass

    def reverse(self, probability_flow):
        timesteps = self.timesteps
        T = self.T
        sde_fn = self.sde
        discretize_fn = self.discretize
        score_fn = self.score_fn

        class RSDE(self.__class__):
            def __init__(self):
                self.timesteps = timesteps
                self.probability_flow = probability_flow

            @property
            def T(self):
                return T

            def sde(self, network, x, t):
                drift, diffusion = sde_fn(x, t)
                score = score_fn(network, x, t)
                drift = (
                    drift
                    - diffusion[:, None, None, None] ** 2
                    * score
                    * (0.5 if self.probability_flow else 1.0)
                ).to(torch.float32)
                diffusion = 0.0 if self.probability_flow else diffusion
                return drift, diffusion

            def discretize(self, network, x, t):
                F, G = discretize_fn(x, t)
                rev_F = F - G[:, None, None, None] ** 2 * score_fn(network, x, t) * (
                    0.5 if self.probability_flow else 1.0
                )
                rev_G = torch.zeros_like(G) if self.probability_flow else G
                return rev_F, rev_G

        return RSDE()

    def denoise_update_fn(self, x, eps=1e-3):
        predictor = ReverseDiffusionPredictor(
            self.sde, self.score_fn, probability_flow=False
        )
        epsV = torch.ones(x.shape[0], device=x.device) * eps
        _, x = predictor.update_fn(x, epsV)
        return x

    def drift_fn(self, network, x, t):
        rsde = self.reverse(probability_flow=True)
        return rsde.sde(network, x, t)[0]
