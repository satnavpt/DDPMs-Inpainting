import numpy as np
import torch
from core.models.sdes.predictors.predictor import Predictor


class EulerMaruyamaPredictor(Predictor):
    def __init__(self, sde, probability_flow=False):
        super().__init__(sde, probability_flow)
        self.probability_flow = probability_flow

    def update_fn(self, network, x, t):
        dt = -1.0 / self.rsde.timesteps
        z = torch.randn_like(x)
        drift, diffusion = self.rsde.sde(network, x, t)
        x_mean = x + drift * dt
        if not self.probability_flow:
            x = x_mean + diffusion[:, None, None, None] * np.sqrt(-dt) * z
        else:
            x = x_mean
        return x, x_mean
