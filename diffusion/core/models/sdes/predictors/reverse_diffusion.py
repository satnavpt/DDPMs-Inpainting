import torch
from core.models.sdes.predictors.predictor import Predictor


class ReverseDiffusionPredictor(Predictor):
    def __init__(self, sde, probability_flow=False):
        super().__init__(sde, probability_flow)

    def update_fn(self, network, x, t):
        f, G = self.rsde.discretize(network, x, t)
        z = torch.randn_like(x)
        x_mean = x - f
        x = x_mean + G[:, None, None, None] * z
        return x, x_mean
