import abc
import torch

from core.models.model_utils import M, couple, decouple, get_colourisation_mask


class Predictor(abc.ABC):
    def __init__(self, sde, probability_flow):
        super().__init__()
        self.sde = sde
        self.rsde = sde.reverse(probability_flow)

    @abc.abstractmethod
    def update_fn(self, network, x, t):
        pass

    def inpainting_update_fn(self, network, cond, mask, x, t):
        with torch.no_grad():
            vec_t = torch.ones(cond.shape[0], device=cond.device) * t
            x, x_mean = self.update_fn(network, x, vec_t)
            masked_data_mean, std = self.sde.marginal_prob(cond, vec_t)
            masked_data = (
                masked_data_mean + torch.randn_like(x) * std[:, None, None, None]
            )
            x = x * (mask) + masked_data * (1.0 - mask)
            x_mean = x * (mask) + masked_data_mean * (1.0 - mask)
            return x, x_mean

    def colorization_update_fn(self, network, gray_scale_img, x, t):
        with torch.no_grad():
            mask = get_colourisation_mask(x)
            vec_t = torch.ones(x.shape[0], device=x.device) * t
            x, x_mean = self.update_fn(network, x, vec_t)
            masked_data_mean, std = self.sde.marginal_prob(
                decouple(gray_scale_img), vec_t
            )
            masked_data = (
                masked_data_mean + torch.randn_like(x) * std[:, None, None, None]
            )
            x = couple(decouple(x) * (1.0 - mask) + masked_data * mask)
            x_mean = couple(decouple(x) * (1.0 - mask) + masked_data_mean * mask)
            return x, x_mean


class NonePredictor(Predictor):
    def __init__(self, sde, score_fn, probability_flow=False):
        pass

    def update_fn(self, x, t):
        return x, x
