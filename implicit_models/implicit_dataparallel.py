import torch.nn as nn


class ImplicitDataparallel(nn.DataParallel):
    def compute_loss(self, *args, **kwargs):
        return self.module.compute_loss(*args, **kwargs)

    @property
    def temperature(self):
        return self.module.temperature
