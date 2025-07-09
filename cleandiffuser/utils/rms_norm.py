import torch
import torch.nn as nn
from einops import rearrange

class RMSNorm(torch.nn.Module):
    def __init__(self, dim, eps: float = 1e-6, weight=False):
        super().__init__()
        self.eps = eps
        if weight:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.weight=None

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        if self.weight is None:
            return output
        else:
            return output * self.weight
        


if __name__ == "__main__":
    x = torch.randn(2, 4, 128)
    rms = RMSNorm(dim=(2, 4, 128))
    y = rms(x)
    print(x.shape)
    print(y.shape)