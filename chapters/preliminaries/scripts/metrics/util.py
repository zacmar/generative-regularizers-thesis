import torch.nn as nn
import torch.nn.functional as tf
import torch as th
from typing import Union


class SSIM(nn.Module):
    def __init__(self, win_size: int = 7, k1: float = 0.01, k2: float = 0.03):
        """
        Args:
            win_size: Window size for SSIM calculation.
            k1: k1 parameter for SSIM calculation.
            k2: k2 parameter for SSIM calculation.
        """
        super().__init__()
        self.win_size = win_size
        self.k1, self.k2 = k1, k2
        self.register_buffer(
            "w",
            th.ones(1, 1, win_size, win_size) / win_size**2
        )
        NP = win_size**2
        self.cov_norm = NP / (NP - 1)
        self.w: th.Tensor

    def forward(
        self,
        X: th.Tensor,
        Y: th.Tensor,
        data_range: Union[th.Tensor, float] = 1.,
        reduced: bool = True,
    ):
        C1 = (self.k1 * data_range)**2
        C2 = (self.k2 * data_range)**2
        ux = tf.conv2d(X, self.w)
        uy = tf.conv2d(Y, self.w)
        uxx = tf.conv2d(X * X, self.w)
        uyy = tf.conv2d(Y * Y, self.w)
        uxy = tf.conv2d(X * Y, self.w)
        vx = self.cov_norm * (uxx - ux * ux)
        vy = self.cov_norm * (uyy - uy * uy)
        vxy = self.cov_norm * (uxy - ux * uy)
        A1, A2, B1, B2 = (
            2 * ux * uy + C1,
            2 * vxy + C2,
            ux**2 + uy**2 + C1,
            vx + vy + C2,
        )
        D = B1 * B2
        S = (A1 * A2) / D

        if reduced:
            return S.mean((1, 2, 3))
        else:
            return S
