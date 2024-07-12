import io
import math
import os
from typing import Tuple, Union

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.functional as tf
import torchvision.transforms.functional as tvf


def psnr(
    x: th.Tensor,
    y: th.Tensor,
    value_range: Union[th.Tensor, float] = 1.
) -> th.Tensor:
    return 10 * th.log10(value_range**2 / mse(x, y))


def nmse(
    x: th.Tensor,
    y: th.Tensor,
) -> th.Tensor:
    return th.sum((x - y)**2, dim=(1, 2, 3),
                  keepdim=True) / th.sum(x**2, dim=(1, 2, 3), keepdim=True)


def mse(
    x: th.Tensor,
    y: th.Tensor,
) -> th.Tensor:
    return th.mean((x - y)**2, dim=(1, 2, 3), keepdim=True)


def rot180(x):
    return th.rot90(x, dims=(-2, -1), k=2)


def rss(x):
    return (th.abs(x)**2).sum(1, keepdim=True).sqrt()


def shift_interval(
    data: Union[np.ndarray, th.Tensor], from_: Tuple[float, float],
    to_: Tuple[float, float]
) -> Union[np.ndarray, th.Tensor]:
    return ((to_[1] - to_[0]) /
            (from_[1] - from_[0])) * (data - from_[0]) + to_[0]


def _get_gauss_kernel() -> th.Tensor:
    return th.tensor([[
        [1.0, 4.0, 6.0, 4.0, 1.0],
        [4.0, 16.0, 24.0, 16.0, 4.0],
        [6.0, 24.0, 36.0, 24.0, 6.0],
        [4.0, 16.0, 24.0, 16.0, 4.0],
        [1.0, 4.0, 6.0, 4.0, 1.0],
    ]]) / 256.0


def _compute_padding(kernel_size: list[int]) -> list[int]:
    computed = [k // 2 for k in kernel_size]
    out_padding = 2 * len(kernel_size) * [0]

    for i in range(len(kernel_size)):
        computed_tmp = computed[-(i + 1)]
        if kernel_size[i] % 2 == 0:
            padding = computed_tmp - 1
        else:
            padding = computed_tmp
        out_padding[2 * i + 0] = padding
        out_padding[2 * i + 1] = computed_tmp
    return out_padding


def cifft2(z: th.Tensor) -> th.Tensor:
    return th.fft.ifftshift(
        th.fft.ifft2(th.fft.fftshift(z, dim=(-2, -1)), norm='ortho'),
        dim=(-2, -1)
    )


def cfft2(x: th.Tensor) -> th.Tensor:
    return th.fft.fftshift(
        th.fft.fft2(th.fft.ifftshift(x, dim=(-2, -1)), norm='ortho'),
        dim=(-2, -1)
    )


def dct(x_: th.Tensor) -> th.Tensor:
    sh = x_.shape
    N = sh[-1]
    x = x_.contiguous().view(-1, sh[-1])
    temp = th.hstack((x[:, ::2], th.flip(x, (-1, ))[:, N % 2::2]))
    temp = th.fft.fft(temp)
    k = th.exp(-1j * np.pi * th.arange(N).to(x_) / (2 * N))
    return (temp * k).real.view(sh) / math.sqrt(N / 2)


def dct2(x: th.Tensor) -> th.Tensor:
    return dct(dct(x).permute(0, 1, 3, 2)).permute(0, 1, 3, 2)


def idct(x_: th.Tensor) -> th.Tensor:
    sh = x_.shape
    N = sh[-1]
    x = x_.contiguous().view(-1, N)
    factor = -1j * np.pi / (N * 2)
    temp = x * th.exp(th.arange(N).to(x_) * factor)[None]
    temp[:, 0] /= 2
    temp = th.fft.fft(temp).real

    result = th.empty_like(x)
    result[:, ::2] = temp[:, :(N + 1) // 2]
    indices = th.arange(-1 - N % 2, -N, -2)
    result[:, indices] = temp[:, (N + 1) // 2:]
    return result.view(sh) / math.sqrt(N / 2)


def idct2(x: th.Tensor) -> th.Tensor:
    return idct(idct(x).permute(0, 1, 3, 2)).permute(0, 1, 3, 2)


def dst(x):
    """
    https://arxiv.org/pdf/cs/0703150.pdf
    """
    m = th.ones((
        1,
        1,
        1,
        x.shape[-1],
    ), device=x.device)
    m[..., 1::2] = -1
    return dct(x * m).flip((-1, ))


def dst2(x):
    return dst(dst(x).permute(0, 1, 3, 2)).permute(0, 1, 3, 2)


def idst(y):
    m = th.ones((
        1,
        1,
        1,
        y.shape[-1],
    ), device=y.device)
    m[..., 1::2] = -1
    return m * idct(y.flip((-1, )))


def idst2(y):
    """
    https://arxiv.org/pdf/cs/0703150.pdf
    """
    return idst(idst(y).permute(0, 1, 3, 2)).permute(0, 1, 3, 2)


def view_as_real_batch(x):
    return th.cat([x.real, x.imag], 1)


def view_as_real_batch_T(x):
    n_ch = x.shape[1] // 2
    return x[:, :n_ch] + 1j * x[:, n_ch:]


def plot_to_numpy(fig):
    with io.BytesIO() as buff:
        fig.savefig(buff, format='raw')
        buff.seek(0)
        data = np.frombuffer(buff.getvalue(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    return data.reshape((int(h), int(w), -1))


def mri_crop(x):
    return tvf.center_crop(x, [320, 320])


def mri_crop_T(x, shape):
    pad_w = (shape[-1] - 320) // 2
    pad_h = (shape[-2] - 320) // 2
    return tf.pad(x, (pad_w, pad_w, pad_h, pad_h))


def espirit(kspace, radius):
    kspace = kspace.cpu().numpy().transpose(0, 2, 3, 1)
    cfl.writecfl('kspace.tmp', kspace)
    os.system(f"bart ecalib -c0 -m1 -r{radius} kspace.tmp sens.tmp")
    sens = cfl.readcfl('sens.tmp')
    os.system("rm kspace.tmp.cfl kspace.tmp.hdr sens.tmp.cfl sens.tmp.hdr")
    return th.from_numpy(sens.transpose(0, 3, 1, 2)).cuda()


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


class Grad(nn.Module):
    def __init__(self):
        super().__init__()

    def __matmul__(self, x):
        grad = x.new_zeros((*x.shape, 2))
        grad[:, :, :, :-1, 0] += x[:, :, :, 1:] - x[:, :, :, :-1]
        grad[:, :, :-1, :, 1] += x[:, :, 1:, :] - x[:, :, :-1, :]
        return grad

    def forward(self, x):
        return self @ x


class Div(nn.Module):
    def __init__(self):
        super().__init__()

    def __matmul__(self, x):
        div = x.new_zeros(x.shape[:-1])
        div[:, :, :, 1:] += x[:, :, :, :-1, 0]
        div[:, :, :, :-1] -= x[:, :, :, :-1, 0]
        div[:, :, 1:, :] += x[:, :, :-1, :, 1]
        div[:, :, :-1, :] -= x[:, :, :-1, :, 1]
        return div

    def forward(self, x):
        return self @ x


class AsMatrix():
    def __init__(
        self,
        operator,
        adjoint,
    ):
        self.operator = operator
        self.adjoint = adjoint

    def __matmul__(
        self,
        x,
    ):
        return self.operator(x)

    def forward(self, x):
        return self @ x

    def __call__(self, x):
        return self @ x

    @property
    def H(self, ):
        return AsMatrix(self.adjoint, self.operator)


def mahalanobis(x, mus, precs_chol):
    distances = th.empty((mus.shape[0], x.shape[0]), device=x.device)
    for k, (mu, L) in enumerate(zip(mus, precs_chol)):
        distances[k] = (((x - mu) @ L)**2).sum(1)
    return distances


def lognormal(x, mus, prec):
    # print(th.linalg.slogdet(prec)[0])
    prec_chol = th.linalg.cholesky(prec)
    maha_dist = mahalanobis(x, mus, prec_chol)
    logdets = th.linalg.slogdet(prec_chol)[1]
    return -(math.log(2 * math.pi) * mus.shape[1] +
             maha_dist) / 2 + logdets[:, None]


def patch2image(patches, img_dims, size=(2, 2), stride=(1, 1)):
    tmp = patches.view(-1, size[0] * size[1])[None].permute(0, 2, 1)
    return th.squeeze(F.fold(tmp, img_dims, size, 1, 0, stride))


def image2patch(image, size=(2, 2), stride=(1, 1)):
    return image.unfold(0, size[0], stride[0]).unfold(
        1, size[1], stride[1]
    ).contiguous().view(-1, size[0], size[1])


def proj_simplex(
    x: th.Tensor,  # 1-D vector of weights
    s: float = 1.,  # axis intersection
):
    k = th.linspace(1, len(x), len(x), device=x.device)
    x_s = th.sort(x, dim=0, descending=True)[0]
    t = (th.cumsum(x_s, dim=0) - s) / k
    mu = th.max(t)
    return th.clamp(x - mu, 0, s)


def proj_simplex_simul(
    x: th.Tensor,  # 2-D array of weights,
    # projection is performed along the last axis axis
    s: float = 1.,  # axis interesection
):
    K = x.shape[1]
    k = th.linspace(1, K, K, device=x.device)
    x_s = th.sort(x, dim=1, descending=True)[0]
    t = (th.cumsum(x_s, dim=1) - s) / k[None]
    mu = th.max(t, dim=1, keepdim=True)[0]
    return th.clamp(x - mu, 0, s)


def weight_init(
    vmin: float,
    vmax: float,
    n_w: int,
    scale: float,
    mode: str,
) -> th.Tensor:
    x = th.linspace(vmin, vmax, n_w, dtype=th.float32)
    match mode:
        case "constant":
            w = th.ones_like(x) * scale
        case "linear":
            w = x * scale
        case "quadratic":
            w = x**2 * scale
        case "abs":
            w = th.abs(x) * scale
            w -= w.max()
            w = w.abs()
        case "student-t":
            alpha = 100
            w = scale * math.sqrt(alpha) / (1 + 0.5 * alpha * x**2)
        case "Student-T":
            a_ = 0.1 * 78
            b_ = 0.1 * 78**2
            denom = 1 + (a_ * x)**2
            w = b_ / (2 * a_**2) * th.log(denom)

    return w


def _get_gauss_kernel() -> th.Tensor:
    return th.tensor([[
        [1.0, 4.0, 6.0, 4.0, 1.0],
        [4.0, 16.0, 24.0, 16.0, 4.0],
        [6.0, 24.0, 36.0, 24.0, 6.0],
        [4.0, 16.0, 24.0, 16.0, 4.0],
        [1.0, 4.0, 6.0, 4.0, 1.0],
    ]]) / 256.0


def _compute_padding(kernel_size: list[int]) -> list[int]:
    computed = [k // 2 for k in kernel_size]
    out_padding = 2 * len(kernel_size) * [0]

    for i in range(len(kernel_size)):
        computed_tmp = computed[-(i + 1)]
        if kernel_size[i] % 2 == 0:
            padding = computed_tmp - 1
        else:
            padding = computed_tmp
        out_padding[2 * i + 0] = padding
        out_padding[2 * i + 1] = computed_tmp
    return out_padding


def init_params(mode, vmin, vmax, n_f, n_w, symmetric):
    x = th.linspace(vmin, vmax, n_w, dtype=th.float32)
    match mode:
        case "constant":
            w = th.ones_like(x)
        case "linear":
            w = x
        case "quadratic":
            w = -x**2
            w -= w.min()
        case "abs":
            w = -th.abs(x)
            w -= w.min()
        case "student-t":
            alpha = 1000
            w = math.sqrt(alpha) / (1 + 0.5 * alpha * x**2)
        case "Student-T":
            a_ = 0.1 * 78
            b_ = 0.1 * 78**2
            denom = 1 + (a_ * x)**2
            w = b_ / (2 * a_**2) * th.log(denom)
        case 'random':
            w = th.rand((n_w, ))
    w /= w.sum()

    if symmetric:
        w = w[w.shape[0] // 2:][None].repeat(n_f, 1).clone()
    else:
        w = th.nn.Parameter(w[None].repeat(n_f, 1).clone())
    return w


class SSIM(th.nn.Module):
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
        data_range: th.Tensor | float = 1.,
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

        return S.mean((1, 2, 3))
