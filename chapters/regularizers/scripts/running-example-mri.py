import json
import matplotlib.pyplot as plt
from typing import Callable

import imageio.v3 as iio
import models
import nets
import numpy as np
import shltutils.filters as filters
import torch as th
import utils


def mr_model():
    config_path = './config.json'
    with open(config_path) as file:
        config = json.load(file)
    imsize = 320
    R = nets.EnergyNetMR(
        f_mul=config["f_mul"],
        n_c=config["im_ch"],
        n_f=config["n_f"],
        imsize=imsize,
        n_stages=config["stages"],
    )
    R.load_state_dict(th.load('ebm.ckpt'))
    return R.cuda()


def grad(x):
    g = x.new_zeros((*x.shape, 2))
    g[..., :, :-1, 0] += x[..., :, 1:] - x[..., :, :-1]
    g[..., :-1, :, 1] += x[..., 1:, :] - x[..., :-1, :]
    return g


def div(x):
    div = x.new_zeros(x.shape[:-1])
    div[..., :, 1:] += x[..., :, :-1, 0]
    div[..., :, :-1] -= x[..., :, :-1, 0]
    div[..., 1:, :] += x[..., :-1, :, 1]
    div[..., :-1, :] -= x[..., :-1, :, 1]
    return div


class LinearOperator:
    def __init__(self, forward, adjoint, scalar=1.):
        self.forward = forward
        self.adjoint = adjoint
        self.scalar = scalar

    @property
    def T(self):
        return LinearOperator(self.adjoint, self.forward)

    def __matmul__(self, x):
        return self.scalar * self.forward(x)

    def __rmul__(self, s):
        return LinearOperator(self.forward, self.adjoint, s)


def ipiano(x0, nabla_f, prox_g, alpha, beta, x_gt, max_iter=20000):
    x = x0.clone()
    L = 1
    x_prev = x.clone()
    for _ in range(max_iter):
        e, g = nabla_f(x)
        for _ in range(20):
            alpha = (2 * (1 - beta) / L) * .9
            x_proposal = prox_g(x - alpha * g + beta * (x - x_prev), alpha)
            dx = x_proposal - x
            bound = e + (g * dx).sum() + L / 2 * (dx * dx).sum()

            if nabla_f(x_proposal)[0] < bound:
                x_prev = x.clone()
                x = x_proposal
                break
            L = 2 * L
        else:
            break
        L /= 1.5
    return x


def apgd(
    x_init: th.Tensor,
    f_nabla,
    prox: Callable[[th.Tensor, th.Tensor], th.Tensor],
    callback: Callable[[th.Tensor, int], None] = lambda x, i: None,
    max_iter: int = 2000,
    gamma=0,
):
    x = x_init.clone()
    x_old = x.clone()
    L = 1
    beta = 1 / np.sqrt(2)
    for i in range(max_iter):
        x_bar = x + beta * (x - x_old)
        x_old = x.clone()
        n = th.randn_like(x) * 7.5e-3 * (i < 1000) * gamma
        energy, grad = f_nabla(x_bar + n)
        for _ in range(20):
            x = prox(x_bar - grad / L, 1 / L)
            dx = x - x_bar
            bound = energy + (grad * dx).sum() + L * (dx * dx).sum() / 2
            if f_nabla(x + n)[0] < bound:
                break
            L = 2 * L
        else:
            break
        L /= 1.5
        callback(x, i)
    return x


def accelerated_gradient(
    x0, nablaf, tau, max_iter=200, callback=lambda x, k: None
):
    x = x0.clone()
    x_prev = x.clone()
    alpha = 1.
    for k in range(max_iter):
        alpha_prev = alpha
        alpha = (1 + (1 + 4 * alpha_prev**2)**0.5) / 2
        beta = (alpha_prev - 1) / alpha
        x_tilde = x + beta * (x - x_prev)
        x_prev = x.clone()
        x = x_tilde - tau * nablaf(x_tilde)
        callback(x, k)
    return x


def gradient_descent(x0, nablaf, tau, max_iter=20000):
    x = x0.clone()
    for _ in range(max_iter):
        x = x - tau * nablaf(x)
    return x


def pdhg(
    x0: th.Tensor,
    K: LinearOperator,
    tau: float,
    prox_tG: Callable[[th.Tensor], th.Tensor],
    sigma: float,
    prox_sFstar: Callable[[th.Tensor], th.Tensor],
    max_iter: int = 20000,
) -> th.Tensor:
    x = x0.clone()
    y = K @ x0

    for _ in range(max_iter):
        x_prev = x.clone()
        x = prox_tG(x - tau * K.T @ y)
        y = prox_sFstar(y + sigma * K @ (2 * x - x_prev))

    return x


def fft2(x: th.Tensor):
    return th.fft.rfft2(x, norm='ortho')


def ifft2(x: th.Tensor):
    return th.fft.irfft2(x, norm='ortho')


mask = th.fft.fftshift(th.load('mask.pt')).cuda()[:, :161]
x = th.load('reference-rss.pt').cuda()
dx = grad(x).abs().sum(-1)
iio.imwrite('reference-rss.png', (x * 255.).to(th.uint8).cpu().numpy())
iio.imwrite('dx.png', (dx * 255.).to(th.uint8).cpu().numpy())
n = th.load('noise.pt')

sigma_n = 2e-2

y = mask * (fft2(x) + sigma_n * n)
iio.imwrite(
    'log-abs-data.png', (th.log(y.abs()) * 255.).to(th.uint8).cpu().numpy()
)

zero_filled = ifft2(y)


def nabla_l2norm(x):
    return ifft2(mask * (mask * fft2(x) - y)) + lamda * x


lamda = .1e0
# Lipschitz of gradient is (1 + lamda)
# Max step size is 2 / L
# quadratic_intensities = accelerated_gradient(
#     zero_filled, nabla_l2norm, 2 / (1 + lamda), max_iter=20000)
# But we can just directly compute the solution
quadratic_intensities = ifft2(y / (1 + lamda))

D = LinearOperator(grad, div)
L_D = np.sqrt(8)
lamda = 3e0


def nabla_l2gradientnorm(x):
    return ifft2(mask * (mask * fft2(x) - y)) + lamda * D.T @ (D @ x)


quadratic_gradients = accelerated_gradient(
    zero_filled, nabla_l2gradientnorm, 1 / (L_D**2 * lamda)
)


def prox_g(x: th.Tensor):
    return ifft2((fft2(x) + tau * y) / (1 + tau * mask))


def prox_linftyinfty(p: th.Tensor):
    return p.clamp(-lamda, lamda)


def prox_ltwoinfty(p: th.Tensor):
    norm = (p**2).sum(-1, keepdim=True).sqrt()
    return p / th.max(th.ones_like(norm), norm / lamda)


lamda = 8e-3
tau = 1 / L_D
sigma = 1 / L_D
# absolute_gradients = pdhg(zero_filled, D, tau, prox_g, sigma, prox_linftyinfty)
# isotropic_tv = pdhg(zero_filled, D, tau, prox_g, sigma, prox_ltwoinfty)

lamda = 8e-6
kernel_size = 7
M, N = 320, 320
div7 = utils.patch2image(
    utils.image2patch(th.ones((M, N)).cuda(), (kernel_size, kernel_size)),
    (M, N), (kernel_size, kernel_size)
)
this = th.rand((320, 320))
other = th.rand_like(utils.image2patch(this, (7, 7)))
assert th.allclose((utils.image2patch(this, (7, 7)) * other).sum(),
                   (utils.patch2image(other, (320, 320), (7, 7)) * this).sum())


def f_nabla(x):
    e_reg, grad_reg = R.grad(x[None, None])
    return (lamda * e_reg, lamda * grad_reg.squeeze())


n_f = kernel_size**2 - 1
n_w = 63 * 2 - 1
sigmas = [0, .025, .05, .1, .2]

n_scales = 20
R = models.ProductGSM(
    n_f=n_f,
    bound_norm=False,
    zero_mean=True,
    ortho=True,
    n_scales=n_scales,
    kernel_size=kernel_size,
    K_init='random',
).cuda()
th.set_grad_enabled(False)

state = th.load('./models/gsm/state_final.pth')
R.load_state_dict(state)
R.set_sigma(0.01)


def cb(x_alg, i):
    if i == 0:
        print('here')
        noise = th.randn_like(x) * 0
        e_reg, grad_reg = R.grad(x_alg[None, None] + noise)

        iio.imwrite(
            f'gradient/gradient.png',
            (th.abs(grad_reg[0, 0]) * 255. * 100.).clamp_min(0).to(th.uint8).cpu().numpy()
        )
        iio.imwrite(
            f'gradient/input.png',
            ((x_alg + noise) * 255.).clamp_min(0).to(th.uint8).cpu().numpy()
        )
        iio.imwrite(
            f'gradient/noise.png',
            (noise * 255.).clamp_min(0).to(th.uint8).cpu().numpy()
        )
        exit(0)
    print(utils.psnr(x[None, None], x_alg[None, None].clamp_min(0)).item())


def prox_g(x: th.Tensor, alpha):
    return ifft2((fft2(x) + alpha * y) / (1 + alpha * mask))


# with th.no_grad():
#     patch_prior = apgd(zero_filled, f_nabla, prox_g, callback=cb, max_iter=300)

h = th.tensor([
    0.0104933261758410, -0.0263483047033631, -0.0517766952966370,
    0.276348304703363, 0.582566738241592, 0.276348304703363,
    -0.0517766952966369, -0.0263483047033631, 0.0104933261758408
],
              device='cuda')
h0, _ = filters.dfilters('dmaxflat4', 'd')
P = th.from_numpy(filters.modulate2(h0, 'c')).float()
R = models.GMMConv(
    n_scales=2,
    symmetric=True,
    vmin=-.5,
    vmax=.5,
    n_w=n_w,
    w_init='abs',
    lamda=th.ones((21, ), dtype=th.float32, device='cuda').cuda(),
    h=h.cuda(),
    dims=(320, 320),
    P=P.cuda()
).cuda()
# print(R.state_dict().keys())
# exit(0)
R.load_state_dict(th.load('./models/shearlets/state_final.pth'))
R.set_eta(shape=(320, 320))
R.set_sigma(0.)
lamda = 7e-5
# with th.no_grad():
#     conv_prior = apgd(zero_filled, f_nabla, prox_g)

R = mr_model()
lamda = 9
with th.no_grad():
    energybased = apgd(zero_filled, f_nabla, prox_g, cb, gamma=0)

# for rec, name in zip([
#     zero_filled, quadratic_intensities, quadratic_gradients,
#     absolute_gradients, isotropic_tv, patch_prior, conv_prior, energybased
# ], [
#     'zero-filled', 'quadratic-intensities', 'quadratic-gradients',
#     'absolute-gradients', 'isotropic-tv', 'patch-prior', 'conv-prior',
#     'energy-based'
# ]):
#     print(
#         name,
#         utils.psnr(x[None, None], rec[None,
#                                       None].clamp_min(0)).squeeze().item()
#     )
#     iio.imwrite(
#         f'reconstructions/{name}.png',
#         (rec * 255.).clamp_min(0).to(th.uint8).cpu().numpy()
#     )
