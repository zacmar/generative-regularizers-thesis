import json
import matplotlib.pyplot as plt
from typing import Callable

import imageio.v3 as iio
import nets
import numpy as np
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
    R.load_state_dict(th.load('../../../../regularizers/scripts/ebm.ckpt'))
    return R.cuda()


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


def fft2(x: th.Tensor):
    return th.fft.rfft2(x, norm='ortho')


def ifft2(x: th.Tensor):
    return th.fft.irfft2(x, norm='ortho')


mask = th.fft.fftshift(th.load('mask.pt')).cuda()[:, :161]
n = th.load('noise.pt')

sigma_n = 2e-2



def f_nabla(x):
    e_reg, grad_reg = R.grad(x[None, None])
    return (lamda * e_reg, lamda * grad_reg.squeeze())


def prox_g(x: th.Tensor, alpha):
    return ifft2((fft2(x) + alpha * y) / (1 + alpha * mask))


R = mr_model()

for filename, directory in zip([
    './file_prostate_AXT2_013.pt', './file_brain_AXT1_202_2020190_s8.pt'
], ['prostate', 'brain']):
    x = th.load(filename).cuda()
    y = mask * (fft2(x) + sigma_n * n)
    zero_filled = ifft2(y)
    iio.imwrite(f'{directory}/reference-rss.png', (x * 255.).to(th.uint8).cpu().numpy())
    iio.imwrite(
        f'{directory}/log-abs-data.png', (th.log(y.abs()) * 255.).to(th.uint8).cpu().numpy()
    )
    for lamda in [0.00, 0.01, 0.10, 1.00, 10.00, 100.00, 1000.00]:
        with th.no_grad():
            energybased = apgd(zero_filled, f_nabla, prox_g, gamma=1)

        iio.imwrite(
            f'{directory}/rec_{lamda:.2f}.png',
            (energybased * 255.).clamp_min(0).to(th.uint8).cpu().numpy()
        )
