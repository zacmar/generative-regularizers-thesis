import fastmri.models as fm
import json
import imageio.v3 as iio
import numpy as np
import nets
from typing import Callable
import torch as th
import matplotlib.pyplot as plt
import data


def ula(
    x_init: th.Tensor,
    nabla: Callable[[th.Tensor], th.Tensor],
    n: int = 500,
    epsilon: float = 7.5e-3,
    callback: Callable[[th.Tensor, int], None] = lambda x, i: None,
) -> th.Tensor:
    x = x_init.clone()
    for i in range(n):
        x -= nabla(x) + epsilon * th.randn_like(x)
        callback(x, i)

    return x


def apgd(
    x_init: th.Tensor,
    f_nabla,
    prox: Callable[[th.Tensor, th.Tensor], th.Tensor],
    callback: Callable[[th.Tensor, int], None] = lambda x, i: None,
    max_iter: int = 100,
    gamma=0,
):
    x = x_init.clone()
    x_old = x.clone()
    L = 1
    beta = 1 / np.sqrt(2)
    for i in range(max_iter):
        x_bar = x + beta * (x - x_old)
        x_old = x.clone()
        n = th.randn_like(x) * 7.5e-3 * (i < 50) * gamma
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


def unet_normalize(data: th.Tensor, eps=1e-11):
    mean = data.mean(dim=(1, 2, 3), keepdim=True)
    std = data.std(dim=(1, 2, 3), keepdim=True)
    normalized = (data - mean) / (std + eps)
    return normalized, mean, std


def call_unet(net, samples):
    normalized, mean, std = unet_normalize(samples)
    out = th.stack([net(im[None])[0] for im in normalized])
    return out * std + mean


def unet_model():
    R = fm.Unet(
        in_chans=1, out_chans=1, chans=256, num_pool_layers=4, drop_prob=0.0
    )
    state_dict = th.load('./unet.ckpt')
    R.load_state_dict(state_dict)
    R = R.eval()
    return R.cuda()


unet = unet_model()
x = data.synthetic_data()[13:14]
iio.imwrite(
    'reference.png',
    ((x[0, 0] / x.amax()) * 255.).cpu().numpy().clip(0, 255).astype(np.uint8)
)
x = (x - x.amin((1, 2, 3), keepdim=True)
     ) / (x.amax((1, 2, 3), keepdim=True) - x.amin((1, 2, 3), keepdim=True))
center_fraction = 0.08
cols = 320
acceleration = 4

rng = np.random.default_rng()
num_low_freqs = int(round(cols * center_fraction))
prob = (cols / acceleration - num_low_freqs) / (cols - num_low_freqs)
removed = []
psnrs_unet = np.zeros(294)
psnrs_ours = np.zeros(294)
psnrs_naive = np.zeros(294)

z_init = th.fft.fft2((x + 0j), norm='ortho')
z_init += th.randn_like(z_init) * 0.007
repeats = 1
mask = th.ones_like(th.fft.fftshift(th.from_numpy(iio.imread('mask.png'))).cuda() / 255.)[None, None]
z = z_init * mask[None, None]


def f_nabla(x):
    e_reg, grad_reg = R.grad(x)
    return (lamda * e_reg, lamda * grad_reg)


def fft2(x: th.Tensor):
    return th.fft.fft2(x, norm='ortho')


def ifft2(x: th.Tensor):
    return th.fft.ifft2(x, norm='ortho')


def prox_g(x: th.Tensor, alpha):
    return ifft2((fft2(x) + alpha * z) / (1 + alpha * mask)).real

R = mr_model()

def cb(x, i):
    return
    plt.figure()
    plt.imshow(x[0,0].cpu().numpy())
    plt.show()

with th.no_grad():
    for _ in range(repeats):
        z = z_init.clone()
        to_remove = list(range(num_low_freqs // 2, cols - num_low_freqs // 2))
        while to_remove:
            reconstruction = call_unet(
                unet,
                th.fft.ifft2(z, norm='ortho').abs()
            )
            iio.imwrite(
                f'recos/unet/{26+len(to_remove):03d}.png',
                ((reconstruction[0, 0] / reconstruction[0, 0].amax()) *
                 255.).cpu().numpy().clip(0, 255).astype(np.uint8)
            )
            psnr = (
                10 * th.log10((x**2).amax() / th.mean((reconstruction - x)**2))
            ).item()
            psnrs_unet[len(to_remove) - 1] += psnr
            naive = th.fft.ifft2(z, norm='ortho').abs()
            iio.imwrite(
                f'recos/naive/{26+len(to_remove):03d}.png',
                ((naive[0, 0] / naive[0, 0].amax()) *
                 255.).cpu().numpy().clip(0, 255).astype(np.uint8)
            )
            psnr = (10 *
                    th.log10((x**2).amax() / th.mean((naive - x)**2))).item()
            psnrs_naive[len(to_remove) - 1] += psnr
            lamda = 1

            samples = 5000
            each = 1
            burn_in = 10000
            def nabla(f):
                return lamda * (ifft2(mask * fft2(f) - z)).real + R.grad(f)[1]

            def accumulate(f, i):
                global n, mmse, m2
                idx = (i - burn_in) // each
                if idx < samples and i > burn_in and i % each == 0:
                    n = n + 1
                    delta = f - mmse
                    mmse += delta / n
                    # m2 += delta * (f - mmse)

            n = 0
            # MMSE and variance
            mmse = th.zeros_like(x)
            # m2 = th.zeros_like(x)

            # Initialize with MAP estimate
            ours = ula(
                th.fft.ifft2(z, norm='ortho').abs(),
                nabla=nabla,
                callback=accumulate,
                n=burn_in + each * samples,
            )
            psnr = (10 *
                    th.log10((x**2).amax() / th.mean((mmse - x)**2))).item()
            print(psnr)
            psnrs_ours[len(to_remove) - 1] += psnr
            iio.imwrite(
                f'recos/ours/{26+len(to_remove):03d}.png',
                ((mmse[0, 0] / mmse[0, 0].amax()) *
                 255.).cpu().numpy().clip(0, 255).astype(np.uint8)
            )

            line = rng.choice(to_remove)
            z[:, :, :, line] = 0
            mask[:, :, :, line] = 0
            to_remove.remove(line)

xaxis = 26 + np.arange(294)
np.savetxt(
    f'psnr.csv',
    np.stack((xaxis, psnrs_unet, psnrs_naive, psnrs_ours), axis=1),
    delimiter=',',
    header='lines,unet,naive,ours',
    comments=''
)
plt.figure()
plt.plot(26 + np.arange(294), psnrs_unet)
plt.plot(26 + np.arange(294), psnrs_naive)
plt.plot(26 + np.arange(294), psnrs_ours)
plt.show()
