import os
from pathlib import Path

import imageio.v3 as iio
import models
import numpy as np
import shltutils.filters as filters
import torch as th

bs = 16 * 4
patch_size = 320
color = False
rotate = True
flip = True
n_w = 125

h = th.tensor([
    0.0104933261758410, -0.0263483047033631, -0.0517766952966370,
    0.276348304703363, 0.582566738241592, 0.276348304703363,
    -0.0517766952966369, -0.0263483047033631, 0.0104933261758408
],
    device='cuda')
h_ = h.clone()
lamda = th.ones((21, ), dtype=th.float32, device='cuda')
lamda = lamda.float()
h0, _ = filters.dfilters('dmaxflat4', 'd')
P = th.from_numpy(filters.modulate2(h0, 'c')).float()
print(P.shape)
gamma = .5
R = models.GMMConv(
    n_scales=2,
    symmetric=True,
    vmin=-gamma,
    vmax=gamma,
    n_w=n_w,
    w_init='abs',
    lamda=lamda.cuda(),
    h=h.cuda(),
    dims=(patch_size, patch_size),
    P=P.cuda()
).cuda()
R.load_state_dict(th.load('./models/shearlets/state_final.pth'))
R.set_eta((96, 96))

fs = []


def crop_center(img, cropx, cropy):
    y, x = img.shape[-2:]
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)
    return img[..., starty:starty + cropy, startx:startx + cropx]


image_dirs = [
    Path(os.environ.get('DATASETS_ROOT')) / 'bsds500' / sset
    for sset in ['train', 'val']
]
image_files = []
for image_dir in image_dirs:
    image_files += list(image_dir.glob('*.jpg'))
th.set_grad_enabled(False)
ylim = [-.2, 4]
dm = 0.01
bin_edges = th.linspace(
    -gamma - dm / 2,
    gamma + dm / 2,
    n_w + 1, dtype=th.float64
).cuda()
x_hist = (bin_edges[1:] + bin_edges[:-1]) / 2
R.set_sigma(0)

kh, kw = 96, 96
dh, dw = 96, 96
lamdas = R.lamda

k_out = list(range(10)) + [14, 13, 12, 11, 10] + [19, 18, 17, 16, 15]
np.savetxt('./shearlets/lamdas.csv',
           lamdas[k_out].cpu().numpy(), delimiter=',', fmt='%.2f')

sigmas = [0, .025, .05, .1, .2]
hist = th.zeros((len(sigmas), R.K.n_shearlets, n_w), dtype=th.float64).cuda()
for i_s, sigma in enumerate(sigmas):
    for i, file in enumerate(image_files):
        img = th.from_numpy(iio.imread(file) / 255.).mean(-1)
        patches = img.unfold(0, kh, dh).unfold(1, kw, dw)
        patches = patches.reshape(-1, 1, kh, kw).cuda()
        Kx = R.K(patches + sigma * th.randn_like(patches)).squeeze()
        for k in range(R.K.n_shearlets):
            hist[i_s, k] += th.histogram(lamdas[k].cpu()*Kx[:, k].reshape(-1).cpu(),
                                         bin_edges.cpu())[0].cuda()
hist = -th.log(hist)

scale = 1.1
n_points = 20
x = th.linspace(
    -scale * gamma,
    scale * gamma,
    n_points**2,
    dtype=R.w.w.dtype,
    device=R.w.w.device,
)[None].repeat(R.K.n_shearlets, 1)
n_f = R.K.n_shearlets

sigmas = [0, 0.025, 0.05, 0.1, 0.2]
K = crop_center(R.K.shearlets_td, 12, 12).real
for i, k in enumerate(K):
    if i == 20:
        break
    k_normalized = (k - K.min()) / (K.max() - K.min())
    k_normalized *= 255.
    k_normalized = k_normalized.cpu().numpy().squeeze().astype(np.uint8)
    iio.imwrite(f'./shearlets/{k_out[i]}/k_{k_out[i]}.png', k_normalized)

shs = th.abs(R.K.shearlets)
for i, sh in enumerate(shs):
    if i == 20:
        break
    sh_normalized = (sh - shs.min()) / (shs.max() - shs.min())
    sh_normalized *= 255.
    sh_normalized = sh_normalized.cpu().numpy().squeeze().astype(np.uint8)
    iio.imwrite(f'./shearlets/{k_out[i]}/sh_{k_out[i]}.png', sh_normalized)

all_fs = th.empty((len(sigmas), 20, n_points * n_points), dtype=th.float32)
all_fps = th.empty((len(sigmas), 20, n_points * n_points), dtype=th.float32)
for i_s, sigma in enumerate(sigmas):
    R.set_sigma(sigma)
    f, fp = R.pot_act(x.view(1, n_f, n_points, n_points))
    fs = f.view(n_f, n_points * n_points)[:20]
    fps = fp.view(n_f, n_points * n_points)[:20]
    all_fs[i_s] = fs
    all_fps[i_s] = fps

for i in range(20):
    np.savetxt(f'./shearlets/{k_out[i]}/potentials.csv',
               th.cat((x[0:1].cpu(), all_fs[:, i].clone().cpu())).T.numpy(), delimiter=',', header='x,a,b,c,d,e', comments='')
    np.savetxt(f'./shearlets/{k_out[i]}/potentials-prime.csv',
               th.cat((x[0:1].cpu(), all_fps[:, i].clone().cpu())).T.numpy(), delimiter=',', header='x,a,b,c,d,e', comments='')
    np.savetxt(f'./shearlets/{k_out[i]}/hists.csv',
               th.cat((x_hist[None].cpu(), hist[:, i].clone().cpu())).T.numpy(), delimiter=',', header='x,a,b,c,d,e', comments='')
