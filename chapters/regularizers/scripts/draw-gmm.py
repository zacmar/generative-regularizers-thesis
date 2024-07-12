import os
from pathlib import Path

import data
import imageio.v3 as iio
import models
import numpy as np
import torch as th

sigmas = [0, .025, .05, .1, .2]

kernel_size = 7
n_f = kernel_size**2 - 1
bs = 64 * 4000 // 4
patch_size = kernel_size
color = False
rotate = True
flip = True
n_w = 63 * 2 - 1
n_scales = 20

dataset = data.BSDS(color, bs, patch_size, rotate, flip)
gamma = 1.5
R_gsm = models.ProductGSM(
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
state['w.w'] = state['w']
R_gsm.load_state_dict(state)
R_gmm = models.ProductGMM(
    n_f=n_f,
    bound_norm=False,
    zero_mean=True,
    symmetric=True,
    ortho=True,
    vmin=-1,
    vmax=1,
    kernel_size=kernel_size,
    K_init='random',
    n_w=n_w,
    w_init='student-t',
    sigmas=th.Tensor(sigmas)
).cuda()
th.set_grad_enabled(False)

state = th.load('./models/gmm/state_final.pth')
R_gmm.load_state_dict(state)
image_dirs = [
    Path(os.environ.get('DATASETS_ROOT')) / 'bsds500' / sset
    for sset in ['train', 'val']
]
image_files = []
for image_dir in image_dirs:
    image_files += list(image_dir.glob('*.jpg'))

kh, kw = 7, 7
dh, dw = 1, 1
patches = []
for i, file in enumerate(image_files):
    img = th.from_numpy(iio.imread(file) / 255.).mean(-1)
    patches_ = img.unfold(0, kh, dh).unfold(1, kw, dw)
    patches.append(patches_.reshape(-1, kh * kw))

patches = th.cat(patches)
patches -= patches.mean(1, keepdim=True)
# Thats how many fit on my GPU
patches = patches.cuda()[:20_000_000].view(-1, 1, 7, 7).float()
print(patches.shape)

dm = 0.01
bin_edges = th.linspace(
    -gamma - dm / 2,
    gamma + dm / 2,
    n_w + 1,
).cuda()
x_hist = (bin_edges[1:] + bin_edges[:-1]) / 2
for R, name in zip([R_gmm, R_gsm], ['gmm', 'gsm']):
    hist = th.zeros((len(sigmas), n_f, n_w)).cuda()
    for i_s, sigma in enumerate(sigmas):
        R.set_sigma(sigma)
        Kx = R.K(patches + sigma * th.randn_like(patches))

        for k in range(n_f):
            hist[
                i_s,
                k] = -th.log(th.histogram(Kx[:, k].reshape(-1).cpu(),
                                          bin_edges.cpu())[0].to('cuda'))

    fs = []
    fps = []
    K = R.K.weight.data
    scale = 1.1

    n_points = 20
    x = th.linspace(
        -scale * gamma,
        scale * gamma,
        n_points**2,
        dtype=K.dtype,
        device=K.device,
    )[None].repeat(n_f, 1)
    for sig in sigmas:
        R.set_sigma(sig)
        f, fp = R.pot_act(x.view(1, n_f, n_points, n_points))
        f = f.view(n_f, n_points * n_points)
        fp = fp.view(n_f, n_points * n_points)
        fs.append(f)
        fps.append(fp)

    x = x[0]

    fs = th.stack(fs).permute(1, 0, 2)
    fps = th.stack(fps).permute(1, 0, 2)
    norm_k = (K**2).sum((1, 2, 3))
    indices = th.sort(norm_k)[1]
    K = K[indices]
    fs = fs[indices]
    fps = fps[indices]
    hist = hist.permute(1, 0, 2)
    hist = hist[indices]
    kminmax = th.stack((K.amin((-2, -1)), K.amax((-2, -1))), 1).squeeze().cpu()

    np.savetxt(f'./ours/{name}/kvals.csv',
               kminmax.numpy(), delimiter=',', fmt='%.2f')

    for i, (ff, ffp, hh, kk) in enumerate(zip(fs, fps, hist, K)):
        kk_normalized = ((kk - kk.min()) / (kk.max() - kk.min())).cpu().numpy()
        kk_normalized *= 255.
        kk_normalized = kk_normalized.astype(np.uint8)
        iio.imwrite(f'./ours/{name}/{i}/k_{i}.png', kk_normalized.squeeze())
        np.savetxt(f'./ours/{name}/{i}/potentials-prime.csv',
                   th.cat((x[None].cpu(), ffp.cpu())).T.numpy(), delimiter=',', header='x,a,b,c,d,e', comments='')
        np.savetxt(f'./ours/{name}/{i}/potentials.csv',
                   th.cat((x[None].cpu(), ff.cpu())).T.numpy(), delimiter=',', header='x,a,b,c,d,e', comments='')
        np.savetxt(f'./ours/{name}/{i}/hists.csv',
                   th.cat((x_hist[None].cpu(), hh.detach().cpu())).T.numpy(), delimiter=',', header='x,a,b,c,d,e', comments='')
