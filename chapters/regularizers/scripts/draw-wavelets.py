import logsumexp
import matplotlib.pyplot as plt
import models
import numpy as np
import pywt
import torch as th

sigmas = [0, .025, .05, .1, .2]
N = len(sigmas)
gamma = .5
diag_mul = .7
n_w = 125
mus_l1 = th.linspace(-gamma, gamma, n_w, device='cuda')[None].repeat(2, 1)
mus_l1d = th.linspace(-gamma * diag_mul, gamma * diag_mul, n_w,
                      device='cuda')[None]
mus_l2 = th.linspace(-gamma * 2, gamma * 2, n_w,
                     device='cuda')[None].repeat(2, 1)
mus_l2d = th.linspace(
    -gamma * 2 * diag_mul, gamma * 2 * diag_mul, n_w, device='cuda'
)[None]
mus = th.cat((mus_l1, mus_l1d, mus_l2, mus_l2d), dim=0)

n_points = 20
ylimss_f = [[-.2, 7.2], [-.2, 7.2]]
levels = 2

for wave, ylims_f in zip(['db2', 'db4'], ylimss_f):
    R = models.WaveletGMM(
        levels=2,
        mus=mus,
        vmin=-.7,
        vmax=.7,
        n_w=n_w,
        w_init='student-t',
        im_sz=64,
        wave=wave
    ).cuda()
    R.set_sigma(0)
    R.load_state_dict(th.load(f'./models/wavelets/{wave}/state_final.pth'))

    xs_pot = []
    for level in range(levels):
        for direction in range(3):
            qs = R.mus[level * 3 + direction].amax()
            xs_pot.append(
                th.linspace(
                    -1.1 * qs,
                    1.1 * qs,
                    n_points**2,
                    device='cuda',
                    dtype=th.float32
                )
            )

    for level in range(2):
        for direction in range(3):
            idx_flat = level * 3 + direction
            x = xs_pot[idx_flat]
            pots = th.empty((len(sigmas), n_points * n_points))
            tweedies = th.empty((len(sigmas), n_points * n_points))

            for i_s, s in enumerate(sigmas):
                R.set_sigma(s)
                pot, act = logsumexp.pot_act(
                    x.view(1, 1, n_points, n_points),
                    R.w.get()[idx_flat:idx_flat + 1], R.mus[idx_flat],
                    R.sigma[idx_flat:idx_flat + 1]
                )
                pot = pot.view(n_points * n_points)
                pots[i_s] = pot.cpu()
                act = act.view(n_points * n_points)
                tweedie = x - s**2 * act
                tweedies[i_s] = tweedie.cpu()

            np.savetxt(f'./wavelets/{wave}/pot/{level}_{direction}.csv',
                       th.cat((x[None].cpu(), pots.detach().cpu())).T.numpy(), delimiter=',', header='x,a,b,c,d,e', comments='')

            np.savetxt(f'./wavelets/{wave}/tweedie/{level}_{direction}.csv',
                       th.cat((x[None].cpu(), tweedies.detach().cpu())).T.numpy(), delimiter=',', header='x,a,b,c,d,e', comments='')

    h = R.h.detach()
    g = R.get_highpass(h)
    h_list = h.cpu().numpy().tolist()
    g_list = g.cpu().numpy().tolist()

    wavelet = pywt.Wavelet(f'{wave}learned', [h_list, g_list, h_list, g_list])
    phi_d, psi_d, phi_r, psi_r, x = wavelet.wavefun(6)
    np.savetxt(f'./wavelets/{wave}/phi-learned.csv',
               np.stack((x, np.flip(phi_d))).T, delimiter=',', header='x,y', comments='')
    np.savetxt(f'./wavelets/{wave}/psi-learned.csv',
               np.stack((x, np.flip(psi_d))).T, delimiter=',', header='x,y', comments='')
    a = np.arange(len(h))
    b = np.flip(h.cpu().numpy())
    np.savetxt(f'./wavelets/{wave}/h-learned.csv',
               np.stack((a, b)).T, delimiter=',', header='x,y', comments='')

    wavelet = pywt.Wavelet(wave)
    h_ = wavelet.dec_lo
    phi, psi, x = wavelet.wavefun(6)
    np.savetxt(f'./wavelets/{wave}/phi.csv',
               np.stack((x, phi)).T, delimiter=',', header='x,y', comments='')
    np.savetxt(f'./wavelets/{wave}/psi.csv',
               np.stack((x, psi)).T, delimiter=',', header='x,y', comments='')
    np.savetxt(f'./wavelets/{wave}/h.csv',
               np.stack((np.arange(len(h_)), h_)).T, delimiter=',', header='x,y', comments='')

