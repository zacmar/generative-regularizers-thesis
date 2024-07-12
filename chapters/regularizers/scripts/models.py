import conv
import logsumexp
import numpy as np
import pytorch_wavelets.dwt.lowlevel as lowlevel
import pywt
import shltutils.shearlet as sh
import torch as th
import utils


class SimplexWeights(th.nn.Module):
    def __init__(
        self,
        n_f: int = 32,
        n_w: int = 63,
        symmetric: bool = True,
        vmin: float = -1,
        vmax: float = 1,
        w_init: str = 'student-t',
        softmax: bool = False,
    ):
        super().__init__()
        self.w = th.nn.Parameter(
            utils.init_params(w_init, vmin, vmax, n_f, n_w, symmetric)
        )
        if softmax:
            self.w.data = th.log(self.w.data)

        self.symmetric = symmetric
        self.softmax = softmax
        if not softmax:

            def proj():
                weights = self.get().data
                simplex_weights = utils.proj_simplex_simul(weights
                                                           ).clamp_min(1e-14)
                if self.symmetric:
                    self.w.data.copy_(simplex_weights[:, self.w.shape[1] - 1:])
                else:
                    self.w.data.copy_(simplex_weights)

            self.w.proj = proj
            self.w.reduction_dim = (1, )
            self.w.proj()

    def get(self):
        if self.symmetric:
            weights = th.cat((th.flip(self.w, (1, ))[:, :-1], self.w), dim=1)
        else:
            weights = self.w

        if self.softmax:
            return th.nn.functional.softmax(weights, dim=1)
        else:
            return weights


class WaveletGMM(th.nn.Module):
    def __init__(
        self,
        mus,
        levels: int = 2,
        vmin: float = -1,
        vmax: float = 1,
        n_w: int = 65,
        im_sz: int = 32,
        n_c: int = 32,
        w_init: str = 'abs',
        wave: str = 'db1',
    ):
        super().__init__()
        self.n_f = 3 * levels
        self.levels = levels
        self.lambdas = th.nn.Parameter(th.ones((self.n_f + 1, )).cuda())
        self.lambdas.proj = lambda: self.lambdas.data.clamp_(min=1e-9)
        self.h = th.nn.Parameter(
            th.from_numpy(np.array(pywt.Wavelet(wave).dec_lo)).flip(0)
        )
        self.lamdas = th.rand((2 + len(self.h) - 1, ))

        def proj_c():
            h_tilde = self.h.data
            x_ = th.cat((h_tilde, self.lamdas.to(self.h.device)))
            for _ in range(10):

                def nabla_l(args):
                    h = args[:h_tilde.shape[0]]
                    lamda = args[h_tilde.shape[0]:]
                    alternating = th.tensor([1, -1] * (len(h) // 2),
                                            device=h.device,
                                            dtype=h.dtype).flip(0)
                    nabla_h = h - h_tilde + lamda[0] + lamda[1] * alternating
                    nabla_lamda_ortho = []
                    for i in range(-len(h) // 2 + 1, len(h) // 2):
                        # pretty sure padded and rolled are equivalent!
                        # Here we essentially have the same constraint twice
                        # rolled
                        # Actually not true, but rolling is probably good
                        # enough since we're close to a solution..
                        nabla_h += 2 * lamda[2 + i - (-len(h) // 2 +
                                                      1)] * h.roll(2 * i)
                        nabla_lamda_ortho.append((h *
                                                  h.roll(2 * i)).sum()[None] -
                                                 1 * (i == 0))

                        # padded
                        # pad = th.zeros((2 * abs(i), )).to(h.device)
                        # h_padded = th.cat(
                        #     (h[2 * i:], pad)
                        # ) if i > 0 else th.cat((pad, h[:2 * abs(i)]))
                        # if i == 0:
                        #     h_padded = h
                        # # print(h_padded)
                        # nabla_h += 2 * lamda[2 + i -
                        #                      (-len(h) // 2 + 1)] * h_padded
                        # nabla_lamda_ortho.append((h * h_padded).sum()[None] -
                        #                          1 * (i == 0))

                    return th.cat((
                        nabla_h, h.sum()[None] - np.sqrt(2),
                        self.get_highpass(h).sum()[None], *nabla_lamda_ortho
                    ))

                jac = th.autograd.functional.jacobian(nabla_l, x_)
                rhs = jac.T @ nabla_l(x_)
                lhs = jac.mT @ jac + 1e-10 * th.eye(jac.shape[1]
                                                    ).to(jac.device)
                x_ -= th.linalg.solve(lhs, rhs)
            self.h.data.copy_(x_[:h_tilde.shape[0]])
            self.lamdas.copy_(x_[h_tilde.shape[0]:])

        self.h.proj = proj_c
        self.h.proj()
        self.h.reduction_dim = (0, )

        self.mode = 'reflect'
        # Hacky way to get the shape of the lower features by just using the
        # transform once
        yl, yh = self.wave_forward(
            th.ones((1, 1, im_sz, im_sz)).to(self.h.dtype)
        )
        feat_lowest = yl.shape[2]**2
        self.pot_sizes = [yh[i].shape[3] for i in range(self.levels)]

        self.w = SimplexWeights(
            self.n_f, n_w, vmin=vmin, vmax=vmax, w_init=w_init, symmetric=True
        )
        self.register_buffer('mus', mus)
        self._sigma_0 = (mus.amax(1) - mus.amin(1)) / (n_w - 1)
        self.register_buffer('sigma_0', self._sigma_0)

    def get_highpass(self, h):
        # Not sure why this doesn't start with -1, but this complies with the
        # library
        alternating = th.tensor([1, -1] * (len(h) // 2),
                                device=h.device,
                                dtype=h.dtype)
        return alternating * h.flip(0)

    def _forward(self, x, h0_row, h1_row, h0_col, h1_col, mode):
        # Taken from pytorch wavelets, but we need the gradient wrt h,
        # so i just let autograd do it...
        mode = lowlevel.int_to_mode(mode)
        lohi = lowlevel.afb1d(x, h0_row, h1_row, mode=mode, dim=3)
        y = lowlevel.afb1d(lohi, h0_col, h1_col, mode=mode, dim=2)
        s = y.shape
        y = y.reshape(s[0], -1, 4, s[-2], s[-1])
        low = y[:, :, 0].contiguous()
        highs = y[:, :, 1:].contiguous()
        return low, highs

    def wave_forward(self, x):
        self.g = self.get_highpass(self.h)
        yh = []
        ll = x
        mode = lowlevel.mode_to_int(self.mode)

        for j in range(self.levels):
            ll, high = self._forward(
                ll, self.h[None, None, :, None], self.g[None, None, :, None],
                self.h[None, None, None, :], self.g[None, None, None, :], mode
            )
            yh.append(high)

        return ll, yh

    def _backward(low, highs, g0_row, g1_row, g0_col, g1_col, mode):
        # Taken from pytorch wavelets, but we need the gradient wrt h,
        # so i just let autograd do it...
        mode = lowlevel.int_to_mode(mode)
        lh, hl, hh = th.unbind(highs, dim=2)
        lo = lowlevel.sfb1d(low, lh, g0_col, g1_col, mode=mode, dim=2)
        hi = lowlevel.sfb1d(hl, hh, g0_col, g1_col, mode=mode, dim=2)
        y = lowlevel.sfb1d(lo, hi, g0_row, g1_row, mode=mode, dim=3)
        return y

    def wave_backward(self, coeffs):
        self.g = self.get_highpass(self.h)
        yl, yh = coeffs
        ll = yl
        mode = lowlevel.mode_to_int(self.mode)

        for h in yh[::-1]:
            if ll.shape[-2] > h.shape[-2]:
                ll = ll[..., :-1, :]
            if ll.shape[-1] > h.shape[-1]:
                ll = ll[..., :-1]
            ll = lowlevel.SFB2D.apply(
                ll, h, self.h[None, None, :, None], self.g[None, None, :,
                                                           None],
                self.h[None, None, None, :], self.g[None, None, None, :], mode
            )
        return ll

    def grad(self, x):
        Kx = self.wave_forward(x)
        act_buf = [
            x.new_zeros(
                (*x.shape[:2], 3, self.pot_sizes[i], self.pot_sizes[i])
            ) for i in range(self.levels)
        ]
        # pot_global, act_global = self.global_gmm.grad(self.lambdas[-1] * Kx[0])
        act_global = th.zeros_like(Kx[0], requires_grad=True)
        pot_sum = 0  # pot_global[:, None, None, None]

        for level in range(self.levels):
            for direction in range(3):
                idx_flat = level * 3 + direction
                pot, act = logsumexp.pot_act(
                    self.lambdas[idx_flat] * Kx[1][level][:, :, direction],
                    self.w.get()[idx_flat:idx_flat + 1],
                    self.mus[idx_flat],
                    self.sigma[idx_flat:idx_flat + 1],
                )
                pot_sum += pot.sum((1, 2, 3), keepdim=True)
                act_buf[level][:, :,
                               direction] = act * self.lambdas[level * 3 +
                                                               direction]
        g1 = self.wave_backward((0 * self.lambdas[-1] * act_global, act_buf))
        return pot_sum, g1

    def set_sigma(self, sigma):
        self.sigma = th.sqrt(
            (self.sigma_0**2 + self.lambdas[:-1]**2 * sigma**2)
        )


class ProductGSM(th.nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        n_f: int = 5**2 - 1,
        kernel_size: int = 5,
        bound_norm: bool = False,
        zero_mean: bool = True,
        ortho: bool = True,
        n_scales: int = 20,
        K_init: str = 'random',
        sigma_0: float = 0.01,
        mult: float = 1.4,
    ):
        super().__init__()
        self.n_f = n_f
        self.n_scales = n_scales
        self.K = conv.Conv2d(
            in_channels,
            n_f,
            kernel_size,
            zero_mean=zero_mean,
            bound_norm=bound_norm,
            ortho=ortho,
            init=K_init,
        )
        self.w = SimplexWeights(
            n_f, n_scales, symmetric=False, w_init='random'
        )
        sigmas_0 = th.tensor([sigma_0 * mult**i for i in range(self.n_scales)]
                             )[None].repeat(self.n_f, 1).clone()
        self.register_buffer('sigmas_0', sigmas_0)
        self.set_sigma(0)

    def pot_act(self, x):
        '''
        too lazy to implement in cuda, although would help
        here we have to broadcast over the mixture dimension
        '''
        _w = self.w.get()[None, :, :, None, None]
        _sigmas = self.sigmas[None, :, :, None, None]
        _x = x[:, :, None]
        max_exp = (-(_x / _sigmas)**2 / 2).amax(2, keepdim=True)
        gsm = th.sum(
            _w / _sigmas / np.sqrt(2 * np.pi) *
            th.exp(-(_x / _sigmas)**2 / 2 - max_exp),
            dim=(2, ),
        )
        pot = -(th.log(gsm) + max_exp[:, :, 0])
        act = th.sum(
            _w * _x / _sigmas**3 / np.sqrt(2 * np.pi) *
            th.exp(-(_x / _sigmas)**2 / 2 - max_exp),
            dim=(2, )
        ) / gsm
        return pot, act

    def grad(self, x):
        Kx = self.K(x)
        pot, act = self.pot_act(Kx)
        e1 = pot.sum((1, 2, 3), keepdim=True)
        g1 = self.K.backward(act)
        return e1, g1

    def set_sigma(
        self,
        sigma: float,
    ):
        norm_k2 = (self.K.weight**2).sum((1, 2, 3))
        self.sigmas = th.sqrt(self.sigmas_0**2 + norm_k2[:, None] * sigma**2)


class ProductGMM(th.nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        n_f: int = 32,
        kernel_size: int = 7,
        bound_norm: bool = False,
        zero_mean: bool = True,
        symmetric: bool = True,
        ortho: bool = True,
        vmin: float = -1,
        vmax: float = 1,
        n_w: int = 34,
        w_init: str = 'abs',
        K_init: str = 'dct',
        sigmas=None,
    ):
        super().__init__()
        self.n_f = n_f
        self.vmin = vmin
        self.n_w = n_w
        self.vmax = vmax
        self.K = conv.Conv2d(
            in_channels,
            n_f,
            kernel_size,
            zero_mean=zero_mean,
            bound_norm=bound_norm,
            ortho=ortho,
            init=K_init,
        )
        self.symmetric = symmetric
        self._sigma_0 = (vmax - vmin) / (n_w - 1)
        self.w = SimplexWeights(n_f, n_w, symmetric, vmin, vmax, w_init)
        self.register_buffer('mus', th.linspace(vmin, vmax, n_w))
        self.register_buffer('sigma_0', th.ones((n_f, )) * self._sigma_0)

    def pot_act(self, x):
        weights = self.w.get()
        return logsumexp.pot_act(x, weights, self.mus, self.sigma)

    def grad(self, x):
        Kx = self.K(x)
        pot, act = self.pot_act(Kx)
        e1 = pot.sum((1, 2, 3), keepdim=True)
        g1 = self.K.backward(act)
        return e1, g1

    def set_sigma(
        self,
        sigma: float,
    ):
        norm_k2 = (self.K.weight**2).sum((1, 2, 3))
        self.sigma = th.sqrt(self.sigma_0**2 + norm_k2 * sigma**2)


class GMMConv(th.nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        symmetric: bool = True,
        vmin: float = -1,
        vmax: float = 1,
        n_w: int = 34,
        w_init: str = 'abs',
        sigmas=None,
        n_scales=1,
        dims=(128, 128),
        lamda=None,
        h=None,
        P=None
    ):
        super().__init__()
        self.vmin = vmin
        self.n_w = n_w
        self.vmax = vmax
        self.lamda = th.nn.Parameter(lamda)
        self.dims = dims
        self.h = th.nn.Parameter(h)
        self.P = th.nn.Parameter(P)

        def proj():
            self.h.data.copy_(
                self.h.data - (self.h.data.sum() - 1) / self.h.data.shape[0]
            )

        self.h.proj = proj
        self.h.proj()
        self.h.reduction_dim = (0, )

        def proj_P():
            sign = th.sign(self.P.data)
            abs_simplex_proj = utils.proj_simplex(self.P.data.abs().view(-1))
            self.P.data.copy_(sign * abs_simplex_proj.view(self.P.shape))

        self.P.proj = proj_P
        self.P.proj()
        self.P.reduction_dim = (0, 1)

        def proj_lamda():
            self.lamda.data.copy_(th.clamp(self.lamda.data, min=0, max=None))
            self.lamda.data[-1] = 0

        self.lamda.proj = proj_lamda
        self.lamda.proj()
        self.w = SimplexWeights(21, n_w, symmetric, vmin, vmax, w_init)
        self._sigma_0 = (vmax - vmin) / (n_w - 1)
        self.register_buffer('mus', th.linspace(vmin, vmax, n_w))
        self.register_buffer('sigma_0', th.ones((21, )) * self._sigma_0)

    def pot_act(self, x):
        weights = self.w.get()
        return logsumexp.pot_act(x, weights, self.mus, self.sigma)

    def grad(self, x):
        Kx = self.lamda[None, :, None, None] * self.K(x[:,0])
        pot, act = self.pot_act(Kx)
        e1 = pot.sum((1, 2, 3), keepdim=True)
        g1 = self.K.backward(self.lamda[None, :, None, None] * act)
        return e1, g1

    def set_eta(self, shape=(96, 96)):
        # We have to construct the shearlet system each time since
        # we learn the mother shearlets
        self.K = sh.ShearletSystem2D(2, shape, self.h, self.P, 1)
        # Apparently we didnt take lambda into account here when training the
        # old model
        self.eta = th.abs(self.K.shearlets).amax((1, 2))
        # self.eta = self.lamda.data * th.abs(self.K.shearlets).amax((1, 2))

    def set_sigma(
        self,
        sigma: float,
        shape=(96, 96),
    ):
        self.sigma = th.sqrt(self.sigma_0**2 + self.eta**2 * sigma**2)

