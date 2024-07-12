import numpy as np
import torch as th
import shltutils.misc as misc
import torch.nn.functional as F


class ShearletSystem2D(th.nn.Module):
    def __init__(
        self,
        n_scales,
        dims,
        h=None,
        P=None,
        version=0,
    ):
        super().__init__()
        # version 0 is as in the original toolbox,
        # version 1 is as described in the paper (g \dyad h)
        self.version = version
        assert self.version in [0, 1]
        self.P = P
        self.device = P.device
        self.datatype = th.complex64

        self.n_scales = n_scales
        self.shear_levels = np.ceil(np.arange(1, n_scales + 1) / 2).astype(int)

        self.n_shearlets = self.get_nshearlets(self.shear_levels)
        self.cone_offs = int(self.n_shearlets - 1) // 2
        self.dims = dims

        # prepare other filters
        self.hj = self.scale_hj(h)
        self.gj = self.scale_gj(self.hj)
        # bandpass commputed from g used to multiply on all shearlets
        self.W_high = self.get_bandpass()
        # lowpass for last shearlet responsible for contrast
        self.lowpass = self.get_lowpass()

        # get shearlets
        self.shearlets = self.compute_shearlets()
        self.shearlets_norm = th.sum(th.pow(th.abs(self.shearlets), 2), axis=0)
        self.shearlets_td = th.fft.fftshift(
            th.fft.ifftn(self.shearlets, dim=(-2, -1)), dim=(-2, -1)
        )

    def get_lowpass(self):
        lowpass = th.fft.fftshift(
            th.fft.fftn(
                th.fft.ifftshift(
                    misc.padArray(
                        th.outer(self.hj[0], self.hj[0]),
                        np.array([self.dims[0], self.dims[1]])
                    )
                )
            )
        )
        return lowpass

    def get_bandpass(self):
        # bandpass (extract highpass of W)
        W_high = th.zeros((self.n_scales, self.dims[0], self.dims[1]),
                          dtype=self.datatype,
                          device=self.device)
        for j in range(len(self.gj)):
            W_high[j] = th.fft.fftshift(
                th.fft.fftn(
                    th.fft.ifftshift(
                        misc.padArray(
                            self.gj[j], np.array([self.dims[0], self.dims[1]])
                        )
                    )
                )
            )
        return W_high

    def scale_hj(self, h):
        hj = [None] * self.n_scales
        hj[-1] = h

        PadW = hj[-1].shape[0] - 1 if hj[-1].shape[0] % 2 == 1 else hj[
            -1].shape[0]
        for j in range(self.n_scales - 2, -1, -1):
            hj[j] = F.conv1d(
                misc.upsample(hj[j + 1], 1, 1).reshape(1, 1, -1),
                hj[-1].reshape(1, 1, -1),
                padding=PadW
            )[0, 0]

        return hj

    def scale_gj(self, hj):
        g = th.pow(-1, th.arange(hj[-1].shape[0], device=self.device)) * hj[-1]
        gj = [None] * self.n_scales
        gj[-1] = g

        PadW = hj[-1].shape[0] - 1 if hj[-1].shape[0] % 2 == 1 else hj[
            -1].shape[0]
        for j in range(self.n_scales - 2, -1, -1):
            gj[j] = F.conv1d(
                misc.upsample(gj[j + 1], 1, 1).reshape(1, 1, -1),
                hj[-1].reshape(1, 1, -1),
                padding=PadW
            )[0, 0]

        return gj

    def get_nshearlets(
        self, shearLevel
    ):  # determine number of resultant shearlets
        nSh = 1  # scale 0, contrast shearlet
        for i in range(len(shearLevel)):
            # nSh += (2*2**(shearLevel[i]) + 1) + (2*2**(shearLevel[i]-1) + 1)
            # two terms (two cones), second cone shearlets at border omitted
            nSh += 2 * (
                2 * 2**(shearLevel[i]) + 1
            )  # full shearlet system (*2 for both cones)
        return nSh

    def apply_transpose(self, coeff):
        img = (
            th.fft.fftshift(
                th.fft.
                fftn(th.fft.ifftshift(coeff, dim=(-2, -1)), dim=(-2, -1)),
                dim=(-2, -1)
            ) * (self.shearlets[None, :])
        ).sum(1)
        return th.real(
            th.fft.fftshift(
                th.fft.ifftn(
                    th.fft.ifftshift(img, dim=(-2, -1)), dim=(-2, -1)
                ),
                dim=(-2, -1)
            )
        )

    def apply(self, img):
        # input image img in (bs,m,n)
        # output coefficients (bs,ncoeff,m,n)
        img_freq = th.fft.fftshift(
            th.fft.fftn(th.fft.ifftshift(img, dim=(-2, -1)), dim=(-2, -1)),
            dim=(-2, -1)
        )
        coeffs = th.fft.fftshift(
            th.fft.ifftn(
                th.fft.ifftshift(
                    img_freq[:, None] * th.conj(self.shearlets[None, :, :, :]),
                    dim=(-2, -1)
                ),
                dim=(-2, -1)
            ),
            dim=(-2, -1)
        )
        return th.real(coeffs)

    def apply_inv(self, coeff):
        coeff = coeff
        batchsize = coeff.shape[0]
        img = th.zeros((batchsize, coeff.shape[-2], coeff.shape[-1]),
                       dtype=self.datatype,
                       device=self.device)
        img = (
            th.fft.fftshift(
                th.fft.
                fftn(th.fft.ifftshift(coeff, dim=(-2, -1)), dim=(-2, -1)),
                dim=(-2, -1)
            ) * self.shearlets[None, :, :, :]
        ).sum(1)
        img = th.fft.fftshift(
            th.fft.ifftn(
                th.fft.ifftshift(
                    img / self.shearlets_norm[None, :], dim=(-2, -1)
                ),
                dim=(-2, -1)
            ),
            dim=(-2, -1)
        )
        return th.real(img)

    def compute_shearlets(self):
        shearlets = th.zeros((self.n_shearlets, self.dims[0], self.dims[1]),
                             dtype=self.datatype,
                             device=self.device)
        self.scale_idx = th.zeros((self.n_shearlets, ))
        self.shear_idx = th.zeros((self.n_shearlets, ))
        shearlet_idx = 0

        for scale in np.arange(self.n_scales):
            shearLevel = self.shear_levels[scale]

            # upsample directional filter
            P_up = misc.upsample(self.P, 0, np.power(2, shearLevel + 1) - 1)

            # convolve P_up with lowpass -> remove high frequencies along vertical direction
            if self.version == 0:
                padH = self.hj[len(self.hj) - shearLevel - 1].shape[0] - 1
                psi_j0 = F.conv2d(
                    P_up[None, None, :, :],
                    self.hj[len(self.hj) - shearLevel - 1][None, None, :,
                                                           None],
                    padding=(padH, 0)
                )[0, 0]
            elif self.version == 1:
                # g_dyad_h = th.outer(self.hj[len(self.hj)-shearLevel-1],self.gj[len(self.gj)-scale-1])
                g_dyad_h = th.outer(
                    self.hj[len(self.hj) - shearLevel - 1], self.gj[scale]
                )
                padH, padW = g_dyad_h.shape[0] - 1, g_dyad_h.shape[1] - 1
                psi_j0 = F.conv2d(
                    P_up[None, None, :, :],
                    g_dyad_h[None, None, :, :],
                    padding=(padH, padW)
                )[0, 0]

            # psi_j0 = g_dyad_h.clone()
            psi_j0 = misc.padArray(
                psi_j0, np.array([self.dims[0], self.dims[1]])
            )

            # upsample psi_j0
            psi_j0_up = misc.upsample(psi_j0, 1, np.power(2, shearLevel) - 1)

            # convolve with lowpass
            lp_tmp = misc.padArray(
                self.hj[len(self.hj) - max(shearLevel - 1, 0) - 1][None, :],
                np.asarray(psi_j0_up.shape)
            )
            lp_tmp_flip = th.fliplr(lp_tmp)

            psi_j0_up = th.fft.fftshift(
                th.fft.ifftn(
                    th.fft.ifftshift(
                        th.fft.fftshift(th.fft.fftn(th.fft.ifftshift(lp_tmp)))
                        * th.fft.fftshift(
                            th.fft.fftn(th.fft.ifftshift(psi_j0_up))
                        )
                    )
                )
            )

            for shearing in range(
                -np.power(2, shearLevel),
                np.power(2, shearLevel) + 1
            ):
                psi_j0_up_shear = misc.dshear(psi_j0_up, shearing, 1)

                # convolve with flipped lowpass
                psi_j0_up_shear = th.fft.fftshift(
                    th.fft.ifftn(
                        th.fft.ifftshift(
                            th.fft.fftshift(
                                th.fft.fftn(th.fft.ifftshift(lp_tmp_flip))
                            ) * th.fft.fftshift(
                                th.fft.fftn(th.fft.ifftshift(psi_j0_up_shear))
                            )
                        )
                    )
                )

                shearlets[shearlet_idx] = th.fft.fftshift(
                    th.fft.fftn(
                        th.fft.ifftshift(
                            np.power(2, shearLevel) *
                            psi_j0_up_shear[:, 0:np.power(2, shearLevel) *
                                            self.dims[1] -
                                            1:np.power(2, shearLevel)]
                        )
                    )
                )

                if self.version == 0:
                    shearlets[shearlet_idx] *= th.conj(self.W_high[scale])

                shearlets[shearlet_idx +
                          self.cone_offs] = shearlets[shearlet_idx].T

                self.scale_idx[shearlet_idx] = scale + 1
                self.scale_idx[shearlet_idx + self.cone_offs] = scale + 1

                self.shear_idx[shearlet_idx] = shearing
                self.shear_idx[shearlet_idx + self.cone_offs] = shearing

                shearlet_idx += 1

        shearlets[-1] = self.lowpass
        self.scale_idx[-1] = 1.

        return shearlets

    def forward(self, x):
        return self.apply(x)

    def backward(self, x):
        return self.apply_transpose(x)[:, None]
