import math

import numpy as np
import torch as th

__all__ = ['Conv2d']


class Conv2d(th.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        invariant=False,
        stride=1,
        dilation=1,
        groups=1,
        bias=False,
        zero_mean=False,
        bound_norm=False,
        pad=False,
        ortho=False,
        init='random',
    ):
        super(Conv2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.invariant = invariant
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.bias = th.nn.Parameter(th.zeros(out_channels)) if bias else None
        self.zero_mean = zero_mean
        self.bound_norm = bound_norm
        self.padding = 0
        self.pad = pad

        # add the parameter
        if self.invariant:
            assert self.kernel_size == 3
            self.weight = th.nn.Parameter(
                th.empty(out_channels, in_channels, 1, 3)
            )
            self.register_buffer(
                'mask',
                th.tensor([1., 4., 4.])[None, None, None, :]
            )
        else:
            self.weight = th.nn.Parameter(
                th.empty(
                    out_channels, in_channels, self.kernel_size,
                    self.kernel_size
                )
            )
            self.register_buffer(
                'mask',
                th.ones((self.kernel_size, self.kernel_size))[None, None, :, :]
            )

        if init == "dct":
            assert in_channels == 1

            def mat1d(n):
                r = th.arange(n)[..., None].to(th.float64)
                C = math.sqrt(2 / n) * th.cos((math.pi * r * (0.5 + r.T)) / n)
                C[0, :] = math.sqrt(1 / n)
                return C

            def mat2d(n):
                C = mat1d(n)
                return th.kron(C, C)

            dct_basis = mat2d(self.kernel_size)[1:][:self.out_channels].float()
            self.weight.data = dct_basis[:self.out_channels, None].reshape(
                self.out_channels, 1, self.kernel_size, self.kernel_size
            )
        else:
            th.nn.init.normal_(
                self.weight.data, 0.,
                np.sqrt(1 / (in_channels * kernel_size**2))
            )

        self.weight.L_init = 1e-4
        if zero_mean or bound_norm or ortho:
            self.weight.reduction_dim = (1, 2, 3)

            def proj(surface=False):
                if zero_mean:
                    mean = th.mean(
                        self.weight.data * self.mask, (1, 2, 3), keepdim=True
                    )
                    self.weight.data.sub_(mean)
                if ortho:
                    self.weight.data.copy_(
                        self.closest_ortho_basis_polar(
                            self.weight.data.reshape(
                                self.out_channels, self.in_channels,
                                kernel_size**2
                            ).permute(1, 2, 0)
                        ).permute(2, 0, 1).reshape(*self.weight.data.shape)
                    )
                if bound_norm:
                    norm = th.sum(
                        self.weight.data**2 * self.mask, (1, 2, 3), True
                    ).sqrt_()
                    if surface:
                        self.weight.data.div_(norm)
                    else:
                        self.weight.data.div_(th.max(norm, th.ones_like(norm)))

            # Call projection initially
            self.weight.proj = proj
            self.weight.proj(True)

        # Normalize regardless of projection
        # norm = (self.weight ** 2).sum((1, 2, 3), keepdim=True)
        # self.weight.data.div_(norm.sqrt() * 1.5)

    def closest_ortho_basis_polar(self, M_: th.Tensor) -> th.Tensor:
        M = M_.cpu()
        retval = th.empty_like(M)
        for in_ch in range(M.shape[0]):
            D = th.eye(self.out_channels, device=M.device, dtype=M.dtype)
            for _ in range(3):
                U, _, Vh = th.linalg.svd(M[in_ch] @ D, full_matrices=False)
                D = th.diag(th.diag(((U @ Vh).T @ M[in_ch]).clamp_(min=0)))
            retval[in_ch] = U @ Vh @ D
        return retval.cuda()

    def closest_ortho_basis_newton(
        self,
        M_,
    ):
        device = self.weight.device
        k = self.out_channels
        lamda = th.empty((k * (k - 1)) // 2,
                         device=self.weight.device).normal_() * 0.1
        indices_upper = th.triu_indices(
            k, k, offset=1, device=self.weight.device
        )

        def f(ll):
            la = th.zeros((k, k), device=device)
            la[indices_upper[0], indices_upper[1]] = ll
            la += la.clone().T
            x = th.linalg.inv(th.eye(k, device=device) + la)
            F = x.T @ M_.T @ M_ @ x
            return F[indices_upper[0], indices_upper[1]]

        for _ in range(15):
            B = f(lamda)
            J = th.autograd.functional.jacobian(f, lamda)
            delta = th.linalg.solve(J, B)
            lamda -= delta

        la = th.zeros((k, k), device=device)
        la[indices_upper[0], indices_upper[1]] = lamda
        la += la.clone().T
        return th.linalg.solve(th.eye(k, device=device) + la, M_.T).T

    def gram_schmidt(
        self,
        v: th.Tensor,
    ) -> th.Tensor:
        u = th.zeros_like(v)
        u[0] = v[0]
        for k in range(1, v.shape[0]):
            u[k] = v[k] - (((v[None, k] * u[:k]).sum(1) /
                            (u[:k]**2).sum(1))[:, None] * u[:k]).sum(0)
        return u

    def get_weight(self):
        if self.invariant:
            weight = th.empty(
                self.out_channels,
                self.in_channels,
                self.kernel_size,
                self.kernel_size,
                device=self.weight.device
            )
            weight[:, :, 1, 1] = self.weight[:, :, 0, 0]
            weight[:, :, ::2, ::2] = self.weight[:, :, 0, 2].view(
                self.out_channels, self.in_channels, 1, 1
            )
            weight[:, :, 1::2, ::2] = self.weight[:, :, 0, 1].view(
                self.out_channels, self.in_channels, 1, 1
            )
            weight[:, :, ::2, 1::2] = self.weight[:, :, 0, 1].view(
                self.out_channels, self.in_channels, 1, 1
            )
        else:
            weight = self.weight
        return weight

    def forward(self, x):
        weight = self.get_weight()

        return th.nn.functional.conv2d(
            x, weight, self.bias, self.stride, self.padding, self.dilation,
            self.groups
        )

    def backward(self, x, output_shape=None):
        weight = self.get_weight()

        if output_shape is not None:
            output_padding = (
                output_shape[2] - ((x.shape[2] - 1) * self.stride + 1),
                output_shape[3] - ((x.shape[3] - 1) * self.stride + 1)
            )
        else:
            output_padding = 0

        x = th.nn.functional.conv_transpose2d(
            x, weight, self.bias, self.stride, self.padding, output_padding,
            self.groups, self.dilation
        )
        return x

    def extra_repr(self):
        s = "({out_channels}, {in_channels}, {kernel_size}), invariant={invariant}"
        if self.stride != 1:
            s += ", stride={stride}"
        if self.dilation != 1:
            s += ", dilation={dilation}"
        if self.groups != 1:
            s += ", groups={groups}"
        if self.bias is not None:
            s += ", bias=True"
        if self.zero_mean:
            s += ", zero_mean={zero_mean}"
        if self.bound_norm:
            s += ", bound_norm={bound_norm}"
        return s.format(**self.__dict__)
