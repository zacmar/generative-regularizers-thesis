import numpy as np
import imageio.v3 as imageio
import scipy.signal as ss
import matplotlib.pyplot as plt

rng = np.random.default_rng()


def gkern(l=5, sig=1.):
    """\
    creates gaussian kernel with side length `l` and a sigma of `sig`
    """
    ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sig))
    kernel = np.outer(gauss, gauss)
    return kernel / np.sum(kernel)


x = imageio.imread('./006.png') / 255.
b = 7
kernel = rng.normal(size=(b, b))
kernel = gkern(b, 1)
conv = ss.convolve2d(x, kernel, mode='same', boundary='wrap')
conv_fourier = ss.fftconvolve(x, kernel, mode='same')

print((x[:b, :b] * np.rot90(kernel, 2)).sum())
print(conv[b // 2, b // 2])
print(conv_fourier[b // 2, b // 2])

# Compute the convolution in the Fourier domain
kernel_padded = np.zeros_like(x)
kernel_padded[x.shape[0] // 2 - b // 2:x.shape[0] // 2 + b // 2 + 1,
              x.shape[1] // 2 - b // 2:x.shape[1] // 2 + b // 2 + 1] = kernel
kernel_padded = np.fft.ifftshift(kernel_padded)
conv_fourier = np.fft.ifft2(np.fft.fft2(x) * np.fft.fft2(kernel_padded)).real
print(conv_fourier[b // 2, b // 2])

kernel_padded = np.zeros_like(x)
kernel_padded[:b, :b] = kernel
kernel_padded = np.roll(kernel_padded, (-b // 2 + 1, -b // 2 + 1), axis=(0, 1))
# kernel_padded = np.fft.ifftshift(kernel_padded)
conv_fourier = np.fft.ifft2(np.fft.fft2(x) * np.fft.fft2(kernel_padded)).real
print(conv_fourier[b // 2, b // 2])

plt.figure()
plt.imshow(conv, cmap='gray')
plt.figure()
plt.imshow(conv_fourier, cmap='gray')
plt.figure()
plt.imshow(conv - conv_fourier, cmap='gray')
plt.show()
