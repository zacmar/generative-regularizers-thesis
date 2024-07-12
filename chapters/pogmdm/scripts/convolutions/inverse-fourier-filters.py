import numpy as np
import imageio.v3 as imageio
import matplotlib.pyplot as plt

filter = np.zeros((60, 30), dtype=complex)
filter[10:30,
       10:30] = np.random.normal(size=(20, 20)
                                 ) + 1j * np.random.normal(size=(20, 20))
# normalize to have unit energy
filter[10:30, 10:30] /= np.abs(filter[10:30, 10:30])
filter_time = np.fft.ifftshift(np.fft.irfft2(filter)).real
filter_time -= filter_time.min()
filter_time /= filter_time.max()

imageio.imwrite(
    'prescribed-spectrum/spectrum_magnitude.png',
    (np.abs(filter) * 255.).clip(0, 255).astype(np.uint8)
)
imageio.imwrite(
    'prescribed-spectrum/spectrum_phase.png',
    ((np.angle(filter) + np.pi) * 255. /
     (2 * np.pi)).clip(0, 255).astype(np.uint8)
)
imageio.imwrite(
    'prescribed-spectrum/time.png', (filter_time * 255.).astype(np.uint8)
)

b = 11
filter = np.zeros((60, 60))
x = np.zeros((60, 60))
filter[x.shape[0] // 2 - b // 2:x.shape[0] // 2 + b // 2 + 1,
    x.shape[1] // 2 - b // 2:x.shape[1] // 2 + b // 2 + 1] = np.random.normal(size=(b, b))
# kernel_padded = np.fft.ifftshift(kernel_padded)
filter_rolled = np.fft.ifftshift(filter)
spectrum = np.fft.rfft2(filter_rolled)
spectrum /= np.abs(spectrum).max()

imageio.imwrite(
    'prescribed-filter/spectrum_magnitude.png',
    (np.abs(spectrum) * 255.).clip(0, 255).astype(np.uint8)
)
imageio.imwrite(
    'prescribed-filter/spectrum_phase.png',
    ((np.angle(spectrum) + np.pi) * 255. /
     (2 * np.pi)).clip(0, 255).astype(np.uint8)
)
imageio.imwrite(
    'prescribed-filter/time.png', (filter * 255.).astype(np.uint8)
)
