import numpy as np
import matplotlib.pyplot as plt
import imageio.v3 as iio

image = iio.imread('./reference.png') / 255.
s1 = np.ones_like(image)
s1[:, :image.shape[1] // 2] = 0
s2 = np.ones_like(image)
s2[:, image.shape[1] // 2:] = 0

mask = np.ones_like(image)
mask[:, 1::2] = 0

aliased = np.fft.ifft2(np.fft.fft2(image, norm='ortho') * mask, norm='ortho')
data1 = np.fft.fft2(image * s1, norm='ortho') * mask
data2 = np.fft.fft2(image * s2, norm='ortho') * mask
image1 = np.fft.ifft2(data1, norm='ortho').real
image2 = np.fft.ifft2(data2, norm='ortho').real
reco = np.zeros_like(image)
reco[:, :image.shape[1] // 2] = image2[:, :image.shape[1] // 2]
reco[:, image.shape[1] // 2:] = image1[:, image.shape[1] // 2:]
mask = mask[:41, :41]

for im, name in zip([image1, image2, reco, aliased, mask, s1, s2, image*s1, image*s2],
                    ['image1', 'image2', 'reco', 'aliased', 'mask', 's1', 's2', 'ims1', 'ims2']):
    im /= im.max()
    iio.imwrite(f'./{name}.png', (im * 255).astype(np.uint8))

plt.figure()
plt.imshow(mask)
plt.figure()
plt.imshow(image1)
plt.figure()
plt.imshow(image2)
plt.figure()
plt.imshow(aliased.real)
plt.figure()
plt.imshow(s1)
plt.figure()
plt.imshow(s1)
plt.figure()
plt.imshow(reco)
plt.figure()
plt.imshow(reco - image)
plt.show()
