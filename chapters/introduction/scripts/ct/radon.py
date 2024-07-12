import imageio.v3 as iio
import numpy as np
from skimage.transform import radon

image = iio.imread('./anatomy.png') / 255.
print(image.shape, image.max())
theta = np.linspace(0.0, 180.0, 256, endpoint=False)
sinogram = radon(image, theta=theta)
sinogram = (sinogram - sinogram.min()) / (sinogram.max() - sinogram.min())
sinogram *= 255
sinogram = sinogram.astype(np.uint8)
iio.imwrite('./sinogram.png', sinogram)
