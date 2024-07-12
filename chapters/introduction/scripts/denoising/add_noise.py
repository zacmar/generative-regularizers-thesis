import numpy as np
import imageio.v3 as imageio

im = imageio.imread('./natural.jpeg') / 255.
im += np.random.normal(0, 0.1, im.shape)
im = np.clip(im, 0, 1)
im = (im * 255).astype(np.uint8)
imageio.imwrite('./noisy.png', im)
print(im.max())
