import imageio.v3 as imageio
import torch as th
import util
import scipy.signal as ss
import numpy as np
from skimage.metrics import structural_similarity as ssim

rng = np.random.default_rng(42)
Ssim = util.SSIM().double()

ref = imageio.imread('./reference.png') / 255.
mean = ref + 0.047
contrast = 1.287 * (ref - ref.mean()) + ref.mean()
impulse = ref.copy()
# Hehe
for i in range(ref.shape[0]):
    for j in range(ref.shape[1]):
        if rng.uniform() < 0.0038:
            impulse[i, j] = 1.0
        if rng.uniform() < 0.0038:
            impulse[i, j] = .0

blur = ss.convolve2d(ref, np.ones((4, 4)) / 16, mode='same', boundary='wrap')
# convert reference.png -quality 6 jpeg.jpg
jpeg = imageio.imread('./jpeg.jpg') / 255.

for other, name in zip([mean, contrast, impulse, blur, jpeg],
                       ['mean', 'contrast', 'impulse', 'blur', 'jpeg']):
    print(name)
    print('mse:  ', ((ref - other)**2).mean() * 100)
    print('ssim: ', Ssim(th.from_numpy(ref)[None, None], th.from_numpy(other)[None, None], data_range=1.0).mean().item())
    imageio.imwrite(
        f'./{name}.png', (other * 255).clip(0, 255).astype(np.uint8)
    )
