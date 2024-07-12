import imageio.v3 as iio
import numpy as np

for i in range(10):
    noise_map = iio.imread(f'{i}/estimate.png')
    padded = np.pad(noise_map, (3, 3), mode='constant', constant_values=0)
    iio.imwrite(f'{i}/estimate_padded.png', padded)
