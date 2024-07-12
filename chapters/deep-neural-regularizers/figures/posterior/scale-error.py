import imageio.v3 as iio
import numpy as np
import matplotlib.pyplot as plt

cm = plt.get_cmap('inferno')


for mask in ['random', 'cartesian', 'spiral', 'radial']:
    for cherry in ['file1000702.h5', 'file1000831.h5']:
        reco = iio.imread(f'{mask}/{cherry}/mean.png') / 255.
        gt = iio.imread(f'{mask}/{cherry}/gt.png') / 255.
        error = 10 * np.abs(reco - gt)
        colored_image = cm(error)[:, :, :3]
        plt.figure()
        plt.imshow(colored_image)
        plt.show()
        iio.imwrite(f'{mask}/{cherry}/error_scaled.png', ((colored_image) * 255).clip(0, 255).astype(np.uint8))
