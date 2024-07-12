import os
from pathlib import Path

import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as ss
import torch as th

kh, kw = 7, 7
dh, dw = 1, 1

# image_dirs = [
#     Path(os.environ.get('DATASETS_ROOT')) / 'bsds500' / sset
#     for sset in ['train', 'val']
# ]
# image_files = []
# for image_dir in image_dirs:
#     image_files += list(image_dir.glob('*.jpg'))


# patches = []
# for i, file in enumerate(image_files):
#     img = th.from_numpy(iio.imread(file) / 255.).mean(-1)
#     patches_ = img.unfold(0, kh, dh).unfold(1, kw, dw)
#     patches.append(patches_.reshape(-1, kh * kw))

# # Thats how many fit on my GPU
# patches = th.cat(patches)[:30_000_000]
# patches -= patches.mean(1, keepdim=True)
# _, sigmas, vs = th.linalg.svd(patches, full_matrices=False)
# np.savetxt('pca/sigmas.txt', sigmas[:-1].numpy(), delimiter=',', fmt='%.2f')

# vs = vs.view(49, 7, 7)
# hists = th.zeros((48, 200), dtype=th.float64)
# bin_edges = th.linspace(-1, 1, 202, dtype=th.float64)
# x_hist = (bin_edges[1:] + bin_edges[:-1]) / 2
# th.save(vs, './pca/directions/vs.pt')
vs = th.load('./pca/directions/vs.pt').numpy()
# vs = np.random.randn(49, 7, 7)
# vs -= vs.mean((1, 2), keepdims=True)
vs_convolved = []
vc_sum = np.zeros((13, 13))
for i, v in enumerate(vs[:-1]):
    vs_convolved.append(ss.convolve2d(v, v, mode='full'))

    print(vs_convolved[i].shape)
    vc_sum += vs_convolved[i]

invfilter = np.fft.fftshift(np.fft.ifft2(np.conj(np.fft.fft2(vc_sum)) / (np.abs(np.fft.fft2(vc_sum))**2 + 1e-8)).real)


laplace = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
lap = laplace.copy()
for _ in range(5):
    lap = ss.convolve2d(lap, laplace, mode='full')
    print(lap.shape)

plt.figure()
plt.imshow(vc_sum)
plt.figure()
plt.imshow(np.log(np.abs(np.fft.fftshift(np.fft.fft2(vc_sum)))))
plt.figure()
plt.plot(vc_sum[6])
plt.figure()
plt.imshow(lap)
plt.figure()
plt.imshow(np.log(np.abs(np.fft.fftshift(np.fft.fft2(lap)))))
plt.figure()
plt.imshow(invfilter)
plt.show()


for i, v in enumerate(vs):
    hist, _ = th.histogram(
        (v * patches.view(-1, 7, 7)).sum((-2, -1)), bin_edges)
    hist = -th.log(hist)
    np.savetxt(f'./pca/directions/{i}/hist.csv',
               th.stack((x_hist, hist)).T.numpy(), delimiter=',')
    v -= v.min()
    v /= v.max()
    iio.imwrite(f'./pca/directions/{i}/d_{i}.png',
                (v * 255.).to(th.uint8).numpy())
