import numpy as np
import torch as th

xs = th.load('./patches.pth').cuda()
xs = xs.view(-1, 2, 2)
d_h = xs[:, :, 1:] - xs[:, :, :-1]
d_v = xs[:, 1:, :] - xs[:, :-1, :]

n_bins = 301
_min = -0.4
_max = 0.4
x = th.linspace(_min, _max, n_bins)
for diff, name in zip([d_h, d_v], ['d_h', 'd_v']):
    hist = -th.log(th.histc(diff, bins=n_bins, min=_min, max=_max))
    data = np.vstack([x.cpu().numpy(), hist.cpu().numpy()]).T
    print(data.shape)

    np.savetxt(f'./{name}.csv', data, delimiter=',')
