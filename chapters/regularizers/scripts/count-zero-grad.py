import torch as th


def grad(x):
    g = x.new_zeros((*x.shape, 2))
    g[..., :, :-1, 0] += x[..., :, 1:] - x[..., :, :-1]
    g[..., :-1, :, 1] += x[..., 1:, :] - x[..., :-1, :]
    return g


x = th.load('reference-rss.pt')
print(x.max())
dx = grad(x)

# 640 entries are set to zero due to boundary conditions!
print((dx == 0).sum().item() - 640)

# Here more explicitly by removing the rows which are set to 0
dvx = dx[:, :-1, 0]
dhx = dx[:-1, :, 1]
print((dvx == 0).sum().item() + (dhx == 0).sum().item())

eps = 1e-7
print(((dvx.abs() - eps) < 0).sum().item() + ((dvx.abs() - eps) < 0).sum().item())
