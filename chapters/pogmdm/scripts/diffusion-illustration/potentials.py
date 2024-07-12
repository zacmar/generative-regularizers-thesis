import torch as th
import numpy as np
import matplotlib.pyplot as plt

L = 125
eta = 1
sigma_0 = 2 * eta / (L - 1)
mus = th.linspace(-eta, eta, L, dtype=th.float64)
s = .3
x = th.linspace(-.9, .9, L, dtype=th.float64)
ws = th.exp((th.exp(-x**2/(2*s**2)) * (-s**2 + x**2))/s**4)
# ws = th.exp(-th.abs(th.linspace(-.9, .9, L, dtype=th.float64)))
ws /= th.sum(ws)

x = th.linspace(-.9, .9, 1000, dtype=th.float64)



def gmm(x, mus, sigma, ws):
    return th.sum(
        ws[None] * th.exp(-(x[:, None] - mus[None])**2 / (2 * sigma**2)) /
        np.sqrt(2 * np.pi * sigma**2),
        axis=1
    )



y = gmm(x, mus, sigma_0, ws)
# plt.figure()
# for mu, w in zip(mus, ws):
#     ys = gmm(x, np.array([mu]), sigma_0, np.array([w]))
#     ys[ys<1e-5] = np.nan
#     plt.plot(x, ys)
# plt.plot(x, y)
#
# plt.figure()
# for mu, w in zip(mus, ws):
#     ys = gmm(x, np.array([mu]), sigma_0, np.array([w]))
#     plt.plot(x, (-np.log(ys)).clip(0, 2))
# plt.plot(x, -np.log(y))

# plt.show()

def apgd(
    x_init: th.Tensor,
    f_nabla,
    callback=lambda x, i: None,
    max_iter: int = 100000,
    gamma=0,
):
    print(x_init)
    x = x_init.clone()
    x_old = x.clone()
    L = 1
    beta = 1 / np.sqrt(2)
    for i in range(max_iter):
        x_bar = x + beta * (x - x_old)
        x_old = x.clone()
        energy, grad = f_nabla(x_bar)
        for _ in range(200):
            x = x_bar - grad / L
            dx = x - x_bar
            bound = energy + (grad * dx).sum() + L * (dx * dx).sum() / 2
            if f_nabla(x)[0] < bound:
                break
            L = 2 * L
        else:
            break

        L /= 1.5
        callback(x, i)
    return x


def f_nabla(ws):
    ws.requires_grad = True
    ours = -th.log(gmm(x, mus, sigma_0, ws))
    # student t
    # alpha = 0.05
    # wanted = (alpha + 1) / 2 * th.log(1 + x**2/alpha)
    # abs
    # wanted = th.abs(x)
    # mexican hat
    s = .3
    wanted = (-(th.exp(-x**2/(2*s**2)) * (-s**2 + x**2))/s**4) * .1
    loss = ((ours - wanted) ** 2).sum()
    gw = th.autograd.grad(loss, ws)[0]
    ws.requires_grad = False
    return loss.detach(), gw.detach()


def callback(ws, i):
    print(f_nabla(ws)[0])

ws = apgd(ws, f_nabla, callback=callback, max_iter=10000)
components = []

plt.figure()
for mu, w in zip(mus, ws):
    components.append(gmm(x, mu, sigma_0, w).detach().numpy())
full = gmm(x, mus, sigma_0, ws).detach().numpy()
neglog = -np.log(full)

all = np.concatenate((x[None], np.array(components), full[None], neglog[None])).T
np.savetxt(f"mhat.csv", all, delimiter=",")

plt.figure()
for mu, w in zip(mus, ws):
    ys = gmm(x, mu, sigma_0, w)
    plt.plot(x.detach().numpy(), (-th.log(ys)).clip(0, 2).detach().numpy())
plt.plot(x.detach().numpy(), -np.log(y).detach().numpy())

plt.show()
