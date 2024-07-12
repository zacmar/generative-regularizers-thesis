import numpy as np
import matplotlib.pyplot as plt


def onedgmm(x, mus, var):
    return np.exp(-(x[None, :] - mus[:, None])**2 /
                  (2 * var)).sum(0) / np.sqrt(2 * np.pi * var)


k1 = np.array([1., 0.])
k2 = np.array([1., 1.]) / np.sqrt(2)
# k2 = np.array([0., 1.])

mus = np.array([-2, 2])
sigma_0 = 1

x = np.linspace(-10, 10, 1000)
dx = x[1] - x[0]
xx, yy = np.meshgrid(x, x)

xy = np.c_[xx.ravel(), yy.ravel()]

g1 = onedgmm((xy * k1[None]).sum(1), mus, sigma_0**2).reshape(xx.shape)
g2 = onedgmm((xy * k2[None]).sum(1), mus, sigma_0**2).reshape(xx.shape)
t0 = g1 * g2
t0norm = t0 / (t0.sum())
# plt.figure()
# plt.contourf(xx, yy, t0norm)

g1 = onedgmm((xy * k1[None]).sum(1), mus, sigma_0**2 + 1).reshape(xx.shape)
g2 = onedgmm((xy * k2[None]).sum(1), mus, sigma_0**2 + 1).reshape(xx.shape)
t05_wrong = g1 * g2
plt.figure()
plt.contourf(xx, yy, t05_wrong / t05_wrong.sum())

precision = (k1[None] * k1[:, None] + k2[None] * k2[:, None]) / (sigma_0**2)
print(k1[None] * k1[:, None])
print(k2[None] * k2[:, None])
covariance = np.linalg.inv(precision)
print(precision)
print(covariance)
print(covariance + np.eye(2))
print(np.linalg.inv(covariance + np.eye(2)))
print(1/7* np.array([[4,1],[1,2]]))
means = np.stack([
    covariance @ (k1 * mus[0] + k2 * mus[0]) / sigma_0**2,
    covariance @ (k1 * mus[1] + k2 * mus[0]) / sigma_0**2,
    covariance @ (k1 * mus[0] + k2 * mus[1]) / sigma_0**2,
    covariance @ (k1 * mus[1] + k2 * mus[1]) / sigma_0**2,
])


def twodgmm(x, mus, cov):
    d = (x[:, None] - mus[None]) @ np.linalg.inv(cov)
    dist = ((x[:, None] - mus[None]) * d).sum(2)
    return np.exp(-dist / 2
                  ).sum(1) / np.sqrt(np.linalg.det(2 * np.pi * cov)) / 4


t0 = twodgmm(xy, means, np.linalg.inv(precision)).reshape(xx.shape)
# plt.figure()
# plt.contourf(xx, yy, t0)
print(t0.sum() * dx**2)

t05_right = twodgmm(xy, means, covariance + np.eye(2)).reshape(xx.shape)
plt.figure()
plt.contourf(xx, yy, t05_right / t05_right.sum())
plt.show()
