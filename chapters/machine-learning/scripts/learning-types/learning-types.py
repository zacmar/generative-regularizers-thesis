import numpy as np
import sklearn.mixture as mix
import scipy.cluster.vq as vq
import matplotlib.pyplot as plt

mu0 = [-1, -1]
mu1 = [1, 1]

sigma0 = [[1, 0.5], [0.5, 1]]
sigma1 = [[1, -0.5], [-0.5, 1]]

n_1 = 200
n_2 = 300
# construct gmm with 2 components
x = np.random.multivariate_normal(mu0, sigma0, n_1)
y = np.random.multivariate_normal(mu1, sigma1, n_2)

features = np.vstack((x, y))

codebook, _ = vq.kmeans(features, 2)

gm = mix.GaussianMixture(n_components=2, covariance_type='full').fit(features)

xaxis = np.linspace(-5, 5, 200)
xx, yy = np.meshgrid(xaxis, xaxis)
grid = np.c_[xx.ravel(), yy.ravel()]
p = gm.score_samples(grid)

# Visualize gmm
fig, ax = plt.subplots()
ax.set_aspect('equal')
plt.axis('off')
ax.contourf(xx, yy, p.reshape(200, 200), cmap='Greens', alpha=1)
ax.scatter(x[:, 0], x[:, 1], color=[0, 0, 0], alpha=0.5)
ax.scatter(y[:, 0], y[:, 1], color=[0.8, 0.8, 0.8], alpha=0.5)
ax.scatter(
    gm.means_[:, 0], gm.means_[:, 1], c='w', marker='x', s=200, linewidths=3
)
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
plt.savefig('gmm.pdf', bbox_inches='tight')

# visualize decision boundary of kmeans
# compute the decision boundary
xmin =-5
midway = (codebook[0] + codebook[1]) / 2
direction = codebook[0] - codebook[1]
direction[0], direction[1] = direction[1], -direction[0]
p1 = midway + 3 * direction
p2 = midway - 3 * direction
p3 = [5, 5]
fig, ax = plt.subplots()
ax.set_aspect('equal')
plt.axis('off')
ax.fill([p1[0], p2[0], p3[0]], [p1[1], p2[1], p3[1]], color=[90/255, 183/255, 105/255])
ax.fill([p1[0], p2[0], -p3[0]], [p1[1], p2[1], -p3[1]], color=[17/255, 123/255, 56/255])
ax.scatter(x[:, 0], x[:, 1], color=[0, 0, 0], alpha=0.5)
ax.scatter(y[:, 0], y[:, 1], color=[0.8, 0.8, 0.8], alpha=0.5)
ax.scatter(
    codebook[:, 0], codebook[:, 1], c='w', marker='x', s=200, linewidths=3
)
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
plt.savefig('km.pdf', bbox_inches='tight')


# Visualize class posterior
pcs = np.exp(gm._estimate_log_prob(grid))
fig, ax = plt.subplots()
ax.contour(xx, yy, pcs[:, 0].reshape(200, 200), alpha=1, cmap='Greens')
ax.contour(xx, yy, pcs[:, 1].reshape(200, 200), alpha=1, cmap='Reds')
ax.set_aspect('equal')
plt.axis('off')
ax.scatter(x[:, 0], x[:, 1], color=[0, 0, 0], alpha=0.5)
ax.scatter(y[:, 0], y[:, 1], color=[0.8, 0.8, 0.8], alpha=0.5)
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.scatter(
    gm.means_[0, 0], gm.means_[0, 1], color=[0/255,94/255,30/255], marker='x', s=100, linewidths=3
)
ax.scatter(
    gm.means_[1, 0], gm.means_[1, 1], color=[154/255,0,0], marker='x', s=2/3*100, linewidths=3
)
plt.savefig('generative.pdf', bbox_inches='tight')


fig, ax = plt.subplots()
# pcs = gm.predict_proba(grid)
pcs = np.exp(gm._estimate_weighted_log_prob(grid))
pcs = (pcs / np.sum(pcs, axis=1)[:, None]) 
ax.contourf(xx, yy, pcs[:, 0].reshape(200, 200), alpha=1, cmap='Greens')
ax.set_aspect('equal')
plt.axis('off')
ax.scatter(x[:, 0], x[:, 1], color=[0, 0, 0], alpha=0.5)
ax.scatter(y[:, 0], y[:, 1], color=[0.8, 0.8, 0.8], alpha=0.5)
ax.contour(xx, yy, pcs[:, 0].reshape(200, 200), alpha=1, cmap='Reds', levels=[0.5], linewidths=3)
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
plt.savefig('discriminative.pdf', bbox_inches='tight')
plt.show()
