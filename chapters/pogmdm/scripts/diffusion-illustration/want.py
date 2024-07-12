import numpy as np


prec_diff = np.array([
    [1, 0.1],
    [0.1, .5],
])
cov_diff = np.linalg.inv(prec_diff)
cov = cov_diff - np.eye(2)
prec = np.linalg.inv(cov)
print(prec)
