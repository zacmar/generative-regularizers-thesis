import math
from math import floor

import imageio.v3 as iio
import numpy as np
import torch as th


def mat1d(n):
    r = th.arange(n)[..., None].to(th.float64)
    C = math.sqrt(2 / n) * th.cos((math.pi * r * (0.5 + r.T)) / n)
    C[0, :] = math.sqrt(1 / n)
    return C


def kron(a, b):
    res = a.unsqueeze(1).unsqueeze(3) * b.unsqueeze(0).unsqueeze(2)
    return res.reshape((a.shape[0] * b.shape[0], a.shape[1] * b.shape[1]))


def mat2d(n):
    C = mat1d(n)
    return kron(C, C)


def generate_matrix(n):
    # Construct matrix to store the result
    matrix = np.zeros((n, n), dtype=int)

    # idx is a counter which will be the number that is written for each entry
    idx = 0

    # Iterate over the off diagonals
    for d_idx in range(2 * n - 1):
        # Compute the number of elements in the off diagonal
        num_elements_diag = d_idx + 1 if d_idx + 1 <= n else 2 * n - 1 - d_idx

        # Generate the entries of the off diagonal
        diag = np.zeros(num_elements_diag)
        for j in range(num_elements_diag):
            if j % 2 == 0:
                diag[floor(j // 2)] = idx
            else:
                diag[-floor(j // 2) - 1] = idx
            idx += 1

        # Copy the off diagonal back into the matrix
        for j in range(num_elements_diag):
            if d_idx + 1 <= n:
                matrix[d_idx - j][j] = diag[j]
            else:
                matrix[-1 - j][d_idx - n + 1 + j] = diag[j]

    return matrix


n = 7


if __name__ == '__main__':
    n = 7
    permutation = th.from_numpy(generate_matrix(n))
    C = mat2d(n).reshape(7,7,7,7)
    print(permutation)

    C_permuted = C.clone().float()
    for i in range(n):
        for j in range(n):
            # print(i, j)
            # print(C[permutation[i,j] // 7, permutation[i,j]%7].shape)
            C_permuted[permutation[i,j]//7, permutation[i,j]%7] = C[i, j]
    for i, c in enumerate(C_permuted.view(-1, n, n)):
        c = c.view(n, n)
        c -= c.min()
        c /= c.max()
        iio.imwrite(f'./dct_permute/{i}.png', (c * 255.).to(th.uint8).numpy())
