import numpy as np
import torch


def roll(x, shift, dim):
    """
    Similar to np.roll but applies to PyTorch Tensors
    https://github.com/facebookresearch/fastMRI/blob/master/data/transforms.py
    """
    if isinstance(shift, (tuple, list)):
        assert len(shift) == len(dim)
        for s, d in zip(shift, dim):
            x = roll(x, s, d)
        return x
    shift = shift % x.size(dim)
    if shift == 0:
        return x
    left = x.narrow(dim, 0, x.size(dim) - shift)
    right = x.narrow(dim, x.size(dim) - shift, shift)
    return torch.cat((right, left), dim=dim)


def dshear(inputArray, k, axis):
    """
    Computes the discretized shearing operator for a given inputArray, shear
    number k and axis.
    This version is adapted such that the MATLAB indexing can be used here in the
    Python version.
    """
    if k == 0:
        return inputArray
    rows = np.asarray(inputArray.shape)[0]
    cols = np.asarray(inputArray.shape)[1]

    shearedArray = torch.zeros((rows, cols),
                               dtype=inputArray.dtype,
                               device=inputArray.device)

    if axis == 0:
        for col in range(cols):
            shearedArray[:, col] = torch.roll(
                inputArray[:, col], int(k * np.floor(cols / 2 - col))
            )
    else:
        for row in range(rows):
            shearedArray[row, :] = torch.roll(
                inputArray[row, :], int(k * np.floor(rows / 2 - row))
            )
    return shearedArray


def padArray(array, newSize):
    """
    Implements the padding of an array as performed by the Matlab variant.
    """
    if np.isscalar(newSize):
        #padSizes = np.zeros((1,newSize))
        # check if array is a vector...
        currSize = array.size
        paddedArray = torch.zeros(
            newSize, dtype=array.dtype, device=array.device
        )
        sizeDiff = newSize - currSize
        idxModifier = 0
        if sizeDiff < 0:
            raise ValueError(
                "Error: newSize is smaller than actual array size."
            )
        if sizeDiff == 0:
            print("Warning: newSize is equal to padding size.")
        if sizeDiff % 2 == 0:
            padSizes = sizeDiff // 2
        else:
            padSizes = int(np.ceil(sizeDiff / 2))
            if currSize % 2 == 0:
                # index 1...k+1
                idxModifier = 1
            else:
                # index 0...k
                idxModifier = 0
        print(padSizes)
        paddedArray[padSizes - idxModifier:padSizes + currSize -
                    idxModifier] = array

    else:
        padSizes = torch.zeros(
            newSize.size, dtype=array.dtype, device=array.device
        )
        paddedArray = torch.zeros((newSize[0], newSize[1]),
                                  dtype=array.dtype,
                                  device=array.device)
        idxModifier = np.array([0, 0])
        currSize = np.asarray(array.shape)
        if array.ndim == 1:
            currSize = np.array([len(array), 0])
        for k in range(newSize.size):
            sizeDiff = newSize[k] - currSize[k]
            if sizeDiff < 0:
                raise ValueError(
                    "Error: newSize is smaller than actual array size in dimension "
                    + str(k) + "."
                )
            if sizeDiff == 0:
                print(
                    "Warning: newSize is equal to padding size in dimension " +
                    str(k) + "."
                )
            if sizeDiff % 2 == 0:
                padSizes[k] = sizeDiff // 2
            else:
                padSizes[k] = np.ceil(sizeDiff / 2)
                if currSize[k] % 2 == 0:
                    # index 1...k+1
                    idxModifier[k] = 1
                else:
                    # index 0...k
                    idxModifier[k] = 0
        padSizes = padSizes.int()

        # if array is 1D but paddedArray is 2D we simply put the array (as a
        # row array in the middle of the new empty array). this seems to be
        # the behavior of the ShearLab routine from matlab.
        if array.ndim == 1:
            paddedArray[padSizes[1], padSizes[0]:padSizes[0] + currSize[0] +
                        idxModifier[0]] = array
        else:
            paddedArray[padSizes[0] - idxModifier[0]:padSizes[0] +
                        currSize[0] - idxModifier[0], padSizes[1]:padSizes[1] +
                        currSize[1] + idxModifier[1]] = array
    return paddedArray


def upsample_np(array, dims, nZeros):
    """
    Performs an upsampling by a number of nZeros along the dimenion(s) dims
    for a given array.
    """
    assert dims == 0 or dims == 1

    if array.ndim == 1:
        sz = len(array)
        idx = range(1, sz)
        arrayUpsampled = np.insert(array, idx, 0)
    else:
        sz = np.asarray(array.shape)
        if dims == 0:
            arrayUpsampled = np.zeros(((sz[0] - 1) * (nZeros + 1) + 1, sz[1]))
            for col in range(sz[0]):
                arrayUpsampled[col * (nZeros) + col, :] = array[col, :]
        if dims == 1:
            arrayUpsampled = np.zeros(
                (sz[0], ((sz[1] - 1) * (nZeros + 1) + 1))
            )
            for row in range(sz[1]):
                arrayUpsampled[:, row * (nZeros) + row] = array[:, row]
    return torch.Tensor(arrayUpsampled)


def upsample(array, dims, nZeros):
    """
    Performs an upsampling by a number of nZeros along the dimenion(s) dims
    for a given array.
    """
    assert dims == 0 or dims == 1

    if array.ndim == 1:
        sz = len(array)
        array_zero = torch.zeros((sz - 1),
                                 dtype=array.dtype,
                                 device=array.device)
        arrayUpsampled = torch.empty((2 * sz - 1),
                                     dtype=array.dtype,
                                     device=array.device)
        arrayUpsampled[0::2] = array
        arrayUpsampled[1:-1:2] = array_zero
    else:
        sz = np.asarray(array.shape)
        # behaves like in matlab: dims == 1 and dims == 2 instead of 0 and 1.
        if dims == 0:
            arrayUpsampled = torch.zeros(((sz[0] - 1) *
                                          (nZeros + 1) + 1, sz[1]),
                                         dtype=array.dtype,
                                         device=array.device)
            for col in range(sz[0]):
                arrayUpsampled[col * (nZeros) + col, :] = array[col, :]
        if dims == 1:
            arrayUpsampled = torch.zeros(
                (sz[0], ((sz[1] - 1) * (nZeros + 1) + 1)),
                dtype=array.dtype,
                device=array.device
            )
            for row in range(sz[1]):
                arrayUpsampled[:, row * (nZeros) + row] = array[:, row]
    return arrayUpsampled


if __name__ == "__main__":
    # test upsample vs upsample_np (reference)
    # 1D
    a_1D_np = np.random.randn(100)
    a_1D_torch = torch.from_numpy(a_1D_np)

    res1 = upsample_np(a_1D_np, 0, 0)
    res2 = upsample(a_1D_torch, 0, 0)
    print('1D upsampling same: ', np.allclose(res1, res2.cpu().numpy()))

    # 2D
    a_2D_np = np.random.randn(100, 100)
    a_2D_torch = torch.from_numpy(a_2D_np)

    res1 = upsample_np(a_2D_np, 1, 3)
    res2 = upsample(a_2D_torch, 1, 3)
    print('2D upsampling same: ', np.allclose(res1, res2.cpu().numpy()))
