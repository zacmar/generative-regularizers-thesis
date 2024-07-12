import torch
import numpy as np
from scipy import signal as signal
import matplotlib.pyplot as plt
from torch._C import Value


def dmaxflat(N, d):
    """
    THIS IS A REWRITE OF THE ORIGINAL MATLAB IMPLEMENTATION OF dmaxflat.m
    FROM THE Nonsubsampled Contourlet Toolbox.   -- Stefan Loock, Dec 2016.
    returns 2-D diamond maxflat filters of order 'N'
    the filters are nonseparable and 'd' is the (0,0) coefficient, being 1 or 0
    depending on use.
    by Arthur L. da Cunha, University of Illinois Urbana-Champaign
    Aug 2004
    """
    if (N > 7) or (N < 1):
        raise ValueError('Error: N must be in {1,2,...,7}')

    if N == 4:
        h = np.array([[0, -5, 0, -3, 0], [-5, 0, 52, 0, 34], [
            0, 52, 0, -276, 0
        ], [-3, 0, -276, 0, 1454], [0, 34, 0, 1454, 0]]) / np.power(2, 12)
        h = np.append(h, np.fliplr(h[:, 0:-1]), 1)
        h = np.append(h, np.flipud(h[0:-1, :]), 0)
        h[4, 4] = d
    else:
        raise ValueError('Not implemented!')
    return h


def mctrans(b, t):
    """
    This is a translation of the original Matlab implementation of mctrans.m
    from the Nonsubsampled Contourlet Toolbox by Arthur L. da Cunha.
    MCTRANS McClellan transformation
        H = mctrans(B,T)
    produces the 2-D FIR filter H that corresponds to the 1-D FIR filter B
    using the transform T.
    Convert the 1-D filter b to SUM_n a(n) cos(wn) form
    Part of the Nonsubsampled Contourlet Toolbox
    (http://www.mathworks.de/matlabcentral/fileexchange/10049-nonsubsampled-contourlet-toolbox)
    """

    # Convert the 1-D filter b to SUM_n a(n) cos(wn) form
    # if mod(n,2) != 0 -> error
    n = (b.size - 1) // 2

    b = np.fft.fftshift(b[::-1])  #inverse fftshift
    b = b[::-1]
    a = np.zeros(n + 1)
    a[0] = b[0]
    a[1:n + 1] = 2 * b[1:n + 1]

    inset = np.floor((np.asarray(t.shape) - 1) / 2)
    inset = inset.astype(int)
    # Use Chebyshev polynomials to compute h
    P0 = 1
    P1 = t
    h = a[1] * P1
    rows = int(inset[0] + 1)
    cols = int(inset[1] + 1)
    h[rows - 1, cols - 1] = h[rows - 1, cols - 1] + a[0] * P0
    for i in range(3, n + 2):
        P2 = 2 * signal.convolve2d(t, P1)
        rows = (rows + inset[0]).astype(int)
        cols = (cols + inset[1]).astype(int)
        if i == 3:
            P2[rows - 1, cols - 1] = P2[rows - 1, cols - 1] - P0
        else:
            P2[rows[0] - 1:rows[-1],
               cols[0] - 1:cols[-1]] = P2[rows[0] - 1:rows[-1],
                                          cols[0] - 1:cols[-1]] - P0
        rows = inset[0] + np.arange(np.asarray(P1.shape)[0]) + 1
        rows = rows.astype(int)
        cols = inset[1] + np.arange(np.asarray(P1.shape)[1]) + 1
        cols = cols.astype(int)
        hh = h
        h = a[i - 1] * P2
        h[rows[0] - 1:rows[-1], cols[0] -
          1:cols[-1]] = h[rows[0] - 1:rows[-1], cols[0] - 1:cols[-1]] + hh
        P0 = P1
        P1 = P2
    h = np.rot90(h, 2)
    return h


def modulate2(x, type, center=np.array([0, 0])):
    """
    THIS IS A REWRITE OF THE ORIGINAL MATLAB IMPLEMENTATION OF
    modulate2.m FROM THE Nonsubsampled Contourlet Toolbox.
    MODULATE2	2D modulation
            y = modulate2(x, type, [center])
    With TYPE = {'r', 'c' or 'b'} for modulate along the row, or column or
    both directions.
    CENTER secify the origin of modulation as floor(size(x)/2)+1+center
    (default is [0, 0])
    Part of the Nonsubsampled Contourlet Toolbox
    (http://www.mathworks.de/matlabcentral/fileexchange/10049-nonsubsampled-contourlet-toolbox)
    """
    size = np.asarray(x.shape)
    if x.ndim == 1:
        if np.array_equal(center, [0, 0]):
            center = 0
    origin = np.floor(size / 2) + 1 + center
    n1 = np.arange(size[0]) - origin[0] + 1
    if x.ndim == 2:
        n2 = np.arange(size[1]) - origin[1] + 1
    else:
        n2 = n1
    if type == 'r':
        m1 = np.power(-1, n1)
        if x.ndim == 1:
            y = x * m1
        else:
            y = x * np.transpose(np.tile(m1, (size[1], 1)))
    elif type == 'c':
        m2 = np.power(-1, n2)
        if x.ndim == 1:
            y = x * m2
        else:
            y = x * np.tile(m2, np.array([size[0], 1]))
    elif type == 'b':
        m1 = np.power(-1, n1)
        m2 = np.power(-1, n2)
        m = np.outer(m1, m2)
        if x.ndim == 1:
            y = x * m1
        else:
            y = x * m
    return y


def dfilters(fname, type):
    """
    generate directional 2D filters

    input: 
        fname: filter names, default: 'dmaxflat' (maximally flat 2D fan filter)
        type: 'd' or 'r' for decomposition or reconstruction filters

    output:
        h0, h1: diamond filter pair (lowpass and highpass)
    """

    if fname == 'dmaxflat4':
        M1 = 1 / np.sqrt(2)
        M2 = M1
        k1 = 1 - np.sqrt(2)
        k3 = k1
        k2 = M1
        h = np.array([0.25 * k2 * k3, 0.5 * k2, 1 + 0.5 * k2 * k3]) * M1
        h = np.append(h, h[-2::-1])
        g = np.array([
            -0.125 * k1 * k2 * k3, 0.25 * k1 * k2,
            -0.5 * k1 - 0.5 * k3 - 0.375 * k1 * k2 * k3, 1 + 0.5 * k1 * k2
        ]) * M2
        g = np.append(g, h[-2::-1])

        B = dmaxflat(4, 0)
        h0 = mctrans(h, B)
        g0 = mctrans(g, B)

        h0 = np.sqrt(2) * h0 / np.sum(h0)
        g0 = np.sqrt(2) * g0 / np.sum(g0)

        h1 = modulate2(g0, 'b')
        if type == 'r':
            h1 = modulate2(h0, 'b')
            h0 = g0
    else:
        raise ValueError("Filter type not implemented!")

    return h0, h1


if __name__ == "__main__":
    h0, h1 = dfilters('dmaxflat4', 'd')
    print('h0 shape:', h0.shape)
    print('h1 shape:', h1.shape)

    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(h0), ax[0].set_title('h0')
    ax[1].imshow(h1), ax[1].set_title('h1')
    plt.show()
