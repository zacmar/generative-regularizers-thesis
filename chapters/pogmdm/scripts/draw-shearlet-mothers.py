import imageio.v3 as iio
import numpy as np
import torch as th
import shltutils.filters as filters


sigmas = [0, .025, .05, .1, .2]
N = len(sigmas)

h = [
    0.0104933261758410, -0.0263483047033631, -0.0517766952966370,
    0.276348304703363, 0.582566738241592, 0.276348304703363,
    -0.0517766952966369, -0.0263483047033631, 0.0104933261758408
]
h0, _ = filters.dfilters('dmaxflat4', 'd')
P = filters.modulate2(h0, 'c')

dict = th.load('./state_final.pth', map_location=th.device('cpu'))
print(h, dict['h'].numpy().tolist(), sep='\n')

P_learned = dict['P'].cpu().numpy()
P_learned_normalized = P_learned - P_learned.min()
P_learned_normalized /= P_learned_normalized.max()


P_normalized = P - P.min()
P_normalized /= P_normalized.max()

iio.imwrite('p-learned.png', (P_learned_normalized * 255.).astype(np.uint8))
iio.imwrite('p-initial.png', (P_normalized * 255.).astype(np.uint8))