import os
import random
from pathlib import Path

import imageio
import numpy as np
import skimage.transform as sktr
import torch as th
import torchvision.transforms.functional as ttf


class Set68():
    def __init__(
        self,
        color: bool = False,
        device: th.device = th.device('cuda'),
    ):
        n_ch = 3 if color else 1
        self.images = th.empty((68, n_ch, 321, 481), device=device)
        path = Path(os.environ['DATASETS_ROOT']) / 'set68'
        for i, p in enumerate(sorted(path.iterdir())):
            image = imageio.imread(str(p)) / 255
            image = image if color else image.mean(-1, keepdims=True)
            image = image if image.shape[0] == 321 else np.rot90(
                image, axes=(0, 1)
            )
            self.images[i] = th.from_numpy(image.copy()).permute(2, 0, 1)

    def data(self):
        ims = ttf.center_crop(self.images, (320, 320))
        return ims


class CelebA():
    def __init__(
        self,
        color: bool = False,
        size: int = 64,
        device: th.device = th.device('cuda'),
        n_im: int = 30_000,
        batch_size: int = 64,
        data: th.Tensor | None = None,
    ):
        n_ch = 3 if color else 1
        self.images = th.empty((n_im, n_ch, size, size), device=device)
        self.batch_size = batch_size
        self.n_im = n_im
        if data is None:
            path = Path(
                os.environ['DATASETS_ROOT']
            ) / 'CelebAMask-HQ' / 'CelebA-HQ-img'
            for i, p in enumerate(path.iterdir()):
                print(i)
                image = sktr.resize(
                    imageio.imread(str(p)) / 255,
                    output_shape=(size, size),
                    anti_aliasing=True,
                )

                if not color:
                    image = image.mean(-1, keepdims=True)
                self.images[i] = th.from_numpy(image).permute(2, 0, 1)
                if i == n_im - 1:
                    break
        else:
            self.images = data.clone()

    def data(self, ) -> th.Tensor:
        return self.images

    def __iter__(self, ) -> th.Tensor:
        while True:
            ims = random.sample(list(range(self.n_im)), k=self.batch_size)
            yield self.images[th.tensor(ims).long()]


class BSDS():
    def __init__(
        self,
        color: bool = False,
        batch_size: int = 50,
        patch_size: int = 90,
        rotate: bool = True,
        flip: bool = True,
    ):
        self.images = []
        self.ch = 3 if color else 1
        self.patch_size = patch_size
        self.batch_size = batch_size
        base = Path(os.environ['DATASETS_ROOT']) / 'bsds500'
        for set in ['train', 'test']:
            path = base / set
            for im in path.iterdir():
                image = imageio.imread(str(im)) / 255
                if not color:
                    image = image.mean(-1, keepdims=True)
                self.images.append(image)

        self.flips = [
            lambda x: x, lambda x: x[::-1, :, :], lambda x: x[:, ::-1, :],
            lambda x: x[::-1, ::-1, :]
        ]
        self.rng = np.random.default_rng(42)

        self._transforms = [self.crop]
        if rotate:
            self._transforms.append(self.rotate)
        if flip:
            self._transforms.append(self.flip)

    def transform(
        self,
        im: np.ndarray,
    ):
        for tr in self._transforms:
            im = tr(im)

        return im

    def rotate(
        self,
        im: np.ndarray,
    ) -> np.ndarray:
        return np.rot90(im, self.rng.choice(4))

    def flip(
        self,
        im: np.ndarray,
    ) -> np.ndarray:
        return self.rng.choice(self.flips)(im)

    def crop(
        self,
        im: np.ndarray,
    ) -> np.ndarray:
        y = np.random.randint(0, im.shape[0] - self.patch_size)
        x = np.random.randint(0, im.shape[1] - self.patch_size)
        return im[y:y + self.patch_size, x:x + self.patch_size]

    def __iter__(self, ) -> th.Tensor:
        while True:
            ims = random.choices(self.images, k=self.batch_size)
            arr = np.empty(
                (self.batch_size, self.ch, self.patch_size, self.patch_size)
            )
            for i, im in enumerate(ims):
                arr[i] = self.transform(im).transpose(2, 0, 1)
            ims = th.from_numpy(arr).cuda().float()
            yield ims
