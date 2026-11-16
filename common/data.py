"""CIFAR-10 / CIFAR-100 loading with fully GPU-resident data and GPU-side augmentation.

Keeping the whole dataset on the GPU (50k x 3 x 32 x 32 uint8 = 150 MB) removes the
data-loader bottleneck on Windows and makes epochs much faster.
"""
import os
import numpy as np
import torch
import torch.nn.functional as F
from common.prepare_data import load_numpy

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

STATS = {
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616), 10),
    "cifar100": ((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762), 100),
}


def _load_raw(name, train):
    x, y = load_numpy(name, train)
    return torch.from_numpy(x).permute(0, 3, 1, 2).contiguous(), torch.from_numpy(y)


class GPUData:
    """Holds a dataset on the device and yields normalised (optionally augmented) batches."""

    def __init__(self, name="cifar100", train=True, device="cuda", subset=None, seed=0, resolution=32):
        mean, std, self.num_classes = STATS[name]
        self.x, self.y = _load_raw(name, train)
        if subset is not None and subset < len(self.x):
            g = torch.Generator().manual_seed(seed)
            idx = torch.randperm(len(self.x), generator=g)[:subset]
            self.x, self.y = self.x[idx], self.y[idx]
        self.x, self.y = self.x.to(device), self.y.to(device)
        self.mean = torch.tensor(mean, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(std, device=device).view(1, 3, 1, 1)
        self.device, self.train, self.resolution = device, train, resolution

    def view(self, idx, train):
        """Shallow copy restricted to the sample indices `idx` (used for train/val splits)."""
        import copy
        v = copy.copy(self)
        v.x, v.y, v.train = self.x[idx], self.y[idx], train
        return v

    def __len__(self):
        return len(self.x)

    def normalize(self, xb):
        xb = xb.float().div_(255.0)
        return (xb - self.mean) / self.std

    def augment(self, xb):
        """Random crop (pad 4) + horizontal flip, done on the GPU."""
        xb = xb.float()
        n = xb.shape[0]
        flip = torch.rand(n, device=xb.device) < 0.5
        xb = torch.where(flip.view(-1, 1, 1, 1), xb.flip(3), xb)
        xp = F.pad(xb, (4, 4, 4, 4), mode="reflect")
        i = torch.randint(0, 9, (n,), device=xb.device)
        j = torch.randint(0, 9, (n,), device=xb.device)
        ar = torch.arange(32, device=xb.device)
        rows = (i.view(-1, 1) + ar.view(1, -1))  # n,32
        cols = (j.view(-1, 1) + ar.view(1, -1))
        xp = xp[torch.arange(n, device=xb.device).view(-1, 1, 1, 1), torch.arange(3, device=xb.device).view(1, 3, 1, 1),
                rows.view(n, 1, 32, 1), cols.view(n, 1, 1, 32)]
        return xp

    def batches(self, batch_size, shuffle=None):
        shuffle = self.train if shuffle is None else shuffle
        n = len(self.x)
        idx = torch.randperm(n, device=self.device) if shuffle else torch.arange(n, device=self.device)
        for s in range(0, n, batch_size):
            b = idx[s:s + batch_size]
            if self.train and shuffle:
                xb = self.normalize(self.augment(self.x[b]))
            else:
                xb = self.normalize(self.x[b])
            if self.resolution != 32:
                xb = F.interpolate(xb, size=(self.resolution, self.resolution), mode="bilinear", align_corners=False)
            yield xb, self.y[b]
