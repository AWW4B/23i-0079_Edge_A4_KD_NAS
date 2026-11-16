"""Automated architecture generator: turns a sampled configuration dict into an `nn.Module`."""
import torch
import torch.nn as nn
import torch.nn.functional as F


def make_act(name):
    return {"relu": nn.ReLU, "relu6": nn.ReLU6, "leaky_relu": lambda: nn.LeakyReLU(0.1), "silu": nn.SiLU,
            "hardswish": nn.Hardswish}[name]()


class ConvBlock(nn.Module):
    """Standard KxK convolution + BN + activation."""

    def __init__(self, cin, cout, k, stride, act):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(cin, cout, k, stride, k // 2, bias=False), nn.BatchNorm2d(cout), make_act(act))

    def forward(self, x):
        return self.body(x)


class DSConvBlock(nn.Module):
    """Depthwise-separable convolution: depthwise KxK + BN + act, then pointwise 1x1 + BN + act."""

    def __init__(self, cin, cout, k, stride, act):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(cin, cin, k, stride, k // 2, groups=cin, bias=False), nn.BatchNorm2d(cin), make_act(act),
            nn.Conv2d(cin, cout, 1, bias=False), nn.BatchNorm2d(cout), make_act(act))

    def forward(self, x):
        return self.body(x)


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return x + self.fn(x)


class GeneratedNet(nn.Module):
    """Network built from an architecture configuration. Input is always a normalised 3x32x32 image; the net
    resizes it internally to `resolution`, so every NAS model shares one deployment interface."""

    def __init__(self, cfg, num_classes=10):
        super().__init__()
        self.cfg = cfg
        res = cfg["resolution"]
        layers = []
        if res != 32:
            layers.append(nn.Upsample(size=(res, res), mode="bilinear", align_corners=False))
        stem = cfg["stem_channels"]
        first_act = cfg["stages"][0]["act"]
        layers.append(ConvBlock(3, stem, 3, 1, first_act))
        cin, size = stem, res
        for st in cfg["stages"]:
            Block = DSConvBlock if st["block"] == "dsconv" else ConvBlock
            stride = st["stride"] if size // st["stride"] >= 2 else 1   # never shrink below 2x2
            size //= stride
            for i in range(st["depth"]):
                s = stride if i == 0 else 1
                blk = Block(cin, st["channels"], st["kernel"], s, st["act"])
                layers.append(Residual(blk) if (s == 1 and cin == st["channels"]) else blk)
                cin = st["channels"]
        layers.append(ConvBlock(cin, cfg["head_channels"], 1, 1, cfg["stages"][-1]["act"]))
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(cfg["head_channels"], num_classes))

    def forward(self, x):
        return self.classifier(self.features(x))


def build_model(cfg, num_classes=10):
    return GeneratedNet(cfg, num_classes)
