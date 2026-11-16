"""Teacher (ResNet-50) and the three lightweight student architectures (CIFAR-adapted).

All networks share the same interface:

    logits                        = model(x)
    logits, [feat_mid, feat_last] = model(x, return_feats=True)

`feat_mid`  : intermediate feature map at 8x8 spatial resolution (for hint-style KD)
`feat_last` : last convolutional feature map at 4x4 spatial resolution (for feature KD)

The teacher exposes layer3 (1024 x 4 x 4) and layer4 (2048 x 2 x 2); student maps are average-pooled to match.
Students are implemented from scratch (not imported from torchvision) so that they can be
adapted to 32x32 inputs (the ImageNet stems downsample 32x too aggressively for CIFAR).
"""
import torch
import torch.nn as nn
import torchvision

NUM_CLASSES = 100
TEACHER_CH = (1024, 2048)  # channels of teacher layer3 / layer4


# --------------------------------------------------------------------------- teacher
class ResNet50Teacher(nn.Module):
    """torchvision ResNet-50 with a CIFAR stem: two 3x3 convs (the second with stride 2, no max-pool), so
    layer1..layer4 run at 16x16, 8x8, 4x4, 2x2. (A stride-1 stem is ~6x slower on a laptop GPU.)"""

    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        m = torchvision.models.resnet50(weights=None, num_classes=num_classes)
        m.conv1 = nn.Sequential(nn.Conv2d(3, 32, 3, 1, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
                                nn.Conv2d(32, 64, 3, 2, 1, bias=False))
        m.maxpool = nn.Identity()
        self.m = m

    def forward(self, x, return_feats=False):
        m = self.m
        x = m.relu(m.bn1(m.conv1(x)))
        x = m.layer2(m.layer1(x))
        f3 = m.layer3(x)
        f4 = m.layer4(f3)
        logits = m.fc(torch.flatten(m.avgpool(f4), 1))
        return (logits, [f3, f4]) if return_feats else logits


# --------------------------------------------------------------------------- helpers
def conv_bn(cin, cout, k, s=1, groups=1, act=True):
    layers = [nn.Conv2d(cin, cout, k, s, k // 2, groups=groups, bias=False), nn.BatchNorm2d(cout)]
    if act:
        layers.append(nn.ReLU6(inplace=True))
    return nn.Sequential(*layers)


class InvertedResidual(nn.Module):
    """MBConv block (no squeeze-excite, which is what makes EfficientNet 'Lite')."""

    def __init__(self, cin, cout, stride, expand, k=3):
        super().__init__()
        mid = cin * expand
        layers = []
        if expand != 1:
            layers.append(conv_bn(cin, mid, 1))
        layers += [conv_bn(mid, mid, k, stride, groups=mid), conv_bn(mid, cout, 1, act=False)]
        self.block = nn.Sequential(*layers)
        self.use_res = stride == 1 and cin == cout

    def forward(self, x):
        return x + self.block(x) if self.use_res else self.block(x)


class Student(nn.Module):
    """Base class: features_a -> (mid feature) -> features_b -> (last feature) -> head."""

    def forward(self, x, return_feats=False):
        mid = self.features_a(x)
        last = self.features_b(mid)
        logits = self.head(last)
        return (logits, [mid, last]) if return_feats else logits

    def feature_channels(self):
        return self._fch


def _head(cin, num_classes, p=0.2):
    return nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(p), nn.Linear(cin, num_classes))


# --------------------------------------------------------------------------- MobileNetV2
class MobileNetV2(Student):
    # (expand t, out c, repeats n, stride s) -- strides adapted for 32x32 inputs
    CFG = [(1, 16, 1, 1), (6, 24, 2, 2), (6, 32, 3, 2), (6, 64, 4, 1), (6, 96, 3, 1), (6, 160, 3, 2), (6, 320, 1, 1)]

    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        layers, cin, ends = [conv_bn(3, 32, 3, 1)], 32, []
        for t, c, n, s in self.CFG:
            for i in range(n):
                layers.append(InvertedResidual(cin, c, s if i == 0 else 1, t))
                cin = c
            ends.append(len(layers))
        cut = ends[4]  # after the 96-channel stage (8x8)
        self.features_a = nn.Sequential(*layers[:cut])
        self.features_b = nn.Sequential(*layers[cut:], conv_bn(320, 1280, 1))
        self.head = _head(1280, num_classes)
        self._fch = (96, 1280)


# --------------------------------------------------------------------------- SqueezeNet
class Fire(nn.Module):
    def __init__(self, cin, squeeze, e1, e3):
        super().__init__()
        self.squeeze = nn.Sequential(nn.Conv2d(cin, squeeze, 1, bias=False), nn.BatchNorm2d(squeeze), nn.ReLU(inplace=True))
        self.e1 = nn.Sequential(nn.Conv2d(squeeze, e1, 1, bias=False), nn.BatchNorm2d(e1), nn.ReLU(inplace=True))
        self.e3 = nn.Sequential(nn.Conv2d(squeeze, e3, 3, padding=1, bias=False), nn.BatchNorm2d(e3), nn.ReLU(inplace=True))

    def forward(self, x):
        s = self.squeeze(x)
        return torch.cat([self.e1(s), self.e3(s)], 1)


class SqueezeNet(Student):
    """SqueezeNet v1.1 layout (+BatchNorm for from-scratch training, 3x3 stride-1 stem for CIFAR)."""

    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.features_a = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                   # 16x16
            Fire(64, 16, 64, 64), Fire(128, 16, 64, 64),
            nn.MaxPool2d(2),                                   # 8x8
            Fire(128, 32, 128, 128), Fire(256, 32, 128, 128))  # -> 256 x 8 x 8
        self.features_b = nn.Sequential(
            nn.MaxPool2d(2),                                   # 4x4
            Fire(256, 48, 192, 192), Fire(384, 48, 192, 192),
            Fire(384, 64, 256, 256), Fire(512, 64, 256, 256))  # -> 512 x 4 x 4
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Conv2d(512, num_classes, 1), nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self._fch = (256, 512)


# --------------------------------------------------------------------------- EfficientNet-Lite0
class EfficientNetLite0(Student):
    """EfficientNet-B0 topology with the 'Lite' modifications: no squeeze-excite, ReLU6 instead of
    swish, and the stem/head are not scaled. Strides adapted for 32x32 inputs."""

    # (expand, kernel, stride, out, repeats)
    CFG = [(1, 3, 1, 16, 1), (6, 3, 2, 24, 2), (6, 5, 2, 40, 2), (6, 3, 1, 80, 3), (6, 5, 1, 112, 3), (6, 5, 2, 192, 4), (6, 3, 1, 320, 1)]

    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        layers, cin, ends = [conv_bn(3, 32, 3, 1)], 32, []
        for t, k, s, c, n in self.CFG:
            for i in range(n):
                layers.append(InvertedResidual(cin, c, s if i == 0 else 1, t, k))
                cin = c
            ends.append(len(layers))
        cut = ends[4]  # after the 112-channel stage (8x8)
        self.features_a = nn.Sequential(*layers[:cut])
        self.features_b = nn.Sequential(*layers[cut:], conv_bn(320, 1280, 1))
        self.head = _head(1280, num_classes)
        self._fch = (112, 1280)


STUDENTS = {"mobilenetv2": MobileNetV2, "squeezenet": SqueezeNet, "efficientnet_lite0": EfficientNetLite0}


def build_student(name, num_classes=NUM_CLASSES):
    return STUDENTS[name](num_classes)


def load_checkpoint(path, device="cpu"):
    """Load any checkpoint produced by this project and return (model, metadata)."""
    ck = torch.load(path, map_location=device, weights_only=False)
    arch = ck["arch"]
    nc = ck.get("num_classes", NUM_CLASSES)
    model = ResNet50Teacher(nc) if arch == "resnet50" else build_student(arch, nc)
    model.load_state_dict(ck["state_dict"])
    return model.to(device).eval(), ck
