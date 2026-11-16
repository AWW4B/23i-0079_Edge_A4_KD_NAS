"""NAS search space: definition, random sampling, mutation and crossover of architecture configurations.

An architecture configuration ("genome") is a plain JSON-serialisable dict:

    {
      "resolution": 24,                 # input resolution the network internally resizes to (16/24/32)
      "stem_channels": 16,
      "stages": [                       # 2..4 stages (variable depth)
         {"depth": 2,                   # number of blocks in the stage (variable convolutional depth)
          "channels": 32,               # number of filters (variable width)
          "kernel": 5,                  # kernel size 3/5/7
          "block": "dsconv",            # "conv" (standard) or "dsconv" (depthwise-separable)
          "act": "hardswish",           # activation function
          "stride": 2},                 # spatial down-sampling at the start of the stage
         ...],
      "head_channels": 128              # 1x1 conv before global pooling
    }
"""
import copy
import random

RESOLUTIONS = [16, 24, 32]
STEM_CHANNELS = [8, 16, 24]
NUM_STAGES = [2, 3, 4]
DEPTHS = [1, 2, 3]
CHANNELS = [8, 16, 24, 32, 48, 64, 96, 128]
KERNELS = [3, 5, 7]
BLOCKS = ["conv", "dsconv"]
ACTIVATIONS = ["relu", "relu6", "leaky_relu", "silu", "hardswish"]
STRIDES = [1, 2]
HEAD_CHANNELS = [64, 128, 256]
MAX_STAGES = max(NUM_STAGES)

SPACE = {
    "resolution": RESOLUTIONS, "stem_channels": STEM_CHANNELS, "num_stages": NUM_STAGES,
    "depth": DEPTHS, "channels": CHANNELS, "kernel": KERNELS, "block": BLOCKS, "act": ACTIVATIONS,
    "stride": STRIDES, "head_channels": HEAD_CHANNELS,
}


def space_size():
    per_stage = len(DEPTHS) * len(CHANNELS) * len(KERNELS) * len(BLOCKS) * len(ACTIVATIONS) * len(STRIDES)
    total = 0
    for n in NUM_STAGES:
        total += per_stage ** n
    return total * len(RESOLUTIONS) * len(STEM_CHANNELS) * len(HEAD_CHANNELS)


def random_stage(rng):
    return {"depth": rng.choice(DEPTHS), "channels": rng.choice(CHANNELS), "kernel": rng.choice(KERNELS),
            "block": rng.choice(BLOCKS), "act": rng.choice(ACTIVATIONS), "stride": rng.choice(STRIDES)}


def sample_architecture(rng=random):
    n = rng.choice(NUM_STAGES)
    return {"resolution": rng.choice(RESOLUTIONS), "stem_channels": rng.choice(STEM_CHANNELS),
            "stages": [random_stage(rng) for _ in range(n)], "head_channels": rng.choice(HEAD_CHANNELS)}


def mutate(arch, rng=random, p=0.25):
    """Randomly perturb genes (each with probability p); at least one gene always changes."""
    a = copy.deepcopy(arch)
    changed = False

    def maybe(key, options, holder):
        nonlocal changed
        if rng.random() < p:
            new = rng.choice(options)
            changed |= new != holder[key]
            holder[key] = new

    maybe("resolution", RESOLUTIONS, a)
    maybe("stem_channels", STEM_CHANNELS, a)
    maybe("head_channels", HEAD_CHANNELS, a)
    for st in a["stages"]:
        for key, opts in (("depth", DEPTHS), ("channels", CHANNELS), ("kernel", KERNELS), ("block", BLOCKS),
                          ("act", ACTIVATIONS), ("stride", STRIDES)):
            maybe(key, opts, st)
    if rng.random() < p:  # structural mutation: add / remove a stage
        if len(a["stages"]) < MAX_STAGES and (len(a["stages"]) == min(NUM_STAGES) or rng.random() < 0.5):
            a["stages"].append(random_stage(rng)); changed = True
        elif len(a["stages"]) > min(NUM_STAGES):
            a["stages"].pop(rng.randrange(len(a["stages"]))); changed = True
    if not changed:  # force one change
        st = rng.choice(a["stages"])
        key = rng.choice(["depth", "channels", "kernel", "block", "act", "stride"])
        opts = {"depth": DEPTHS, "channels": CHANNELS, "kernel": KERNELS, "block": BLOCKS, "act": ACTIVATIONS, "stride": STRIDES}[key]
        st[key] = rng.choice([o for o in opts if o != st[key]])
    return a


def crossover(a, b, rng=random):
    """Uniform crossover on the global genes, stage-wise crossover on the stage list."""
    c = {"resolution": rng.choice([a["resolution"], b["resolution"]]),
         "stem_channels": rng.choice([a["stem_channels"], b["stem_channels"]]),
         "head_channels": rng.choice([a["head_channels"], b["head_channels"]])}
    n = rng.choice([len(a["stages"]), len(b["stages"])])
    stages = []
    for i in range(n):
        pool = [x["stages"][i] for x in (a, b) if i < len(x["stages"])]
        stages.append(copy.deepcopy(rng.choice(pool)))
    c["stages"] = stages
    return c


def arch_key(arch):
    """Canonical string used to avoid evaluating the same architecture twice."""
    import json
    return json.dumps(arch, sort_keys=True)
