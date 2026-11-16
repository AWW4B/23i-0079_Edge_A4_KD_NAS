"""Download CIFAR-10 / CIFAR-100 (HuggingFace parquet mirrors: uoft-cs/cifar10, uoft-cs/cifar100) and store them as
compact uint8 .npz files in data/.  Used instead of torchvision's downloader, whose host (toronto.edu) is often slow
or unreachable.

    python common/prepare_data.py            # both datasets
"""
import io, os, sys, urllib.request
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
BASE = "https://huggingface.co/datasets/uoft-cs/{n}/resolve/main/{sub}/{split}-00000-of-00001.parquet"
SUB = {"cifar10": "plain_text", "cifar100": "cifar100"}
LABEL = {"cifar10": "label", "cifar100": "fine_label"}


def _download(url, dest, tries=8):
    for t in range(tries):
        try:
            have = os.path.getsize(dest) if os.path.exists(dest) else 0
            req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
            with urllib.request.urlopen(req, timeout=60) as r, open(dest, "ab" if have else "wb") as f:
                while True:
                    b = r.read(1 << 20)
                    if not b:
                        return
                    f.write(b)
        except urllib.error.HTTPError as e:
            if e.code == 416:      # already complete
                return
            print("retry", e)
        except Exception as e:
            print("retry", repr(e))
    raise RuntimeError("download failed: " + url)


def prepare(name):
    import pandas as pd
    from PIL import Image
    os.makedirs(DATA, exist_ok=True)
    for split in ("train", "test"):
        out = os.path.join(DATA, f"{name}_{split}.npz")
        if os.path.exists(out):
            continue
        pq = os.path.join(DATA, f"{name}_{split}.parquet")
        print(f"downloading {name} {split} ...", flush=True)
        _download(BASE.format(n=name, sub=SUB[name], split=split), pq)
        df = pd.read_parquet(pq)
        imgcol = "img" if "img" in df.columns else "image"
        x = np.stack([np.asarray(Image.open(io.BytesIO(d["bytes"])).convert("RGB")) for d in df[imgcol]]).astype(np.uint8)
        y = df[LABEL[name]].to_numpy().astype(np.int64)
        np.savez_compressed(out, x=x, y=y)
        os.remove(pq)
        print(f"  saved {out}  x{x.shape} y{y.shape}", flush=True)


def load_numpy(name, train):
    """Return (x uint8 N,32,32,3 ; y int64 N), preparing the data on first use."""
    f = os.path.join(DATA, f"{name}_{'train' if train else 'test'}.npz")
    if not os.path.exists(f):
        prepare(name)
    d = np.load(f)
    return d["x"], d["y"]


if __name__ == "__main__":
    for n in (sys.argv[1:] or ["cifar100", "cifar10"]):
        prepare(n)
