"""Unified inference pipeline for ALL models in this project (Q1 students: baseline / distilled, and the Q2 NAS model).

Backends:  --backend torch   (PyTorch checkpoint, .pth)
           --backend onnx    (ONNX Runtime, CPU)           [default]

Examples
    python inference.py --model kd_mobilenetv2 --image path/to/img.png
    python inference.py --model baseline_squeezenet --backend torch --test-samples 10
    python inference.py --model nas_best --test-samples 1000          # accuracy on CIFAR-10 test samples
    python inference.py --list
"""
import argparse, os, sys, time
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import numpy as np
from PIL import Image

CKPT, ONNX = os.path.join(ROOT, "checkpoints"), os.path.join(ROOT, "onnx_models")
CIFAR10 = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
CIFAR100 = ["apple", "aquarium_fish", "baby", "bear", "beaver", "bed", "bee", "beetle", "bicycle", "bottle", "bowl", "boy", "bridge", "bus", "butterfly", "camel", "can", "castle", "caterpillar", "cattle", "chair", "chimpanzee", "clock", "cloud", "cockroach", "couch", "crab", "crocodile", "cup", "dinosaur", "dolphin", "flatfish", "forest", "fox", "girl", "hamster", "house", "kangaroo", "keyboard", "lamp", "lawn_mower", "leopard", "lion", "lizard", "lobster", "man", "maple_tree", "motorcycle", "mountain", "mouse", "mushroom", "oak_tree", "orange", "orchid", "otter", "palm_tree", "pear", "pickup_truck", "pine_tree", "plain", "plate", "poppy", "porcupine", "possum", "rabbit", "raccoon", "ray", "road", "rocket", "rose", "sea", "seal", "shark", "shrew", "skunk", "skyscraper", "snail", "snake", "spider", "squirrel", "streetcar", "sunflower", "tank", "telephone", "television", "tiger", "tractor", "train", "trout", "tulip", "turtle", "vase", "whale", "willow_tree", "wolf", "worm"]
STATS = {"cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
         "cifar100": ((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))}


def available():
    names = sorted(f[:-5] for f in os.listdir(ONNX) if f.endswith(".onnx")) if os.path.isdir(ONNX) else []
    return names


def dataset_of(model_name):
    return "cifar10" if model_name.startswith("nas") else "cifar100"


def preprocess(img, dataset):
    """PIL image / ndarray (HxWx3 uint8) -> normalised float32 1x3x32x32."""
    if not isinstance(img, Image.Image):
        img = Image.fromarray(img)
    img = img.convert("RGB").resize((32, 32), Image.BILINEAR)
    x = np.asarray(img, dtype=np.float32) / 255.0
    mean, std = (np.array(s, dtype=np.float32) for s in STATS[dataset])
    return ((x - mean) / std).transpose(2, 0, 1)[None]


class Predictor:
    """Wraps a trained model behind one `predict(x)->logits` interface, whatever its architecture/backend."""

    def __init__(self, model_name, backend="onnx"):
        self.name, self.backend = model_name, backend
        self.dataset = dataset_of(model_name)
        self.classes = CIFAR10 if self.dataset == "cifar10" else CIFAR100
        if backend == "onnx":
            import onnxruntime as ort
            so = ort.SessionOptions()
            self.sess = ort.InferenceSession(os.path.join(ONNX, model_name + ".onnx"), so, providers=["CPUExecutionProvider"])
            self.in_name = self.sess.get_inputs()[0].name
        else:
            import torch
            self.torch = torch
            if model_name.startswith("nas"):
                from q2_nas.generator import build_model
                ck = torch.load(os.path.join(CKPT, model_name + ".pth"), map_location="cpu", weights_only=False)
                self.model = build_model(ck["config"], ck["num_classes"]); self.model.load_state_dict(ck["state_dict"])
            else:
                from q1_kd.models import load_checkpoint
                self.model, _ = load_checkpoint(os.path.join(CKPT, model_name + ".pth"))
            self.model.eval()

    def predict(self, x):
        if self.backend == "onnx":
            return self.sess.run(None, {self.in_name: x.astype(np.float32)})[0]
        with self.torch.no_grad():
            return self.model(self.torch.from_numpy(x)).numpy()

    def classify(self, img, topk=5):
        t = time.perf_counter()
        logits = self.predict(preprocess(img, self.dataset))[0]
        ms = (time.perf_counter() - t) * 1000
        p = np.exp(logits - logits.max()); p /= p.sum()
        top = np.argsort(-p)[:topk]
        return [(self.classes[i], float(p[i])) for i in top], ms


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", help="e.g. baseline_mobilenetv2, kd_squeezenet, kd_efficientnet_lite0, nas_best")
    ap.add_argument("--backend", choices=["onnx", "torch"], default="onnx")
    ap.add_argument("--image", help="path to an image file")
    ap.add_argument("--test-samples", type=int, default=0, help="run on N samples of the matching CIFAR test set")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if a.list or not a.model:
        print("available models:", ", ".join(available())); return
    pred = Predictor(a.model, a.backend)
    if a.image:
        top, ms = pred.classify(Image.open(a.image))
        print(f"{a.model} [{a.backend}]  {ms:.2f} ms")
        for c, p in top:
            print(f"  {c:20s} {p * 100:6.2f}%")
    if a.test_samples:
        from common.prepare_data import load_numpy
        xs, ys = load_numpy(pred.dataset, False)
        ds = [(Image.fromarray(xs[i]), int(ys[i])) for i in range(min(a.test_samples, len(xs)))]
        n = min(a.test_samples, len(ds)); correct, t0 = 0, time.perf_counter()
        for i in range(n):
            img, y = ds[i]
            top, _ = pred.classify(img, 1)
            correct += pred.classes.index(top[0][0]) == y
            if n <= 10:
                print(f"  sample {i}: predicted {top[0][0]:15s} true {pred.classes[y]:15s}")
        dt = (time.perf_counter() - t0) / n * 1000
        print(f"{a.model} [{a.backend}] accuracy on {n} test samples: {100 * correct / n:.2f}%   ({dt:.2f} ms/img incl. preprocessing)")


if __name__ == "__main__":
    main()
