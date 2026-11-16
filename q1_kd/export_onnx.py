"""Export the trained student checkpoints (baseline + distilled) to ONNX and verify against PyTorch.

    python q1_kd/export_onnx.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from common.utils import CKPT, ONNX, ensure_dirs
from q1_kd.models import STUDENTS, load_checkpoint


def export(ckpt_path, onnx_path, opset=17):
    model, ck = load_checkpoint(ckpt_path, "cpu")
    dummy = torch.randn(1, 3, 32, 32)
    torch.onnx.export(model, dummy, onnx_path, input_names=["input"], output_names=["logits"], opset_version=opset,
                      dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}}, dynamo=False)
    # numerical check against PyTorch
    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    x = torch.randn(4, 3, 32, 32)
    ref = model(x).detach().numpy()
    out = sess.run(None, {"input": x.numpy()})[0]
    err = float(np.abs(ref - out).max())
    print(f"exported {os.path.basename(onnx_path):38s} {os.path.getsize(onnx_path) / 1e6:6.2f} MB  max|torch-onnx| = {err:.2e}")
    assert err < 1e-3, "ONNX output differs from PyTorch"


if __name__ == "__main__":
    ensure_dirs()
    for mode in ("baseline", "kd"):
        for arch in STUDENTS:
            p = os.path.join(CKPT, f"{mode}_{arch}.pth")
            if os.path.exists(p):
                export(p, os.path.join(ONNX, f"{mode}_{arch}.onnx"))
