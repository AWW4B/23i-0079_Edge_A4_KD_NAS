# Edge AI Assignment 4 - Knowledge Distillation & Hardware-Aware NAS

Roll No: **23i-0079**, Section **A**

* **Q1** - ResNet-50 teacher -> MobileNetV2 / SqueezeNet / EfficientNet-Lite0 students on **CIFAR-100**
  (soft-target KD with temperature, hard+soft loss, feature-based KD), baselines without KD, unified inference, ONNX export + ONNX Runtime CPU inference.
* **Q2** - Hardware-aware **evolutionary NAS** on **CIFAR-10** (measured CPU latency, model size, peak memory, constraint rejection),
  best architecture trained from scratch, exported to ONNX and run with ONNX Runtime.

The technical report is `23i-0079_A.pdf` (repo root; copy in `report/`).

## Repository layout

```
common/            data loading (GPU-resident CIFAR), dataset preparation, utilities
q1_kd/             models.py (teacher+3 students)  kd.py (KD losses)  trainer.py
                   train_teacher.py  train_baseline.py  train_kd.py  export_onnx.py
q2_nas/            search_space.py  generator.py  hardware.py  measure_cpu.py
                   nas_search.py (evolutionary search)  train_best.py
inference.py       unified inference pipeline (all students + NAS model; PyTorch or ONNX Runtime)
benchmarks/        benchmark.py  -> results/benchmark_results.csv
checkpoints/       *.pth trained models (teacher, baseline_*, kd_*, kdlogit_squeezenet, nas_best)
onnx_models/       exported *.onnx models
results/           training histories (json), NAS history (csv/json), best architecture, benchmark csv
report/            report generator + figures
logs/              raw training logs
run_all.py         runs the whole pipeline
```

## Installation

Python 3.10+ (developed on Python 3.11, Windows 11, RTX 4050 Laptop GPU 6 GB).

```bash
# 1. PyTorch (GPU build shown; use the CPU build from pytorch.org if you have no NVIDIA GPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
# 2. everything else
pip install -r requirements.txt
```

CIFAR-10/100 are downloaded automatically on first use (HuggingFace mirror, stored as `data/*.npz`), or manually with
`python common/prepare_data.py`.

> Training needs an NVIDIA GPU (CUDA + mixed precision). Inference / ONNX / benchmarking run on CPU only.
> The trained checkpoints and ONNX models are included, so inference works without retraining.

## Quick start (use the included trained models)

```bash
python inference.py --list
python inference.py --model kd_mobilenetv2 --image some_image.png          # ONNX Runtime, CPU
python inference.py --model baseline_squeezenet --test-samples 1000        # accuracy on 1000 CIFAR-100 test images
python inference.py --model nas_best --test-samples 1000                   # NAS model, CIFAR-10
python inference.py --model kd_efficientnet_lite0 --backend torch --test-samples 10
python benchmarks/benchmark.py                                             # regenerates results/benchmark_results.csv
```

Model names: `{baseline,kd}_{mobilenetv2,squeezenet,efficientnet_lite0}` and `nas_best`.

## Reproduce everything step by step

```bash
python run_all.py                    # complete pipeline (about 2.5 h on an RTX 4050 laptop GPU); logs in logs/
```

or run the stages individually:

```bash
# ---- Q1 ----
python q1_kd/train_teacher.py  --epochs 20                       # ResNet-50 teacher  -> checkpoints/teacher_resnet50.pth
python q1_kd/train_kd.py       --arch mobilenetv2 --epochs 15    # distilled students -> checkpoints/kd_*.pth
python q1_kd/train_kd.py       --arch efficientnet_lite0 --epochs 15
python q1_kd/train_kd.py       --arch squeezenet --epochs 15 --beta 0.1   # light feature weight for the tiny SqueezeNet
python q1_kd/train_baseline.py --arch all --epochs 15            # baselines (no KD) -> checkpoints/baseline_*.pth
python q1_kd/export_onnx.py                                      # -> onnx_models/*.onnx (verified vs PyTorch)
# ablation: logit-only KD without the feature loss (-> checkpoints/kdlogit_squeezenet.pth)
python q1_kd/train_kd.py --arch squeezenet --epochs 15 --no-feat

# ---- Q2 ----
python q2_nas/nas_search.py --population 10 --generations 5 --children 5 --epochs 3 --train-subset 15000
                                                                 # -> results/nas_history.csv|json, results/best_architecture.json
python q2_nas/train_best.py --epochs 30                          # train from scratch -> checkpoints/nas_best.pth + onnx_models/nas_best.onnx

# ---- benchmarking, figures, report ----
python benchmarks/benchmark.py
python report/make_figures.py
python report/make_report.py                                     # -> 23i-0079_A.pdf
```

`python run_all.py --quick` runs everything with 1-epoch settings (a few minutes) to verify that the code works.

## Results (CIFAR-100 test accuracy %, 15-epoch student budget; teacher 20 epochs)

| Model | Params | Baseline | Distilled | Gain | ONNX size | CPU latency (1 thread, batch 1) |
|---|---|---|---|---|---|---|
| ResNet-50 (teacher) | 23.7 M | 67.16 | - | - | - | - |
| MobileNetV2 | 2.34 M | 64.52 | **66.17** | +1.65 | 9.4 MB | 1.3 ms |
| SqueezeNet | 0.77 M | 57.28 | **60.74** | +3.46 | 3.1 MB | 0.38 ms |
| EfficientNet-Lite0 | 3.48 M | 65.12 | **67.40** | +2.28 | 14.0 MB | 1.8 ms |

SqueezeNet ablation: feature weight beta=1.0 gives 53.23 (hurts), logit-only KD 61.35, beta=0.1 60.74 (reported).

**NAS (CIFAR-10):** 39 candidates evaluated (7 rejected by the hardware constraints). The best architecture, trained from scratch for 30 epochs,
reaches **85.35 %** test accuracy with 0.053 M parameters, 0.22 MB ONNX size and 0.38 ms CPU latency.
Full numbers: `results/benchmark_results.csv`. All training budgets are short because everything was trained on one laptop GPU (see the report).

## Key hyper-parameters

| | value |
|---|---|
| KD loss | `(1-a)*CE + a*T^2*KL(softmax(t/T) || softmax(s/T)) + beta*FeatLoss`, T = 4, a = 0.7, beta = 1.0 (0.1 for SqueezeNet) |
| Feature KD | 1x1-conv+BN regressors on two stages (8x8 and 4x4 student maps -> teacher layer3/layer4), L2-normalised MSE |
| Optimiser | SGD+Nesterov, OneCycle LR (max 0.1), wd 5e-4, batch 128, AMP, random-crop+flip |
| NAS constraints | CPU latency <= 3 ms (1 thread, batch 1, ONNX Runtime), ONNX size <= 1 MB, peak inference memory <= 20 MB, <= 0.5 M params |

## Deliverable checklist

| # | Deliverable | Where |
|---|---|---|
| 1 | Teacher model | `checkpoints/teacher_resnet50.pth`, `q1_kd/train_teacher.py` |
| 2 | Student architectures | `q1_kd/models.py` |
| 3 | Baseline students | `checkpoints/baseline_*.pth`, `q1_kd/train_baseline.py` |
| 4 | Knowledge distillation | `q1_kd/kd.py`, `q1_kd/train_kd.py`, `checkpoints/kd_*.pth` |
| 5 | NAS search space | `q2_nas/search_space.py`, `q2_nas/generator.py` |
| 6 | NAS algorithm | `q2_nas/nas_search.py` (evolutionary) |
| 7 | NAS results | `results/best_architecture.json`, `results/nas_history.csv/json` |
| 8 | Optimised architecture | `checkpoints/nas_best.pth` |
| 9 | Inference pipeline | `inference.py` |
| 10 | ONNX deployment | `onnx_models/*.onnx`, `inference.py --backend onnx` |
| 11 | Benchmarking | `benchmarks/benchmark.py`, `results/benchmark_results.csv` |
| 12 | Technical report | `23i-0079_A.pdf` |
| 13 | README | this file |
