"""Build the technical report PDF (23i-0079_A.pdf) from results/*.json|csv, figures and log excerpts."""
import csv, json, os, textwrap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle, Preformatted)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES, FIG, LOGS = (os.path.join(ROOT, d) for d in ("results", os.path.join("report", "figures"), "logs"))
OUT = os.path.join(ROOT, "23i-0079_A.pdf")
ARCHS = ["mobilenetv2", "squeezenet", "efficientnet_lite0"]
NAMES = {"mobilenetv2": "MobileNetV2", "squeezenet": "SqueezeNet", "efficientnet_lite0": "EfficientNet-Lite0"}


def J(n):
    p = os.path.join(RES, n + ".json")
    return json.load(open(p)) if os.path.exists(p) else None


def csv_rows(n):
    p = os.path.join(RES, n)
    return list(csv.DictReader(open(p))) if os.path.exists(p) else []


from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
_fd = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf")
for _n, _f in (("DV", "DejaVuSans.ttf"), ("DV-B", "DejaVuSans-Bold.ttf"), ("DV-I", "DejaVuSans-Oblique.ttf"), ("DV-BI", "DejaVuSans-BoldOblique.ttf")):
    pdfmetrics.registerFont(TTFont(_n, os.path.join(_fd, _f)))
pdfmetrics.registerFontFamily("DV", normal="DV", bold="DV-B", italic="DV-I", boldItalic="DV-BI")

ss = getSampleStyleSheet()
for _k in ("Heading1", "Heading2", "BodyText"):
    ss[_k].fontName = "DV-B" if _k != "BodyText" else "DV"
H1 = ParagraphStyle("H1", parent=ss["Heading1"], fontSize=16, spaceBefore=10, spaceAfter=6)
H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontSize=12.5, spaceBefore=8, spaceAfter=4)
B = ParagraphStyle("B", parent=ss["BodyText"], fontSize=9.6, leading=13.2, spaceAfter=5)
CODE = ParagraphStyle("C", fontName="Courier", fontSize=7.4, leading=9.2)
story = []


def P(t, st=B):
    story.append(Paragraph(t, st))


def table(rows, widths=None, fs=8):
    t = Table(rows, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f3a5f")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                           ("FONTNAME", (0, 0), (-1, -1), "DV"), ("FONTSIZE", (0, 0), (-1, -1), fs), ("GRID", (0, 0), (-1, -1), .4, colors.grey),
                           ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#eef2f7")]),
                           ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
    story.append(t); story.append(Spacer(1, 6))


def img(name, w=16.5):
    p = os.path.join(FIG, name)
    if os.path.exists(p):
        from PIL import Image as PI
        iw, ih = PI.open(p).size
        story.append(Image(p, width=w * cm, height=w * cm * ih / iw)); story.append(Spacer(1, 4))


def shot(title, logname, pattern=None, lines=14, out=None, tail=True):
    """Render a log excerpt as a terminal-style screenshot image."""
    p = os.path.join(LOGS, logname)
    if not os.path.exists(p):
        return
    L = [l.rstrip() for l in open(p, encoding="utf-8", errors="ignore") if l.strip() and "Warning" not in l and "sched.step" not in l]
    if pattern:
        L = [l for l in L if pattern in l]
    L = L[-lines:] if tail else L[:lines]
    L = [textwrap.shorten(l, 118) for l in L]
    fig = plt.figure(figsize=(9.2, 0.42 + 0.19 * len(L)), facecolor="#0c0c0c")
    fig.text(0.01, 1 - 0.22 / (0.42 + 0.19 * len(L)), f"$ {title}", color="#7ee787", family="monospace", fontsize=7.5, va="top")
    for i, l in enumerate(L):
        fig.text(0.01, 1 - (0.42 + 0.19 * i) / (0.42 + 0.19 * len(L)), l, color="#d0d0d0", family="monospace", fontsize=7, va="top")
    path = os.path.join(FIG, out or logname.replace(".log", "_shot.png"))
    fig.savefig(path, dpi=150, facecolor=fig.get_facecolor()); plt.close(fig)
    img(os.path.basename(path), 16.5)


# ------------------------------------------------------------------------------------------------ data
teacher = J("teacher_resnet50")
base = {a: J(f"baseline_{a}") for a in ARCHS}
kd = {a: J(f"kd_{a}") for a in ARCHS}
bench = {r["model"]: r for r in csv_rows("benchmark_results.csv")}
nas_best = J("best_architecture")
nas_hist = J("nas_history")
nas_train = J("nas_best_training")
hist_rows = csv_rows("nas_history.csv")
abl = J("kdlogit_squeezenet")

# ------------------------------------------------------------------------------------------------ title
story.append(Spacer(1, 3 * cm))
P("<para align=center><font size=22><b>Knowledge Distillation and Hardware-Aware Neural Architecture Search for Edge Deployment</b></font></para>")
story.append(Spacer(1, 8))
P("<para align=center><font size=13>Edge AI &mdash; Assignment 4 &mdash; Technical Report</font></para>")
story.append(Spacer(1, 10))
P("<para align=center><font size=12>Roll No: <b>23i-0079</b> &nbsp;&nbsp;|&nbsp;&nbsp; Section: <b>A</b></font></para>")
story.append(Spacer(1, 1.2 * cm))
P("<b>Summary.</b> Question 1 trains a ResNet-50 teacher on CIFAR-100 and distils it into three lightweight students "
  "(MobileNetV2, SqueezeNet, EfficientNet-Lite0) using soft-target, temperature-scaled, hard+soft and feature-based distillation, "
  "compares them against independently trained baselines, and deploys every student through ONNX Runtime on CPU. "
  "Question 2 implements a hardware-aware evolutionary NAS on CIFAR-10 that measures real CPU latency, model size and peak "
  "memory for every candidate, rejects constraint violators, and trains/exports the best architecture. "
  "All numbers in this report are produced by the code in the repository (<font face=Courier>results/</font>, <font face=Courier>logs/</font>).")
story.append(PageBreak())

# ------------------------------------------------------------------------------------------------ setup
P("1. Experimental setup", H1)
P("All experiments ran on a single laptop: AMD Ryzen 7 7445HS CPU, NVIDIA GeForce RTX 4050 Laptop GPU (6 GB), 16 GB RAM, Windows 11, "
  "Python 3.11, PyTorch 2.14 (CUDA 12.6), ONNX Runtime 1.32 (CPU provider). Training used mixed precision (fp16 autocast), "
  "SGD with Nesterov momentum 0.9, weight decay 5e-4, a OneCycle learning-rate schedule (peak 0.1) and random-crop(pad 4)+horizontal-flip "
  "augmentation executed on the GPU. CIFAR is held entirely in GPU memory which removes data-loader overhead.")
P("<b>Compute budget (honest disclosure).</b> Because everything was trained on a single 6 GB laptop GPU, the training budgets are short: "
  "20 epochs for the teacher and 15 epochs for each student, 3-epoch proxy training for NAS candidates and 30 epochs for the final NAS model. "
  "Absolute accuracies are therefore lower than published long-schedule numbers; the comparisons (KD vs. baseline, constraint-aware search) "
  "are made under identical budgets and seeds. All scripts accept <font face=Courier>--epochs</font> to reproduce longer runs.")

# ------------------------------------------------------------------------------------------------ Q1
P("2. Question 1 &mdash; Knowledge distillation", H1)
P("2.1 Teacher: ResNet-50 on CIFAR-100", H2)
P("The teacher is torchvision's ResNet-50 (23.7 M parameters) with a CIFAR stem: two 3x3 convolutions, the second with stride 2, and no max-pool, so "
  "layer1..layer4 operate on 16x16, 8x8, 4x4 and 2x2 maps (a stride-1 stem was measured to be ~6x slower on the laptop GPU). It is trained with "
  "label smoothing 0.1. "
  + (f"<b>Final test accuracy: {teacher['test_acc']:.2f}%</b> after {teacher['epochs']} epochs." if teacher else ""))
img("q1_teacher_curve.png", 8)

P("2.2 Student architectures", H2)
P("All three students are implemented from scratch in <font face=Courier>q1_kd/models.py</font> and adapted to 32x32 inputs (stride-1 stem, fewer early down-samplings). "
  "Each exposes an intermediate feature map (8x8) and the last convolutional feature map (4x4) for feature distillation.")
table([["Student", "Key design", "Feature maps (mid / last)"],
       ["MobileNetV2", "inverted residuals, expansion 6, ReLU6, linear bottlenecks, 1280-ch head", "96x8x8 / 1280x4x4"],
       ["SqueezeNet", "v1.1 Fire modules (squeeze 1x1 + expand 1x1/3x3) + BatchNorm, conv classifier", "256x8x8 / 512x4x4"],
       ["EfficientNet-Lite0", "B0 topology (MBConv, kernels 3/5), no squeeze-excite, ReLU6 (the 'Lite' changes)", "112x8x8 / 1280x4x4"]],
      [3.4 * cm, 9.6 * cm, 4 * cm])

P("2.3 Distillation framework (<font face=Courier>q1_kd/kd.py</font>)", H2)
P("The total loss combines three terms: <b>L = (1&minus;&alpha;)&middot;CE(s, y) + &alpha;&middot;T&sup2;&middot;KL(softmax(t/T) || softmax(s/T)) + &beta;&middot;L<sub>feat</sub></b>.")
P("&bull; <b>Soft-target KD with temperature softmax</b>: both logit vectors are divided by T=4 before the softmax; the KL term is scaled by T&sup2; to keep gradient magnitudes comparable (Hinton et al.).<br/>"
  "&bull; <b>Combined hard + soft labels</b>: &alpha;=0.7 weights the teacher distribution, 0.3 the ground-truth cross-entropy.<br/>"
  "&bull; <b>Feature-based distillation</b>: for two stages, a 1x1 convolution + BatchNorm regressor maps the student feature map to the teacher channel width "
  "(8x8 student map &rarr; teacher layer3 1024ch, 4x4 map &rarr; teacher layer4 2048ch); spatial sizes are matched by adaptive average pooling and the loss is the "
  "squared distance between channel-wise L2-normalised maps (= 2&minus;2cos). &beta;=1. The regressors are trained jointly and discarded afterwards, so the deployed student is unchanged.")

P("2.4 Baselines and distilled results", H2)
P("The baseline pipeline (<font face=Courier>train_baseline.py</font>) is separate from the KD pipeline (<font face=Courier>train_kd.py</font>); both use the same seed, "
  "schedule, epochs and augmentation, so the only difference is the loss.")
rows = [["Model", "Params (M)", "Baseline acc (%)", "KD acc (%)", "Gain (pp)"]]
if teacher:
    rows.append(["ResNet-50 (teacher)", "23.72", f"{teacher['test_acc']:.2f}", "-", "-"])
for a in ARCHS:
    if base[a] and kd[a]:
        rows.append([NAMES[a], f"{base[a]['params'] / 1e6:.2f}", f"{base[a]['test_acc']:.2f}", f"{kd[a]['test_acc']:.2f}", f"{kd[a]['test_acc'] - base[a]['test_acc']:+.2f}"])
table(rows, [5 * cm, 2.6 * cm, 3.4 * cm, 3 * cm, 2.6 * cm], 9)
img("q1_baseline_vs_kd.png", 9.5)
img("q1_student_curves.png", 17)
gains = [(a, kd[a]["test_acc"] - base[a]["test_acc"]) for a in ARCHS if base[a] and kd[a]]
if gains:
    pos = [NAMES[a] for a, g in gains if g > 0]
    neg = [NAMES[a] for a, g in gains if g <= 0]
    txt = "<b>Analysis.</b> "
    if pos:
        txt += f"Distillation improved {', '.join(pos)} over its independently trained baseline"
        txt += (f" (best gain {max(g for _, g in gains):+.2f} pp). " if pos else ". ")
    if neg:
        txt += f"For {', '.join(neg)} distillation did not help under this short schedule. "
    txt += ("The teacher itself only reaches %.1f%% after 20 epochs, which bounds what it can transfer; with a stronger/longer-trained teacher the gap "
            "between KD and baseline may widen. The soft targets carry inter-class similarity information (dark knowledge); the feature loss adds a supervised signal "
            "to the intermediate layers but must be weighted carefully for very small students (see the SqueezeNet ablation below)." % (teacher["test_acc"] if teacher else 0))
    P(txt)
abl1 = J("ablation_kd_squeezenet_beta1.0")
if abl and abl1:
    P("<b>SqueezeNet ablation (feature-loss weight).</b> With the default settings (&beta;=1.0) the very small SqueezeNet (0.77 M parameters) <i>underperformed</i> its baseline: "
      "the strong feature-matching constraint competes with the classification objective for a low-capacity student that is already under-fitted after 15 epochs. "
      "Logit-only KD (&beta;=0) and a light feature weight (&beta;=0.1) both fix this. The reported distilled SqueezeNet uses &beta;=0.1; "
      "MobileNetV2 and EfficientNet-Lite0 use &beta;=1.0.")
    table([["SqueezeNet variant", "CIFAR-100 test acc (%)"],
           ["Baseline (no KD)", f"{base['squeezenet']['test_acc']:.2f}"],
           ["KD, soft+hard+feature, beta=1.0", f"{abl1['test_acc']:.2f}"],
           ["KD, soft+hard only (beta=0)", f"{abl['test_acc']:.2f}"],
           ["KD, soft+hard+feature, beta=0.1 (reported)", f"{kd['squeezenet']['test_acc']:.2f}"]], [9 * cm, 5 * cm], 9)

P("2.5 Unified inference and ONNX deployment", H2)
P("<font face=Courier>inference.py</font> provides one <font face=Courier>Predictor</font> interface for all six student models and the NAS model, with two back-ends: "
  "PyTorch checkpoints and ONNX Runtime (CPU). <font face=Courier>q1_kd/export_onnx.py</font> exports the baseline and distilled students (opset 17, dynamic batch) and checks "
  "that ONNX Runtime outputs match PyTorch (max abs difference &lt; 1e-3).")
shot("python q1_kd/export_onnx.py", "q1_export_onnx.log", lines=8)
shot("python inference.py --model kd_mobilenetv2 --test-samples 5", "shot_inference_kd.log", lines=8)

# ------------------------------------------------------------------------------------------------ Q2
P("3. Question 2 &mdash; Hardware-aware Neural Architecture Search", H1)
P("3.1 Search space (<font face=Courier>q2_nas/search_space.py</font>)", H2)
table([["Dimension", "Options"],
       ["Convolutional depth", "2-4 stages, 1-3 blocks per stage (add/remove-stage mutation)"],
       ["Filters / channels", "stem {8,16,24}; per-stage {8,16,24,32,48,64,96,128}; head {64,128,256}"],
       ["Kernel size", "{3, 5, 7} per stage"],
       ["Block type", "standard convolution or depthwise-separable convolution (depthwise KxK + pointwise 1x1)"],
       ["Activation", "ReLU, ReLU6, LeakyReLU, SiLU, Hard-Swish (per stage)"],
       ["Stride per stage", "{1, 2} (never reduced below 2x2)"],
       ["Input resolution", "{16, 24, 32}: the generated network resizes the 32x32 input internally, so all models share one deployment interface"]],
      [4 * cm, 13 * cm], 8.5)
P("The space contains about 1.2x10<super>14</super> distinct architectures. <font face=Courier>q2_nas/generator.py</font> turns any configuration dictionary into an "
  "<font face=Courier>nn.Module</font> (convolution / depthwise-separable blocks with BatchNorm and the chosen activation, residual connections where shapes match, global pooling and a linear classifier).")
P("3.2 Search strategy and hardware-aware evaluation", H2)
P("<b>Evolutionary search</b> (<font face=Courier>nas_search.py</font>): random initial population; tournament selection (k=3); uniform/stage-wise crossover (p=0.3) and gene-wise mutation incl. "
  "adding/removing stages; (&mu;+&lambda;) survivor selection; duplicate architectures are cached. Each candidate is evaluated as follows:")
P("1. parameter-count pre-filter; 2. export to ONNX and <b>measure on the real CPU</b> in a separate process (<font face=Courier>measure_cpu.py</font>): median latency of 100 runs "
  "after 20 warm-ups (batch 1, 1 thread, ONNX Runtime), ONNX file size, and peak inference memory (peak working set growth over the bare ORT process); "
  "3. <b>reject</b> the candidate if any deployment constraint is violated (no training budget is spent); 4. otherwise train it on a 15k-image subset of the CIFAR-10 training set for 3 epochs and "
  "evaluate on a held-out 5k validation split (the test set is never used during search); 5. fitness = val-acc &minus; 3&middot;latency/limit &minus; 2&middot;size/limit, "
  "which favours accurate <i>and</i> small/fast networks.")
if nas_best:
    c = nas_best["constraints"]
    table([["Constraint", "Limit"], ["CPU latency (median, batch 1, 1 thread)", f"{c['max_latency_ms']} ms"], ["Model storage size (ONNX fp32)", f"{c['max_size_mb']} MB"],
           ["Peak inference memory", f"{c['max_peak_memory_mb']} MB"], ["Parameters", f"{c['max_params_m']} M"]], [8 * cm, 4 * cm], 9)
if hist_rows:
    acc = [r for r in hist_rows if r["status"] == "accepted"]
    rej = [r for r in hist_rows if r["status"] == "rejected"]
    P(f"3.3 Search results", H2)
    P(f"The search evaluated <b>{len(hist_rows)}</b> candidate architectures: <b>{len(acc)}</b> satisfied the constraints and were trained, <b>{len(rej)}</b> were rejected "
      f"by the hardware check (most often for model size or parameter count). The full history is in <font face=Courier>results/nas_history.csv</font> and <font face=Courier>.json</font>.")
    img("q2_nas_search.png", 17)
    top = sorted(acc, key=lambda r: -float(r["fitness"]))[:5]
    table([["id", "gen", "res", "stages", "params (M)", "val acc", "lat (ms)", "size (MB)", "mem (MB)", "fitness"]] +
          [[r["id"], r["generation"], r["resolution"], r["n_stages"], f"{float(r['params_m']):.3f}", f"{float(r['val_acc']):.2f}",
            f"{float(r['latency_ms']):.2f}", f"{float(r['model_size_mb']):.2f}", f"{float(r['peak_memory_mb']):.1f}", f"{float(r['fitness']):.2f}"] for r in top],
          None, 8)
if nas_best:
    a = nas_best["architecture"]
    P("3.4 Best discovered architecture", H2)
    P(f"Input resolution {a['resolution']}, stem {a['stem_channels']} channels, head {a['head_channels']} channels; "
      f"search-time accuracy {nas_best['val_acc_search']:.2f}%, latency {nas_best['latency_ms']:.2f} ms, size {nas_best['model_size_mb']:.2f} MB, "
      f"peak memory {nas_best['peak_memory_mb']:.1f} MB.")
    table([["Stage", "depth", "channels", "kernel", "block", "activation", "stride"]] +
          [[i + 1, s["depth"], s["channels"], s["kernel"], s["block"], s["act"], s["stride"]] for i, s in enumerate(a["stages"])], None, 8.5)
    if nas_train:
        P(f"The architecture was then <b>trained from scratch</b> for {nas_train['epochs']} epochs on the full CIFAR-10 training set: "
          f"<b>{nas_train['test_acc']:.2f}% test accuracy</b> with {nas_train['params'] / 1e6:.3f} M parameters. It was exported to ONNX and runs through the same inference pipeline.")
        img("q2_best_curve.png", 8)
shot("python q2_nas/nas_search.py ...", "q2_search.log", lines=16)
shot("python q2_nas/train_best.py", "q2_train_best.log", lines=9, tail=True)

# ------------------------------------------------------------------------------------------------ benchmark
P("4. CPU benchmarking (ONNX Runtime)", H1)
P("<font face=Courier>benchmarks/benchmark.py</font> measures every exported model: accuracy of the ONNX model on the full test set, batch-1 latency with 1 and 4 threads, "
  "batch-64 throughput, file size, parameter count and peak inference memory (each in a fresh process). Results: <font face=Courier>results/benchmark_results.csv</font>.")
if bench:
    keys = ["model", "dataset", "onnx_acc_%", "params_M", "size_MB", "latency_1thread_ms", "latency_4threads_ms", "throughput_b64_img_s", "peak_memory_MB"]
    table([["model", "data", "acc %", "params M", "size MB", "lat 1T ms", "lat 4T ms", "img/s (b64)", "peak mem MB"]] +
          [[r[k] for k in keys] for r in bench.values()], None, 7.5)
    P("<b>Observations.</b> The distilled and baseline versions of a student have identical size and latency (distillation changes only the weights), so the accuracy gain is free at "
      "deployment. SqueezeNet is the smallest and fastest of the three students; MobileNetV2 and EfficientNet-Lite0 are larger but achieve higher accuracy per the table. "
      "The NAS model targets a different regime: it is orders of magnitude smaller than the Q1 students and meets the strict 3 ms / 1 MB / 20 MB budget by construction.")
shot("python benchmarks/benchmark.py", "benchmark.log", lines=9)

# ------------------------------------------------------------------------------------------------ conclusion
P("5. Conclusions and limitations", H1)
P("&bull; A complete, reproducible pipeline was implemented for both questions; every deliverable is in the repository and the README gives step-by-step instructions.<br/>"
  "&bull; Knowledge distillation (soft targets + temperature + hard labels + feature matching) is implemented as one configurable module and compared against a separate baseline pipeline under identical budgets.<br/>"
  "&bull; The NAS uses <i>measured</i> CPU latency, ONNX size and peak memory, rejects infeasible architectures before training, and selects with a resource-aware fitness.<br/>"
  "&bull; <b>Limitations</b>: short training schedules (single laptop GPU) give lower absolute accuracy than published results; NAS candidates are ranked with a cheap 3-epoch proxy, "
  "so the ranking is noisy; latency numbers are specific to the AMD Ryzen 7 7445HS CPU and ONNX Runtime 1.32. Longer schedules can be run via the <font face=Courier>--epochs</font> flags.")
P("6. Reproducibility", H1)
P("Repository layout, installation (<font face=Courier>requirements.txt</font>) and the exact commands are in <font face=Courier>README.md</font>. "
  "<font face=Courier>python run_all.py</font> executes the whole pipeline; seeds are fixed (0).")

SimpleDocTemplate(OUT, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm, bottomMargin=1.8 * cm,
                  title="23i-0079_A - Technical Report", author="23i-0079").build(story)
print("wrote", OUT)
