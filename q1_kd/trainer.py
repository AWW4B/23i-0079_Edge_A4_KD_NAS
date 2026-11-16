"""Shared training loop used by the teacher, the baseline students and the distilled students."""
import time
import torch
import torch.nn.functional as F
from common.utils import evaluate


def fit(model, train, test, epochs, lr, bs, wd=5e-4, teacher=None, kd_loss=None, label_smoothing=0.0, log=print,
        warmup_frac=0.15):
    """SGD + OneCycle schedule + mixed precision. If `teacher` and `kd_loss` are given, performs distillation."""
    dev = "cuda"
    torch.backends.cudnn.benchmark = True
    model = model.to(dev).to(memory_format=torch.channels_last)
    if kd_loss is not None:
        kd_loss.to(dev)
    params = list(model.parameters()) + (list(kd_loss.parameters()) if kd_loss is not None else [])
    opt = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=wd, nesterov=True)
    steps = epochs * ((len(train) + bs - 1) // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=warmup_frac)
    scaler = torch.amp.GradScaler()
    if teacher is not None:
        teacher.eval()
    hist, best = [], 0.0
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        if kd_loss is not None:
            kd_loss.train()
        tl = n = 0
        for xb, yb in train.batches(bs):
            xb = xb.contiguous(memory_format=torch.channels_last)
            with torch.autocast("cuda", dtype=torch.float16):
                if teacher is not None:
                    with torch.no_grad():
                        t_logits, t_feats = teacher(xb, return_feats=True)
                    s_logits, s_feats = model(xb, return_feats=True)
                    loss, _ = kd_loss(s_logits, s_feats, t_logits, t_feats, yb)
                else:
                    out = model(xb)
                    loss = F.cross_entropy(out.float(), yb, label_smoothing=label_smoothing)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tl += loss.item() * len(yb)
            n += len(yb)
        acc = evaluate(model, test)
        best = max(best, acc)
        hist.append({"epoch": ep, "train_loss": tl / n, "test_acc": acc})
        log(f"epoch {ep:3d}/{epochs}  loss {tl / n:.4f}  test_acc {acc:.2f}  best {best:.2f}  ({time.time() - t0:.0f}s)")
    return hist
