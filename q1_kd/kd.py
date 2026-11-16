"""Knowledge-distillation losses.

Total loss  =  (1-alpha) * CE(student, y)                            (hard labels)
             +  alpha * T^2 * KL(softmax(t/T) || softmax(s/T))        (soft targets, temperature softmax)
             +  beta  * sum_l FeatLoss(proj_l(student_l), teacher_l)  (intermediate features)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def soft_target_loss(student_logits, teacher_logits, T):
    """Hinton KD: KL divergence between temperature-softened distributions, scaled by T^2."""
    log_p_s = F.log_softmax(student_logits.float() / T, dim=1)
    p_t = F.softmax(teacher_logits.float() / T, dim=1)
    return F.kl_div(log_p_s, p_t, reduction="batchmean") * (T * T)


class FeatureDistiller(nn.Module):
    """1x1-conv regressors map student feature maps to the teacher's channel dimension; the loss is a
    scale-invariant MSE between L2-normalised (along channels) maps. Spatial sizes are matched by
    adaptive average pooling if they differ."""

    def __init__(self, student_channels, teacher_channels):
        super().__init__()
        self.proj = nn.ModuleList(
            nn.Sequential(nn.Conv2d(sc, tc, 1, bias=False), nn.BatchNorm2d(tc))
            for sc, tc in zip(student_channels, teacher_channels))

    def forward(self, s_feats, t_feats):
        loss = 0.0
        for p, fs, ft in zip(self.proj, s_feats, t_feats):
            fs = p(fs).float()
            ft = ft.float()
            if fs.shape[-2:] != ft.shape[-2:]:
                fs = F.adaptive_avg_pool2d(fs, ft.shape[-2:])
            fs, ft = F.normalize(fs, dim=1), F.normalize(ft, dim=1)
            loss = loss + ((fs - ft) ** 2).sum(1).mean()  # = 2 - 2*cos(student, teacher) per location
        return loss


class DistillationLoss(nn.Module):
    def __init__(self, T=4.0, alpha=0.7, beta=1.0, use_feat=True, student_channels=None, teacher_channels=None):
        super().__init__()
        self.T, self.alpha, self.beta, self.use_feat = T, alpha, beta, use_feat
        self.feat = FeatureDistiller(student_channels, teacher_channels) if use_feat else None

    def forward(self, s_logits, s_feats, t_logits, t_feats, y):
        hard = F.cross_entropy(s_logits.float(), y)
        soft = soft_target_loss(s_logits, t_logits, self.T)
        loss = (1 - self.alpha) * hard + self.alpha * soft
        if self.use_feat:
            loss = loss + self.beta * self.feat(s_feats, t_feats)
        return loss, None
