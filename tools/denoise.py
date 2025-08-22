import math
from typing import Optional
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import NUM_CHANNELS


def _sinc(x: torch.Tensor) -> torch.Tensor:
    return torch.where(x == 0, torch.ones_like(x), torch.sin(math.pi * x) / (math.pi * x))


def design_fir(kind: str, fs: float, ksize: int,
               f1: Optional[float] = None, f2: Optional[float] = None,
               eps: float = 1e-6) -> torch.Tensor:
    """简单的窗函数 FIR 设计工具。"""
    assert ksize % 2 == 1, "FIR kernel size must be odd"
    n = torch.arange(ksize, dtype=torch.float32)
    M = ksize - 1
    t = n - M/2

    def lp(fc):
        return 2*fc/fs * _sinc(2*fc/fs * t)

    if kind == "lowpass":
        h = lp(f2)
    elif kind == "highpass":
        h = -lp(f1); h[M//2] += 1.0
    elif kind == "bandpass":
        h = lp(f2) - lp(f1)
    elif kind == "bandstop":
        h = - (lp(f2) - lp(f1)); h[M//2] += 1.0
    else:
        raise ValueError(kind)

    window = 0.54 - 0.46*torch.cos(2*math.pi*n/M)
    h = h * window
    h = h / (h.sum() + eps)
    return h


class FixedFIR1D(nn.Module):
    """固定系数的一维卷积实现 FIR 滤波。"""
    def __init__(self, in_channels: int, kernel: torch.Tensor):
        super().__init__()
        self.in_channels = in_channels
        self.register_buffer("kernel", kernel.view(1,1,-1))  # (1,1,K)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        K = self.kernel.size(-1)
        pad = K // 2
        # "reflect" padding requires the padding size to be smaller than the
        # input length, otherwise PyTorch raises an error.  When processing very
        # short segments (e.g. 0.5 s signals with only 250 samples), the FIR
        # kernel (K=1001) would need padding larger than the signal itself.  In
        # such cases fall back to zero padding which has no such restriction.
        if x.size(-1) <= pad:
            x = F.pad(x, (pad, pad), mode="constant")
        else:
            x = F.pad(x, (pad, pad), mode="reflect")
        weight = self.kernel.repeat(self.in_channels, 1, 1)  # (C,1,K)
        return F.conv1d(x, weight, groups=self.in_channels)


class DenoiseFrontEnd(nn.Module):
    """带通 + 工频陷波的简单去噪前端。"""
    def __init__(self, in_channels=NUM_CHANNELS, fs=500.0,
                 band=(0.3, 45.0),
                 mains_hz: Optional[float] = 50.0,
                 mains_bw: float = 1.2,
                 use_2nd_harm: bool = False,
                 do_robust_norm: bool = False):
        super().__init__()
        self.do_robust_norm = do_robust_norm
        bp = design_fir("bandpass", fs, ksize=1001, f1=band[0], f2=band[1])
        self.bp = FixedFIR1D(in_channels, bp)
        notches = []
        if mains_hz is not None:
            f0 = float(mains_hz)
            if f0 < fs/2 - 0.5:
                k = design_fir("bandstop", fs, ksize=401, f1=f0- mains_bw/2, f2=f0+ mains_bw/2)
                notches.append(FixedFIR1D(in_channels, k))
            if use_2nd_harm and (2*f0 < fs/2 - 0.5):
                k2 = design_fir("bandstop", fs, ksize=401, f1=2*f0- mains_bw/2, f2=2*f0+ mains_bw/2)
                notches.append(FixedFIR1D(in_channels, k2))
        self.notches = nn.ModuleList(notches)

    @staticmethod
    def _robust_standardize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        med = x.median(dim=-1, keepdim=True).values
        mad = (x - med).abs().median(dim=-1, keepdim=True).values
        scale = torch.clamp(1.4826*mad, min=eps)
        x = (x - med) / scale
        x = torch.tanh(x/4.0) * 4.0
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.bp(x)
        for notch in self.notches:
            y = notch(y)
        if self.do_robust_norm:
            y = self._robust_standardize(y)
        return y
