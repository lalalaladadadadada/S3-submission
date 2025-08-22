import json
from typing import List, Tuple, Optional
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import NUM_CHANNELS, NUM_CLASSES


# ───────────────────── Dataset（保持你的实现不变） ─────────────────────
class EEGJsonl(Dataset):
    """Simple dataset loader for the JSONL format used in this project."""
    def __init__(self, path: str):
        self.samples: List[Tuple[torch.Tensor, int]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                # 期待每个通道为同长度的1D序列（变长批次请在 collate 时做 pad）
                x = torch.tensor([obj["data"][f"C{i+1}"] for i in range(NUM_CHANNELS)],
                                 dtype=torch.float32)
                self.samples.append((x, int(obj["label"]["predicted_class"])))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        return self.samples[idx]


# ───────────────────── Building Blocks ─────────────────────
def make_norm(norm: str, c: int) -> nn.Module:
    if norm == "group":
        g = min(8, c) if c >= 8 else 1
        return nn.GroupNorm(g, c)
    elif norm == "batch":
        return nn.BatchNorm1d(c)
    else:
        raise ValueError(f"Unsupported norm: {norm}")


class SincConv1d(nn.Module):
    """
    SincNet-style 1D convolution with parametrized band-pass filters.
    Good inductive bias for EEG and few-shot settings.

    Args:
        out_channels: number of filters
        kernel_size: odd number (e.g., 129)
        sample_rate: e.g., 500 for your data
        min_low_hz: minimum low cutoff
        min_band_hz: minimum bandwidth
    """
    def __init__(
        self,
        out_channels: int,
        kernel_size: int = 129,
        sample_rate: float = 500.0,
        min_low_hz: float = 0.5,
        min_band_hz: float = 1.0,
    ):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("SincConv1d kernel_size must be odd.")
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.sample_rate = sample_rate
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz

        low_hz = torch.linspace(1.0, sample_rate / 2.0 - (min_band_hz + 1.0), out_channels)
        band_hz = torch.ones(out_channels) * (min_band_hz + 2.0)

        self.low_hz_ = nn.Parameter(low_hz)
        self.band_hz_ = nn.Parameter(band_hz)

        # Hamming window and time axis for filter construction
        n_lin = torch.linspace(0, kernel_size - 1, steps=kernel_size)
        self.register_buffer("window", 0.54 - 0.46 * torch.cos(2 * torch.pi * n_lin / (kernel_size - 1)))
        self.register_buffer("n", (n_lin - (kernel_size - 1) / 2.0) / sample_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, 1, T]
        returns: [B, out_channels, T]
        """
        low = self.min_low_hz + torch.abs(self.low_hz_)
        band = self.min_band_hz + torch.abs(self.band_hz_)
        high = torch.clamp(low + band, max=self.sample_rate / 2.0)

        f_times_t_low = 2 * low[:, None] * self.n[None, :]
        f_times_t_high = 2 * high[:, None] * self.n[None, :]

        band_pass = (torch.sin(torch.pi * f_times_t_high) - torch.sin(torch.pi * f_times_t_low)) / (torch.pi * self.n[None, :].clamp(min=1e-8))
        band_pass[:, self.kernel_size // 2] = 2 * (high - low)

        band_pass = band_pass * self.window[None, :]
        band_pass = band_pass / (2 * (high - low))[:, None]  # L1 normalize per filter

        filters = band_pass[:, None, :]  # [out_channels, 1, K]
        return F.conv1d(x, filters, stride=1, padding=self.kernel_size // 2, groups=1)


class SEBlock(nn.Module):
    def __init__(self, c: int, r: int = 8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(c, max(1, c // r), kernel_size=1),
            nn.SiLU(),
            nn.Conv1d(max(1, c // r), c, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(x)


class DSConv1d(nn.Module):
    """Depthwise-Separable Conv1d: efficient and strong for time series."""
    def __init__(self, in_ch: int, out_ch: int, k: int, d: int = 1, norm: str = "group", p_drop: float = 0.1):
        super().__init__()
        pad = (k - 1) // 2 * d
        self.dw = nn.Conv1d(in_ch, in_ch, kernel_size=k, padding=pad, dilation=d, groups=in_ch, bias=False)
        self.pw = nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False)
        self.norm = make_norm(norm, out_ch)
        self.act = nn.SiLU()
        self.do = nn.Dropout(p_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dw(x)
        x = self.pw(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.do(x)
        return x


class TCNBlock(nn.Module):
    def __init__(self, c: int, k: int, d: int, norm: str = "group", p_drop: float = 0.1):
        super().__init__()
        self.conv1 = DSConv1d(c, c, k, d, norm, p_drop)
        self.conv2 = DSConv1d(c, c, k, 1, norm, p_drop)
        self.se = SEBlock(c)
        self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.se(out)
        return F.silu(out + self.shortcut(x))


class AttnPool1d(nn.Module):
    """Masked attention pooling over time (supports variable lengths)."""
    def __init__(self, c: int):
        super().__init__()
        self.proj = nn.Conv1d(c, 1, kernel_size=1)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: [B, C, T], mask: [B, T] (True for valid, False for pad) or None
        returns: [B, C]
        """
        logits = self.proj(x).squeeze(1)  # [B, T]
        if mask is not None:
            logits = logits.masked_fill(~mask, float("-inf"))
        w = torch.softmax(logits, dim=-1)  # [B, T]
        pooled = torch.einsum("bct,bt->bc", x, w)
        return pooled


# ───────────────────── Main Model ─────────────────────
class FewShotEEGNet1D(nn.Module):
    """
    高阶少样本 EEG 分类模型
    结构：SincConv1D 前端 → 1x1 通道融合 → 多层 TCN 残差（扩张率递增）→ Transformer 编码 → 注意力池化
    训练时 deep_supervision 输出 aux heads（早层 feature 的 GAP 分类）
    """
    def __init__(
        self,
        in_ch: int = NUM_CHANNELS,
        n_cls: int = NUM_CLASSES,
        n_intensity_cls: int = 0,
        norm: str = "group",
        deep_supervision: bool = False,
        sinc_filters: int = 32,
        sinc_kernel: int = 129,
        sample_rate: float = 500.0,
        width: int = 128,
        num_tcn_blocks: int = 3,
        transformer_layers: int = 1,
        transformer_heads: int = 4,
        p_drop: float = 0.1,
    ):
        super().__init__()
        self.deep_supervision = deep_supervision

        # 1) Learnable filterbank per-channel via SincConv
        self.sinc = SincConv1d(out_channels=sinc_filters, kernel_size=sinc_kernel, sample_rate=sample_rate)

        # 2) Apply Sinc channelwise: (B, C, T) -> (B*C, 1, T) -> (B, C*sinc_filters, T)
        self.post_sinc_norm = make_norm(norm, in_ch * sinc_filters)
        self.post_sinc_act = nn.SiLU()

        # 3) Channel mixing to a compact width
        self.mix = nn.Conv1d(in_ch * sinc_filters, width, kernel_size=1, bias=False)
        self.mix_norm = make_norm(norm, width)
        self.mix_act = nn.SiLU()
        self.mix_do = nn.Dropout(p_drop)

        # 4) TCN residual stack with increasing dilation
        tcn_layers = []
        dil = 1
        for i in range(num_tcn_blocks):
            tcn_layers.append(TCNBlock(width, k=7, d=dil, norm=norm, p_drop=p_drop))
            dil *= 2
        self.tcn = nn.Sequential(*tcn_layers)

        # 5) Transformer encoder over time (mask-aware)
        enc_layer = nn.TransformerEncoderLayer(d_model=width, nhead=transformer_heads,
                                               dim_feedforward=width * 2, dropout=p_drop,
                                               batch_first=True, activation="gelu", norm_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=transformer_layers)

        # 6) Attention pooling + head
        self.attnpool = AttnPool1d(width)
        self.head = nn.Linear(width, n_cls)
        self.intensity_head = nn.Linear(width, n_intensity_cls) if n_intensity_cls > 0 else None

        # 7) Auxiliary heads (深监督)
        if self.deep_supervision:
            self.aux_gap1 = nn.AdaptiveAvgPool1d(1)
            self.aux_fc1 = nn.Linear(width, n_cls)
            self.aux_gap2 = nn.AdaptiveAvgPool1d(1)
            self.aux_fc2 = nn.Linear(width, n_cls)

        # init
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def _build_mask_from_lengths(self, T: int, lengths: torch.Tensor) -> torch.Tensor:
        # lengths: [B] int
        device = lengths.device
        idx = torch.arange(T, device=device)[None, :].expand(lengths.shape[0], -1)
        return idx < lengths[:, None]  # [B, T], True = valid

    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None):
        # input [Batch, Ch, Time] for [64, 2, ?]
        B, C, T = x.shape

        # 构造 mask（如果没提供 lengths，就把非零时间点当作有效）
        if lengths is not None:
            mask_bt = self._build_mask_from_lengths(T, lengths)  # [B, T]
        else:
            mask_bt = (x.abs().sum(dim=1) > 0)  # [B, T], 假设 pad 为0

        # 1) SincConv per-channel
        x_ = x.reshape(B * C, 1, T)
        x_ = self.sinc(x_)                         # [B*C, F, T]
        x_ = x_.reshape(B, C * x_.shape[1], T)     # [B, C*F, T]
        x_ = self.post_sinc_act(self.post_sinc_norm(x_))

        # 2) Channel mixing
        x_ = self.mix(x_)
        x_ = self.mix_act(self.mix_norm(x_))
        x_ = self.mix_do(x_)                       # [B, W, T]

        # 3) TCN stack（残差，保持 T 不变）
        #    为了 deep supervision，抓取中间两个点
        aux1 = self.tcn[0](x_) if len(self.tcn) > 0 else x_
        x_cur = aux1
        if len(self.tcn) > 1:
            aux2 = self.tcn[1](x_cur)
            x_cur = aux2
            for blk in self.tcn[2:]:
                x_cur = blk(x_cur)
        else:
            aux2 = aux1

        # 4) Transformer（时间维度置后）
        x_seq = x_cur.transpose(1, 2)              # [B, T, W]
        # key_padding_mask: True=需要mask
        key_padding_mask = ~mask_bt                # [B, T]
        x_seq = self.transformer(x_seq, src_key_padding_mask=key_padding_mask)

        # 5) Attention pooling（带 mask）
        x_seq_t = x_seq.transpose(1, 2)            # [B, W, T]
        pooled = self.attnpool(x_seq_t, mask_bt)   # [B, W]
        main = self.head(pooled)                   # [B, n_cls]
        inten = self.intensity_head(pooled) if self.intensity_head is not None else None

        if self.deep_supervision and self.training:
            a1 = self.aux_fc1(self.aux_gap1(aux1).squeeze(-1))
            a2 = self.aux_fc2(self.aux_gap2(aux2).squeeze(-1))
            if inten is not None:
                return main, inten, [a1, a2]
            return main, [a1, a2]
        if inten is not None:
            return main, inten
        return main


# ───────────────────── Factory & Loss ─────────────────────
def make_model(
    in_ch: int = NUM_CHANNELS,
    n_cls: int = NUM_CLASSES,
    norm: str = "group",
    deep_supervision: bool = False,
    n_intensity_cls: int = 0,
):
    return FewShotEEGNet1D(
        in_ch=in_ch,
        n_cls=n_cls,
        n_intensity_cls=n_intensity_cls,
        norm=norm,
        deep_supervision=deep_supervision,
    )


def cb_focal_loss(logits, targets, counts, beta=0.9, gamma=2.0):
    effective_num = 1.0 - torch.pow(beta, counts)
    weights = (1.0 - beta) / effective_num
    weights = weights / weights.sum() * len(counts)
    weights = weights.to(logits.device)

    ce = F.cross_entropy(logits, targets, reduction="none")
    pt = torch.exp(-ce)
    focal = (1 - pt) ** gamma
    loss = weights[targets] * focal * ce
    return loss.mean()
