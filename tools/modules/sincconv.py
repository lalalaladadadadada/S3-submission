import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SincConv1D(nn.Module):
    """Sinc-based 1D convolution."""

    def __init__(
        self,
        out_channels: int,
        kernel_size: int,
        sample_rate: int = 500,
        in_channels: int = 1,
        min_low_hz: float = 0.5,
        min_band_hz: float = 1.0,
        max_freq: float | None = None,
    ) -> None:
        super().__init__()
        self.out_channels = out_channels
        if kernel_size % 2 == 0:
            kernel_size += 1
        self.kernel_size = kernel_size
        self.sample_rate = sample_rate
        self.in_channels = in_channels
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz
        self.max_freq = max_freq if max_freq is not None else sample_rate / 2

        low_hz = 0.5
        high_hz = self.max_freq - (self.min_low_hz + self.min_band_hz)
        hz = np.linspace(low_hz, high_hz, out_channels + 1)

        self.low_hz_ = nn.Parameter(torch.Tensor(hz[:-1]))
        self.band_hz_ = nn.Parameter(torch.Tensor(np.diff(hz)))

        n_lin = torch.linspace(0, self.kernel_size, steps=self.kernel_size)
        window = 0.54 - 0.46 * torch.cos(2 * np.pi * n_lin / self.kernel_size)
        self.register_buffer("window", window)
        self.n_ = (self.kernel_size - 1) / 2.0
        self.register_buffer("n", torch.linspace(-self.n_, self.n_, steps=self.kernel_size))

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        self.n = self.n.to(waveforms.device)
        self.window = self.window.to(waveforms.device)

        low = self.min_low_hz + torch.abs(self.low_hz_)
        high = torch.clamp(
            low + self.min_band_hz + torch.abs(self.band_hz_),
            self.min_low_hz,
            self.max_freq,
        )

        f_times_t_low = torch.matmul(low[:, None], self.n[None, :])
        f_times_t_high = torch.matmul(high[:, None], self.n[None, :])

        band_pass = (
            torch.sin(2 * np.pi * f_times_t_high)
            - torch.sin(2 * np.pi * f_times_t_low)
        ) / (self.n[None, :] / self.sample_rate + 1e-6)
        band_pass[:, self.kernel_size // 2] = 2 * (high - low).squeeze() / self.sample_rate
        band_pass = band_pass * self.window[None, :]
        filters = band_pass.view(self.out_channels, 1, self.kernel_size)
        filters = filters.repeat(self.in_channels, 1, 1)

        return F.conv1d(
            waveforms,
            filters,
            stride=1,
            padding=self.kernel_size // 2,
            groups=self.in_channels,
        )

    def show_filters(self) -> None:
        """Print and plot learned frequency bands."""
        import matplotlib.pyplot as plt

        with torch.no_grad():
            low = self.min_low_hz + torch.abs(self.low_hz_)
            high = torch.clamp(
                low + self.min_band_hz + torch.abs(self.band_hz_),
                self.min_low_hz,
                self.max_freq,
            )
            low = low.cpu().numpy()
            high = high.cpu().numpy()

        print("Learned Bandpass Filters (Hz):")
        for i, (l, h) in enumerate(zip(low, high)):
            print(f"Filter {i}: {l:.2f} - {h:.2f} Hz")

        plt.figure(figsize=(10, 4))
        for i in range(len(low)):
            plt.plot([low[i], high[i]], [i, i], lw=4)
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Filter Index")
        plt.title("Learned Frequency Bands")
        plt.grid(True)
        plt.show()

    def save_filters(self, path: str) -> None:
        """Save learned frequency bands to a JSON file."""
        import json

        with torch.no_grad():
            low = self.min_low_hz + torch.abs(self.low_hz_)
            high = torch.clamp(
                low + self.min_band_hz + torch.abs(self.band_hz_),
                self.min_low_hz,
                self.max_freq,
            )

        bands = [
            {"low_hz": float(l), "high_hz": float(h)}
            for l, h in zip(low, high)
        ]

        with open(path, "w", encoding="utf-8") as f:
            json.dump(bands, f, ensure_ascii=False, indent=2)

