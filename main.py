import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from tools.model_utils import make_model
from tools.denoise import DenoiseFrontEnd
from config import NUM_CHANNELS, NUM_CLASSES

INTENSITY_CLASSES = 3

def load_model(model_path: str, device: torch.device) -> torch.nn.Module:
    """Load model weights, including optional intensity head."""
    model = make_model(
        in_ch=NUM_CHANNELS,
        n_cls=NUM_CLASSES,
        deep_supervision=False,
        n_intensity_cls=INTENSITY_CLASSES,
    ).to(device)

    state = torch.load(model_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    model_state = model.state_dict()
    filtered = {k: v for k, v in state.items()
                if k in model_state and model_state[k].shape == v.shape}

    model.load_state_dict(filtered, strict=False)
    model.eval()
    return model


def build_denoise(device: torch.device) -> DenoiseFrontEnd:
    denoise = DenoiseFrontEnd(
        in_channels=NUM_CHANNELS,
        fs=500.0,
        band=(0.3, 45.0),
        mains_hz=50.0,
        mains_bw=1.2,
        use_2nd_harm=False,
        do_robust_norm=False,
    ).to(device)
    denoise.eval()
    for p in denoise.parameters():
        p.requires_grad = False
    return denoise


@torch.no_grad()
def predict(
    model: torch.nn.Module,
    denoise: DenoiseFrontEnd,
    array: np.ndarray,
    device: torch.device,
):
    """Return (emotion_idx, intensity_idx)."""
    x = torch.tensor(array, dtype=torch.float32, device=device)
    if x.ndim != 2:
        raise ValueError("Expected 2D array")
    # 确保通道在前 (NUM_CHANNELS, T)
    if x.shape[0] != NUM_CHANNELS and x.shape[1] == NUM_CHANNELS:
        x = x.T
    if x.shape[0] != NUM_CHANNELS:
        raise ValueError(f"Input should have {NUM_CHANNELS} channels")
    x = x.unsqueeze(0)  # (1, NUM_CHANNELS, T)

    x = denoise(x)
    outputs = model(x)
    if isinstance(outputs, tuple):
        logits = outputs[0]
        inten_logits = outputs[1] if len(outputs) > 1 else None
    else:
        logits = outputs
        inten_logits = None

    probs = torch.softmax(logits, dim=1).squeeze(0)
    pred = int(probs.argmax().item())
    max_prob = float(probs[pred].item())
    if inten_logits is not None:
        inten = int(torch.softmax(inten_logits, dim=1).squeeze(0).argmax().item())
    return pred, inten


def run(input_dir: str, output_dir: str, model_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path, device)
    denoise = build_denoise(device)

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "result.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        if NUM_CLASSES == 6:
            writer = csv.writer(f)
            writer.writerow(["file", "result", "intensity"])

            for path in sorted(input_dir.rglob("*.npy")):
                array = np.load(path)
                pred, intensity = predict(model, denoise, array, device)
                writer.writerow([path.name, pred, intensity])

def main():
    parser = argparse.ArgumentParser(description="EEG emotion classification inference")
    parser.add_argument("--input", required=True, help="folder with .npy files")
    parser.add_argument("--output", required=True, help="folder to save results")
    parser.add_argument("--model", required=True, help="model weight file")
    args = parser.parse_args()
    run(args.input, args.output, args.model)


if __name__ == "__main__":
    main()
