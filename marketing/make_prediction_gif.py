"""Generate an animated GIF showing the model paint answers for a sequence
of (a, op, b) probes.

Each frame: three input thumbnails (digit, operator, digit) on the left
and the model's painted 28×84 answer on the right. The GIF cycles through
~12 hand-picked problems that show off addition / subtraction (negative
results) / multiplication / division.

Requires a trained checkpoint. By default it uses the PyTorch v3.1
checkpoint at .context/visual_arithmetic/model_v3.pt — that's the
validated one. To use the JAX checkpoint instead, pass --jax-ckpt.

    python marketing/make_prediction_gif.py
    python marketing/make_prediction_gif.py --jax-ckpt ./ckpts/model.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import imageio.v3 as iio
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from painting_arithmetic import glyphs  # noqa: E402

PROBES = [
    (3, "+", 5),
    (7, "+", 4),
    (9, "+", 9),
    (5, "*", 8),
    (7, "*", 6),
    (9, "*", 9),
    (8, "-", 3),
    (3, "-", 8),
    (9, "-", 9),
    (6, "//", 2),
    (9, "//", 4),
    (0, "*", 7),
]


PALETTE = {
    "bg": "#0f1115",
    "panel": "#151821",
    "ink": "#e6e8ee",
    "muted": "#9aa3b2",
    "accent": "#ffb86b",
    "good": "#a5e075",
}


def _torch_predict_fn(ckpt_path: Path):
    """Return ``predict(a_img, op_idx, b_img) -> (28, 84) float32`` using a
    PyTorch v3.1 checkpoint."""
    import torch
    from torchvision import datasets, transforms

    # The PyTorch v3.1 checkpoint was trained against the older operator-
    # glyph rendering that lives next to it under .context/visual_arithmetic/.
    # We import THAT glyphs module here so the operator pad we feed the
    # model is in-distribution.  (The new painting_arithmetic.glyphs module
    # crops glyphs differently and would be out-of-distribution for the v3.1
    # weights.)
    va_dir = Path(__file__).resolve().parents[2] / "visual_arithmetic"
    sys.path.insert(0, str(va_dir))
    from train_v3 import YatArithmeticGen  # noqa: E402
    import glyphs as old_glyphs  # noqa: E402

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    model = YatArithmeticGen()
    model.load_state_dict(torch.load(str(ckpt_path), map_location=device, weights_only=True))
    model = model.to(device).eval()

    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    # Try the local ./data first, then a couple of well-known fallbacks.
    for cand in [ROOT / "data", ROOT.parent.parent / ".data", ROOT.parent / ".data"]:
        if (cand / "MNIST").exists():
            data_root = cand
            break
    else:
        data_root = ROOT / "data"
    test_ds = datasets.MNIST(str(data_root), train=False, download=True, transform=tf)
    by_digit: dict[int, "torch.Tensor"] = {}
    for img, lbl in test_ds:
        if lbl not in by_digit:
            by_digit[lbl] = img
        if len(by_digit) == 10:
            break

    def predict(a: int, op_idx: int, b: int) -> np.ndarray:
        op_arr = (np.asarray(old_glyphs.render_clean(op_idx), np.float32) / 255.0 - 0.1307) / 0.3081
        ia = by_digit[a].unsqueeze(0).to(device)
        io = torch.from_numpy(op_arr).unsqueeze(0).unsqueeze(0).to(device)
        ib = by_digit[b].unsqueeze(0).to(device)
        with torch.no_grad():
            img, *_ = model(ia, io, ib)
        return img[0, 0].cpu().numpy()

    def get_thumb(d: int) -> np.ndarray:
        # Reverse the MNIST normalisation for display.
        return (by_digit[d].squeeze(0).cpu().numpy() * 0.3081 + 0.1307).clip(0, 1)

    return predict, get_thumb


# ---------------------------------------------------------------------------
# Frame composition
# ---------------------------------------------------------------------------


def _compose_frame(a_thumb, op_idx, b_thumb, pred, truth, ocr):
    matplotlib.rcParams.update({
        "figure.facecolor": PALETTE["bg"],
        "axes.facecolor":   PALETTE["bg"],
        "savefig.facecolor": PALETTE["bg"],
        "font.family": "DejaVu Sans",
    })
    # Compact size for Twitter/HF — 720×260 at dpi=80.
    fig, axes = plt.subplots(
        1, 4, figsize=(9.0, 3.2), dpi=80,
        gridspec_kw={"width_ratios": [1, 1, 1, 3.4]}
    )
    axes[0].imshow(a_thumb, cmap="gray", interpolation="bilinear")
    axes[0].set_title("a", fontsize=12, color=PALETTE["muted"])
    # Use the same glyph the inference-time encoder sees.
    va_dir = Path(__file__).resolve().parents[2] / "visual_arithmetic"
    if str(va_dir) not in sys.path:
        sys.path.insert(0, str(va_dir))
    import glyphs as og  # noqa: F401  (already imported elsewhere if torch path ran)
    axes[1].imshow(np.asarray(og.render_clean(op_idx)),
                   cmap="gray", interpolation="bilinear")
    axes[1].set_title("op", fontsize=12, color=PALETTE["muted"])
    axes[2].imshow(b_thumb, cmap="gray", interpolation="bilinear")
    axes[2].set_title("b", fontsize=12, color=PALETTE["muted"])
    axes[3].imshow(pred, cmap="gray", aspect="auto", interpolation="bilinear")
    match = "✓" if ocr == truth else "✗"
    axes[3].set_title(
        f"the model paints  →  {ocr}    (truth: {truth})  {match}",
        fontsize=13, color=PALETTE["good"] if ocr == truth else PALETTE["accent"],
        fontweight="bold",
    )
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.canvas.draw()
    # NEW matplotlib API
    rgba = np.asarray(fig.canvas.buffer_rgba())
    plt.close(fig)
    return rgba[..., :3]


def _ocr(pred: np.ndarray) -> int:
    """OCR predicted image against the target-rendering function used to
    train this specific checkpoint. The PyTorch v3.1 model was trained
    against the older renderer in .context/visual_arithmetic/glyphs.py."""
    va_dir = Path(__file__).resolve().parents[2] / "visual_arithmetic"
    if str(va_dir) not in sys.path:
        sys.path.insert(0, str(va_dir))
    import glyphs as og
    targets = np.stack(
        [og.render_result_image(r).astype(np.float32) / 255.0 for r in range(-9, 82)]
    )
    d = ((pred[None] - targets) ** 2).sum(axis=(1, 2))
    return int(np.argmin(d)) + glyphs.RESULT_MIN


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--torch-ckpt",
                   default=str(ROOT.parent / "visual_arithmetic" / "model_v3.pt"),
                   help="PyTorch v3.1 checkpoint path.")
    p.add_argument("--out", default=str(ROOT / "marketing" / "prediction.gif"))
    p.add_argument("--frame-ms", type=int, default=1200,
                   help="Per-frame duration in milliseconds.")
    args = p.parse_args()

    ckpt = Path(args.torch_ckpt)
    if not ckpt.is_file():
        sys.exit(f"checkpoint not found: {ckpt}\n"
                 "Train one first via `painting-arithmetic-train` (JAX) "
                 "or `python .context/visual_arithmetic/train_v3.py` (PyTorch).")

    print(f"loading checkpoint {ckpt}")
    predict, get_thumb = _torch_predict_fn(ckpt)

    op_index = {"+": 0, "-": 1, "*": 2, "//": 3}
    op_disp = {"+": "+", "-": "−", "*": "×", "//": "÷"}

    frames = []
    for a, op, b in PROBES:
        truth = {"+": a + b, "-": a - b, "*": a * b, "//": a // max(1, b)}[op]
        pred = predict(a, op_index[op], b)
        ocr = _ocr(pred)
        frame = _compose_frame(get_thumb(a), op_index[op], get_thumb(b), pred, truth, ocr)
        frames.append(frame)
        print(f"  {a:>2} {op_disp[op]} {b:>2} = {truth:>3}   model: {ocr:>3}   "
              f"{'✓' if ocr == truth else '✗'}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(out, frames, duration=args.frame_ms, loop=0)
    size = out.stat().st_size / 1024
    print(f"\nwrote {out}  ({size:.0f} KB, {len(frames)} frames @ {args.frame_ms} ms)")


if __name__ == "__main__":
    main()
