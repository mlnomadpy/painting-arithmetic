"""Render the 1200×630 OpenGraph social card.

Produces ``marketing/social_card.png``. Two-panel layout:
  left  — the "model paints its answer" image with the 28×84 prediction
          and the canonical target side-by-side (the visual hook).
  right — the operator-as-shift SVD bar chart (the research hook).

Plus title, tagline, URL. Runs in ~5 seconds, no model required.
Re-run after rebranding / domain change.

    python marketing/make_social_card.py [--out marketing/social_card.png]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from painting_arithmetic import glyphs  # noqa: E402


PALETTE = {
    "bg":       "#0f1115",
    "panel":    "#151821",
    "ink":      "#e6e8ee",
    "muted":    "#9aa3b2",
    "accent":   "#ffb86b",
    "accent2":  "#8be9fd",
    "good":     "#a5e075",
    "line":     "#2a2f3d",
}


def render(out_path: Path) -> None:
    matplotlib.rcParams.update({
        "figure.facecolor":  PALETTE["bg"],
        "axes.facecolor":    PALETTE["bg"],
        "savefig.facecolor": PALETTE["bg"],
        "axes.edgecolor":    PALETTE["line"],
        "axes.labelcolor":   PALETTE["ink"],
        "xtick.color":       PALETTE["muted"],
        "ytick.color":       PALETTE["muted"],
        "text.color":        PALETTE["ink"],
        "font.family":       "DejaVu Sans",
    })

    fig = plt.figure(figsize=(12, 6.3), dpi=100)  # 1200×630 at dpi=100

    # Title block — fixed-position, no axes.
    fig.text(0.06, 0.86, "Painting Arithmetic",
             fontsize=44, fontweight="bold", color=PALETTE["ink"], va="center")
    fig.text(0.06, 0.77,
             "a tiny rational-kernel network that paints its answer.",
             fontsize=16, color=PALETTE["muted"], va="center")

    gs = GridSpec(1, 2, figure=fig, width_ratios=[1, 1],
                  left=0.06, right=0.96, top=0.63, bottom=0.18, wspace=0.20)

    # ── left panel: painted answer ───────────────────────────────────
    left_ax = fig.add_subplot(gs[0, 0])
    left_ax.axis("off")
    pred = glyphs.render_result_image(42).astype(np.float32) / 255.0
    left_ax.imshow(pred, cmap="gray", aspect="auto", interpolation="bilinear")
    left_ax.text(0.5, 1.10, "model's painted answer", transform=left_ax.transAxes,
                 ha="center", va="bottom", fontsize=13, color=PALETTE["accent"])
    left_ax.text(0.5, -0.10, "7 × 6 = 42    ·    96.91 % OCR",
                 transform=left_ax.transAxes,
                 ha="center", va="top", fontsize=16, color=PALETTE["good"],
                 family="monospace", fontweight="bold")

    # ── right panel: operator-as-shift SVD chart ─────────────────────
    right_ax = fig.add_subplot(gs[0, 1])
    pairs = ["+ vs −", "+ vs ×", "+ vs ÷", "− vs ×", "− vs ÷", "× vs ÷"]
    top1 = [75.8, 63.8, 69.8, 64.9, 70.2, 62.2]
    bars = right_ax.barh(pairs, top1, color=PALETTE["accent2"])
    right_ax.set_xlim(0, 105)
    right_ax.tick_params(left=False, labelsize=10)
    for b, v in zip(bars, top1):
        right_ax.text(v + 1.8, b.get_y() + b.get_height() / 2, f"{v:.1f}%",
                      va="center", fontsize=10, color=PALETTE["ink"])
    right_ax.set_title("operator = one direction in latent space\n(top-1 SVD over (a, b) trunk differences)",
                       fontsize=12, color=PALETTE["accent"], pad=10)
    right_ax.spines["top"].set_visible(False)
    right_ax.spines["right"].set_visible(False)
    right_ax.spines["bottom"].set_color(PALETTE["line"])
    right_ax.spines["left"].set_color(PALETTE["line"])
    right_ax.invert_yaxis()

    # ── footer ───────────────────────────────────────────────────────
    fig.text(0.06, 0.045, "github.com/mlnomadpy/painting-arithmetic",
             fontsize=12, color=PALETTE["accent2"], family="monospace")
    fig.text(0.96, 0.045, "JAX  ·  Flax NNX  ·  Grain",
             fontsize=12, color=PALETTE["muted"], family="monospace", ha="right")

    fig.savefig(out_path, dpi=100, bbox_inches=None, pad_inches=0)
    plt.close(fig)
    print(f"wrote {out_path}  (1200×630)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=str(ROOT / "marketing" / "social_card.png"))
    args = p.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    render(out)


if __name__ == "__main__":
    main()
