"""Visualise the operator-ablation benchmark.

Reads ``experiments/intervene_out/results.json`` and writes two figures:

  fig_specificity_matrix.png   3 heatmaps of Δ-OCR matrices (one per intervention)
  fig_specificity_score.png    a single bar chart: targeted drop − collateral drop

Run::
    python experiments/intervene_viz.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / "visual_arithmetic"))   # for viz_style.py

import viz_style as S


OPS = ["+", "-", "*", "//"]
OP_GLYPHS = ["+", "−", "×", "÷"]
INTERVENTIONS = ["row", "slot", "random"]
INTERVENTION_TITLES = {
    "row":    "row ablation\n(zero unit's full kernel column + bias)",
    "slot":   "middle-slot zero\n(zero only the operator-slot weights)",
    "random": "random baseline\n(same N, randomly chosen)",
}


def _delta_matrix(results: dict, intervention: str) -> np.ndarray:
    """Δ-OCR matrix (% points). Rows = killed op, cols = evaluated op."""
    base = results["baseline"]
    iv = results["interventions"][intervention]
    M = np.zeros((4, 4), dtype=np.float32)
    for i, kill in enumerate(OPS):
        for j, ev in enumerate(OPS):
            M[i, j] = (iv[kill][ev] - base[ev]) * 100.0
    return M


def _specificity_table(results: dict) -> dict[str, dict]:
    """For each intervention, compute the diagonal drop and the mean off-diagonal drop."""
    out: dict[str, dict] = {}
    for iv in INTERVENTIONS:
        M = _delta_matrix(results, iv)
        diag = np.diag(M)
        off = M.copy()
        np.fill_diagonal(off, np.nan)
        off_mean = np.nanmean(off, axis=1)
        out[iv] = {
            "delta": M,
            "targeted_drop":   -diag,
            "collateral_drop": -off_mean,
            "specificity":     -diag + off_mean,
        }
    return out


# ---------------------------------------------------------------------------
# Figure 1 — three Δ heatmaps side by side
# ---------------------------------------------------------------------------


def fig_specificity_matrix(results: dict, out_path: Path) -> None:
    table = _specificity_table(results)
    vmax = max(abs(table[iv]["delta"]).max() for iv in INTERVENTIONS)
    vmax = max(vmax, 6.0)   # at least a 6-pp scale so the colourbar is readable

    with S.themed():
        fig = plt.figure(figsize=(15, 6.8))
        gs = fig.add_gridspec(1, 3, left=0.05, right=0.93, bottom=0.18, top=0.72, wspace=0.30)
        S.add_title_block(fig, S.TitleBlock(
            title="Can we surgically remove one operator from the trunk?",
            subtitle="Δ OCR vs baseline (% points) per (killed op, evaluated op). "
                     "Diagonal = targeted; off-diagonal = collateral.",
            caption="Phased model, 6 000-sample test set · operator-specialised h₁ units "
                    "selected via the prototype gallery's purity ≥ 0.5 criterion",
        ), top=0.93)

        for k, iv in enumerate(INTERVENTIONS):
            ax = fig.add_subplot(gs[0, k])
            M = table[iv]["delta"]
            im = ax.imshow(M, cmap=S.SEISMIC_CMAP, vmin=-vmax, vmax=vmax,
                           aspect="equal", interpolation="nearest")
            ax.set_xticks(range(4)); ax.set_yticks(range(4))
            ax.set_xticklabels(OP_GLYPHS, fontsize=14, color=S.INK)
            ax.set_yticklabels(OP_GLYPHS, fontsize=14, color=S.INK)
            ax.set_xlabel("evaluated", color=S.MUTED, fontsize=11)
            if k == 0:
                ax.set_ylabel("killed", color=S.MUTED, fontsize=11)
            ax.set_title(INTERVENTION_TITLES[iv], color=S.INK, fontsize=12.5,
                         fontweight="bold", loc="left", pad=14)
            for sp in ax.spines.values():
                sp.set_visible(False)
            ax.tick_params(length=0)

            # Annotate every cell with the Δ value.
            for i in range(4):
                for j in range(4):
                    val = M[i, j]
                    txt_col = S.INK if abs(val) < vmax * 0.55 else S.BG
                    ax.text(j, i, f"{val:+.1f}",
                            ha="center", va="center",
                            color=txt_col, fontsize=11.5,
                            fontweight="bold" if i == j else "normal")

            # Outline the diagonal.
            for d in range(4):
                ax.add_patch(plt.Rectangle((d - 0.5, d - 0.5), 1, 1,
                                            fill=False, edgecolor=S.ACCENT,
                                            linewidth=1.6))

        cb = S.colorbar(ax, im, label="Δ OCR (pp)")

        fig.savefig(out_path)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2 — specificity score (targeted drop − collateral drop)
# ---------------------------------------------------------------------------


def fig_specificity_score(results: dict, out_path: Path) -> None:
    table = _specificity_table(results)

    with S.themed():
        fig = plt.figure(figsize=(13.5, 6.4))
        gs = fig.add_gridspec(1, 1, left=0.07, right=0.97, bottom=0.16, top=0.70)
        S.add_title_block(fig, S.TitleBlock(
            title="No operator is cleanly localised in the trunk",
            subtitle="For each intervention, specificity = (drop on the killed op) − "
                     "(mean drop on the other three). Positive = surgical; "
                     "zero / negative = collateral damage matches targeted damage.",
            caption="Bars at zero (or below) mean the random baseline does the same damage "
                    "the targeted edit does — the 'operator units' aren't an operator circuit.",
        ), top=0.93)

        ax = fig.add_subplot(gs[0, 0])

        x = np.arange(4)
        w = 0.26
        offsets = {"row": -w, "slot": 0.0, "random": w}
        colors = {"row": S.ACCENT, "slot": S.ACCENT2, "random": S.MUTED}

        for iv in INTERVENTIONS:
            spec = table[iv]["specificity"]
            ax.bar(x + offsets[iv], spec, width=w,
                   color=colors[iv], edgecolor="none",
                   label=INTERVENTION_TITLES[iv].split("\n")[0])

        ax.axhline(0, color=S.PANEL_EDGE, linewidth=0.8, zorder=1)
        ax.set_xticks(x)
        ax.set_xticklabels([f"kill {g}" for g in OP_GLYPHS], fontsize=13, color=S.INK)
        ax.set_ylabel("specificity (pp)", color=S.MUTED, fontsize=11)
        ax.tick_params(labelsize=10)
        S.clean_axes(ax, keep="lb")
        S.hairline_grid(ax, axis="y")
        ax.legend(loc="upper right", frameon=False, fontsize=10.5,
                  labelcolor=S.INK, ncol=1)

        fig.savefig(out_path)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="experiments/intervene_out/results.json")
    args = p.parse_args()
    results_path = Path(args.results)
    results = json.loads(results_path.read_text())
    out_dir = results_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    fig_specificity_matrix(results, out_dir / "fig_specificity_matrix.png")
    fig_specificity_score(results, out_dir / "fig_specificity_score.png")
    print(f"wrote {out_dir}/fig_specificity_*.png")


if __name__ == "__main__":
    main()
