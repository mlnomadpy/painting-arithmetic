"""E3: Trunk-vector distance between integer answers.

For each integer result $r \in [-9, 81]$ we average the trunk vector
$t(a, op, b)$ over every $(a, op, b)$ on the canonical grid that yields
$r$. The pairwise Euclidean-distance matrix over these "answer
centroids" is the trunk's intrinsic metric on the integer-answer set.

Success: visible band structure — nearest neighbours of $r$ are
$r \pm 1$ in many rows, or at least a monotone metric on the integers.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from _common import OUT_BASE, S, build_grid, load_thin
from painting_arithmetic import glyphs

OUT = OUT_BASE / "E3_paint_distance_out"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    print("loading v3-thin …")
    m = load_thin()
    g = build_grid(m)

    # Average trunk vector per integer result.
    results = sorted(np.unique(g.r).tolist())
    centroids = np.zeros((len(results), g.t.shape[1]), dtype=np.float32)
    counts = []
    for k, r in enumerate(results):
        mask = g.r == r
        counts.append(int(mask.sum()))
        if mask.sum() > 0:
            centroids[k] = g.t[mask].mean(0)
    print(f"  {len(results)} distinct results, {sum(counts)} grid points")

    # Pairwise distances.
    D = np.linalg.norm(centroids[:, None] - centroids[None, :], axis=-1)
    # For each r, find the nearest neighbour (excluding self).
    nn = []
    for k, r in enumerate(results):
        row = D[k].copy(); row[k] = np.inf
        j = int(np.argmin(row))
        nn.append((r, results[j], float(row[j])))

    consecutive_hits = sum(1 for (r, r_nn, _) in nn if abs(r - r_nn) == 1)
    pct = consecutive_hits / len(nn) * 100
    print(f"  nearest-neighbour is r±1 for {consecutive_hits}/{len(nn)} integers ({pct:.0f}%)")

    # Spearman-style rank correlation: does numerical distance |r-r'| correlate
    # with trunk distance?
    pairs = [(abs(results[i] - results[j]), D[i, j])
             for i in range(len(results)) for j in range(i + 1, len(results))]
    p_num = np.array([p[0] for p in pairs], dtype=np.float64)
    p_tr  = np.array([p[1] for p in pairs], dtype=np.float64)
    # Rank corr.
    from scipy.stats import spearmanr
    rho, _ = spearmanr(p_num, p_tr)
    print(f"  Spearman ρ(|r-r'|, ‖t_r - t_r'‖) = {rho:.3f}")

    # ----- figure -----
    with S.themed():
        fig = plt.figure(figsize=(14, 6.8))
        gs = fig.add_gridspec(1, 2, left=0.07, right=0.97, bottom=0.12, top=0.74,
                              wspace=0.30, width_ratios=[1.2, 1.0])
        S.add_title_block(fig, S.TitleBlock(
            title="The trunk knows arithmetic by distance",
            subtitle=f"Pairwise trunk-vector distance between integer answers · "
                     f"Spearman ρ between numerical and trunk distance: {rho:.3f}",
            caption=f"{consecutive_hits}/{len(nn)} integers ({pct:.0f}%) have their numerical "
                    f"neighbour r±1 as the nearest centroid in trunk space",
        ), top=0.92)

        # Distance matrix.
        ax = fig.add_subplot(gs[0, 0])
        im = ax.imshow(D, cmap=S.VALUE_CMAP, aspect="equal",
                       extent=[results[0] - 0.5, results[-1] + 0.5,
                               results[-1] + 0.5, results[0] - 0.5])
        ax.set_xlabel("answer r'", color=S.MUTED, fontsize=10)
        ax.set_ylabel("answer r",  color=S.MUTED, fontsize=10)
        ax.set_title("pairwise centroid distance", color=S.INK, fontsize=11.5,
                     fontweight="bold", loc="left", pad=6)
        for sp in ax.spines.values():
            sp.set_visible(False)
        S.colorbar(ax, im, label="‖t_r − t_r'‖")

        # Scatter: numerical vs trunk distance.
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.scatter(p_num, p_tr, s=6, color=S.ACCENT2, alpha=0.45, edgecolor="none")
        ax2.set_xlabel("|r − r'|", color=S.MUTED, fontsize=10)
        ax2.set_ylabel("trunk distance",  color=S.MUTED, fontsize=10)
        ax2.set_title("numerical vs trunk distance", color=S.INK, fontsize=11.5,
                      fontweight="bold", loc="left", pad=6)
        S.clean_axes(ax2, keep="lb")
        S.hairline_grid(ax2)

        fig.savefig(OUT / "fig_paint_distance.png")
        plt.close(fig)

    out = {
        "results": results,
        "n_points_per_result": counts,
        "consecutive_nn_pct": pct,
        "spearman_rho": float(rho),
    }
    (OUT / "results.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {OUT}")

    win = rho > 0.5
    print(f"\nverdict: Spearman ρ > 0.5? {win}")
    return out


if __name__ == "__main__":
    main()
