"""A3: Prototype Gram-matrix spectrum.

For the v3-thin h₁ layer, treat each unit's 192-d weight row as a
prototype vector and compute three Gram matrices over the 256 prototypes:

  - cosine similarity
  - the Yat kernel: $k(w_i, w_j) = (w_i\cdot w_j)^2 / (\|w_i-w_j\|^2 + \varepsilon)$
  - middle-slot-only cosine (rows restricted to indices 64..128)

We eigendecompose each, report:
  - the spectrum
  - the "effective rank" $\exp(H(\hat\lambda))$ where $\hat\lambda_k = \lambda_k / \sum\lambda$
  - block structure when units are permuted by their operator tag

Success criterion: effective rank meaningfully below 256; visible block
structure under the op-permutation.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from _common import OUT_BASE, S, get_h1_kernel, load_thin, operator_tags

OUT = OUT_BASE / "A3_gram_out"
OUT.mkdir(parents=True, exist_ok=True)


def effective_rank(eigs: np.ndarray) -> float:
    eigs = np.clip(eigs, 0, None)
    s = eigs.sum()
    if s <= 0:
        return 0.0
    p = eigs / s
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def yat_gram(W: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    dot = W @ W.T
    sq = (W * W).sum(-1)
    dist = sq[:, None] + sq[None, :] - 2 * dot
    return (dot ** 2) / (dist + eps)


def main():
    print("loading v3-thin …")
    m = load_thin()
    W = get_h1_kernel(m)                       # (256, 192) — rows = units
    W_op = W[:, 64:128]                        # operator slot
    tags = operator_tags(m)
    op_of_unit = tags["op_of_unit"]

    # Cosine Grams.
    Wn = W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-9)
    K_cos = Wn @ Wn.T
    Won = W_op / (np.linalg.norm(W_op, axis=1, keepdims=True) + 1e-9)
    K_opcos = Won @ Won.T
    K_yat = yat_gram(W)

    diag = {}
    for name, K in {"cos": K_cos, "op-slot cos": K_opcos, "yat": K_yat}.items():
        # Symmetrise (Yat is by construction symmetric, but float drift).
        K = 0.5 * (K + K.T)
        eigs = np.linalg.eigvalsh(K)[::-1]
        eigs = np.clip(eigs, 0, None)
        er = effective_rank(eigs)
        diag[name] = {"eigs": eigs.tolist(), "effective_rank": er,
                      "top5_frac": float(eigs[:5].sum() / eigs.sum())}
        print(f"  {name:>14}:  effective rank = {er:6.1f} / 256   top-5 frac = {diag[name]['top5_frac']:.3f}")

    # Permute by op tag so blocks are visible.
    order = np.argsort(op_of_unit)
    perm_tags = op_of_unit[order]

    # ----- figure -----
    with S.themed():
        fig = plt.figure(figsize=(14, 8))
        gs = fig.add_gridspec(2, 3, left=0.07, right=0.96, bottom=0.10, top=0.78,
                              wspace=0.30, hspace=0.55)
        S.add_title_block(fig, S.TitleBlock(
            title="The 256 prototypes are not independent",
            subtitle="Three Gram matrices over h₁'s prototype rows; spectrum on top, op-sorted matrix on bottom",
            caption=(f"effective rank: cos {diag['cos']['effective_rank']:.0f},"
                     f" op-slot cos {diag['op-slot cos']['effective_rank']:.0f},"
                     f" yat {diag['yat']['effective_rank']:.0f}"
                     f"  ·  full rank would be 256"),
        ), top=0.94)

        names = ["cos", "op-slot cos", "yat"]
        for k, name in enumerate(names):
            ax = fig.add_subplot(gs[0, k])
            eigs = np.array(diag[name]["eigs"])
            cum = np.cumsum(eigs) / eigs.sum()
            ax.plot(np.arange(1, len(eigs) + 1), cum, color=S.ACCENT2, lw=1.8)
            ax.axhline(0.9, color=S.MUTED, linestyle=":", lw=0.8)
            ax.set_xlim(0, 256)
            ax.set_ylim(0, 1.02)
            ax.set_title(f"{name}: cumulative variance", color=S.INK, fontsize=12,
                         fontweight="bold", loc="left", pad=8)
            ax.set_xlabel("rank", color=S.MUTED, fontsize=10)
            S.clean_axes(ax, keep="lb")
            S.hairline_grid(ax, axis="y")
            # Annotate effective rank.
            er = diag[name]["effective_rank"]
            ax.text(0.97, 0.08, f"eff rank ≈ {er:.0f}",
                    transform=ax.transAxes, ha="right", va="bottom",
                    color=S.ACCENT, fontsize=11, fontweight="bold")

        for k, (name, K) in enumerate(zip(names, [K_cos, K_opcos, K_yat])):
            ax = fig.add_subplot(gs[1, k])
            Kp = K[order][:, order]
            vmax = float(np.percentile(np.abs(Kp), 99))
            ax.imshow(Kp, cmap=S.SEISMIC_CMAP, vmin=-vmax, vmax=vmax,
                      interpolation="nearest", aspect="equal")
            # Tag boundaries: where op_of_unit changes.
            boundaries = [0]
            for i in range(1, len(perm_tags)):
                if perm_tags[i] != perm_tags[i - 1]:
                    boundaries.append(i)
            boundaries.append(len(perm_tags))
            for bi in boundaries[1:-1]:
                ax.axhline(bi - 0.5, color=S.ACCENT, lw=0.7, alpha=0.5)
                ax.axvline(bi - 0.5, color=S.ACCENT, lw=0.7, alpha=0.5)
            ax.set_title(f"{name} (sorted by op-tag)", color=S.INK, fontsize=11.5,
                         fontweight="bold", loc="left", pad=8)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)

        fig.savefig(OUT / "fig_gram.png")
        plt.close(fig)

    (OUT / "results.json").write_text(json.dumps(
        {name: {"effective_rank": d["effective_rank"], "top5_frac": d["top5_frac"]}
         for name, d in diag.items()}, indent=2))
    print(f"\nwrote {OUT}")

    # ---- success judgement ----
    er_threshold = 200  # 256 would be full rank; <200 means meaningful compression
    block_likely = any(diag[n]["effective_rank"] < er_threshold for n in names)
    print(f"\nverdict: effective rank below {er_threshold}? {block_likely}")
    return diag


if __name__ == "__main__":
    main()
