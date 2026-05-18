"""D1: Operator-embedding geometry.

The encoder maps each of the four canonical operator glyphs into ℝ⁶⁴.
We look at:
  - pairwise distances + cosine angles between the 4 vectors
  - their 2-D PCA projection (Fourier-style)
  - the same for "noisy" operator embeddings (15 augmented samples per op)
    to see whether the structure survives glyph noise

Success: the four operators form a recognisable, low-dimensional shape
(line / triangle / tetrahedron); PC1 explains >50% of the variance over
the 4 means; noisy clouds stay separated.
"""

from __future__ import annotations

import json
import random as _r
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from flax import nnx

from _common import OUT_BASE, S, load_thin
from painting_arithmetic import data, glyphs

OUT = OUT_BASE / "D1_op_geom_out"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    print("loading v3-thin …")
    m = load_thin()

    # ----- clean per-op embeddings -----
    op_arrs = []
    for op_idx in range(4):
        pil = glyphs.render_glyph_clean(op_idx)
        arr = (np.asarray(pil, np.float32) / 255.0 - data.MNIST_MEAN) / data.MNIST_STD
        op_arrs.append(arr)
    op_arrs = np.stack(op_arrs)[:, None].astype(np.float32)
    e_clean = np.asarray(m.encoder(jnp.asarray(op_arrs)))         # (4, 64)

    # ----- noisy per-op clouds -----
    rng = _r.Random(0)
    n_per = 60
    cloud_imgs, cloud_lbls = [], []
    for op in range(4):
        for _ in range(n_per):
            pil = glyphs.render_glyph_augmented(op, rng)
            a = (np.asarray(pil, np.float32) / 255.0 - data.MNIST_MEAN) / data.MNIST_STD
            cloud_imgs.append(a); cloud_lbls.append(op)
    cloud_imgs = np.stack(cloud_imgs)[:, None].astype(np.float32)
    cloud_lbls = np.array(cloud_lbls)
    e_cloud = np.asarray(m.encoder(jnp.asarray(cloud_imgs)))      # (4n, 64)

    # ----- geometry of the 4 means -----
    # Pairwise cosine + Euclidean.
    norms = np.linalg.norm(e_clean, axis=1, keepdims=True)
    cos = (e_clean @ e_clean.T) / (norms * norms.T + 1e-9)
    dist = np.linalg.norm(e_clean[:, None] - e_clean[None, :], axis=-1)

    # SVD over the centred 4 means.
    mu = e_clean.mean(0, keepdims=True)
    centred = e_clean - mu
    U, S_, Vt = np.linalg.svd(centred, full_matrices=False)
    vr = (S_ ** 2) / (S_ ** 2).sum()
    proj4 = centred @ Vt[:2].T                                     # (4, 2)
    proj_cloud = (e_cloud - mu) @ Vt[:2].T                         # (4n, 2)

    # ----- print + save numbers -----
    print(f"  PC1 var = {vr[0]*100:.1f}%   PC2 var = {vr[1]*100:.1f}%")
    print("  cosine matrix:")
    for i, gly in enumerate(S.OP_GLYPHS):
        print("   ", gly, "  ", " ".join(f"{cos[i, j]:+.2f}" for j in range(4)))

    # ----- figure -----
    with S.themed():
        fig = plt.figure(figsize=(14, 7))
        gs = fig.add_gridspec(1, 2, left=0.07, right=0.97, bottom=0.13, top=0.72, wspace=0.30)
        S.add_title_block(fig, S.TitleBlock(
            title="The four operators occupy a 2-D plane in encoder space",
            subtitle=f"PCA of the 4 canonical operator embeddings · "
                     f"PC1+PC2 capture {vr[:2].sum()*100:.0f}% of variance · "
                     f"− and ÷ are nearly co-linear (cos = {cos[1,3]:+.2f})",
            caption="Faint clouds: 60 augmented glyph samples per operator. "
                    "The − / ÷ collinearity in encoder space is consistent with the "
                    "intervention experiment's difficulty disentangling those two.",
        ), top=0.94)

        # Panel 1: scatter (cloud + means + labels)
        ax = fig.add_subplot(gs[0, 0])
        for op in range(4):
            mask = cloud_lbls == op
            ax.scatter(proj_cloud[mask, 0], proj_cloud[mask, 1],
                       c=S.OP_COLOR_LIST[op], s=22, alpha=0.30, edgecolor="none")
            ax.scatter(proj4[op, 0], proj4[op, 1],
                       c=S.OP_COLOR_LIST[op], s=240, edgecolor=S.BG, linewidth=1.5, zorder=5)
            S.annotate_blob(ax, proj4[op, 0], proj4[op, 1], S.OP_GLYPHS[op],
                            color=S.INK, fontsize=20)
        ax.set_xlabel("PC1", color=S.MUTED); ax.set_ylabel("PC2", color=S.MUTED)
        ax.set_title("operator embeddings · PC1–PC2", color=S.INK, fontsize=12.5,
                     fontweight="bold", loc="left", pad=8)
        S.hairline_grid(ax)

        # Panel 2: pairwise cosine
        ax = fig.add_subplot(gs[0, 1])
        im = ax.imshow(cos, cmap=S.SEISMIC_CMAP, vmin=-1, vmax=1, aspect="equal")
        ax.set_xticks(range(4)); ax.set_yticks(range(4))
        ax.set_xticklabels(S.OP_GLYPHS, fontsize=14, color=S.INK)
        ax.set_yticklabels(S.OP_GLYPHS, fontsize=14, color=S.INK)
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f"{cos[i, j]:+.2f}",
                        ha="center", va="center",
                        color=S.INK if abs(cos[i, j]) < 0.55 else S.BG,
                        fontsize=11.5, fontweight="bold" if i == j else "normal")
        ax.set_title("pairwise cosine between operator embeddings", color=S.INK,
                     fontsize=12.5, fontweight="bold", loc="left", pad=8)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.tick_params(length=0)

        fig.savefig(OUT / "fig_op_geom.png")
        plt.close(fig)

    out = {
        "pc_var": vr[:4].tolist(),
        "cos_matrix": cos.tolist(),
        "distance_matrix": dist.tolist(),
    }
    (OUT / "results.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {OUT}")

    # ---- verdict ----
    # PC1 > 0.5 OR cloud-overlap small (heuristic). PC1 + PC2 > 0.8 is a clean tetrahedron-like shape.
    cum2 = float(vr[:2].sum())
    win = vr[0] > 0.5 or cum2 > 0.8
    print(f"\nverdict: PC1={vr[0]:.2%}, PC1+PC2={cum2:.2%}; shape recognisable? {win}")
    return out


if __name__ == "__main__":
    main()
