"""Generate the four paper figures from the trained JAX checkpoint.

Run on the Kaggle kernel after training completes. Loads
``ckpts/model.npz`` produced by ``painting-arithmetic-train --phase joint
--single-yat``, then uses ``painting_arithmetic.viz`` to compute the four
analyses (trunk PCA, operator-as-shift SVD, decoder unit atlas,
prototype winners) and writes the PNGs to ``./figs``.
"""

from __future__ import annotations

import os
import random
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from flax import nnx

sys.path.insert(0, "/kaggle/working/pa/src")
from painting_arithmetic import data, glyphs, viz  # noqa: E402
from painting_arithmetic.eval import load_checkpoint  # noqa: E402
from painting_arithmetic.model import YatArithmeticGen  # noqa: E402

CKPT = Path(os.environ.get("CKPT", "/kaggle/working/ckpts/model.npz"))
OUT = Path(os.environ.get("OUT", "/kaggle/working/figs"))
OUT.mkdir(exist_ok=True)
OP_COLORS = ["#D87F26", "#2E7D8A", "#8B3A62", "#5B7C3E"]


def load_model() -> YatArithmeticGen:
    rngs = nnx.Rngs(0)
    model = YatArithmeticGen(use_yat_encoder=False, single_yat=True, rngs=rngs)
    load_checkpoint(model, str(CKPT))
    print(f"loaded {CKPT}")
    return model


def load_mnist_bins() -> list[np.ndarray]:
    """Reuse the training pipeline's MNIST loading + binning by digit."""
    return data.load_mnist_binned("/kaggle/working/data", train=False)


# ---------------------------------------------------------------------------


def fig_trunk_pca(grid, out_path: Path) -> None:
    pca = viz.trunk_pca(grid)
    coords, vr = pca["coords"], pca["var_ratio"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    sc = axes[0].scatter(coords[:, 0], coords[:, 1], c=grid.results, cmap="turbo",
                         s=24, alpha=0.85, edgecolor="none")
    axes[0].set_title(f"trunk PCA — coloured by result\nPC1 = {vr[0]*100:.1f}%   PC2 = {vr[1]*100:.1f}%")
    axes[0].set_xlabel("PC1"); axes[0].set_ylabel("PC2")
    plt.colorbar(sc, ax=axes[0], label="result", shrink=0.85)
    for i in range(4):
        m = grid.ops == i
        axes[1].scatter(coords[m, 0], coords[m, 1], c=OP_COLORS[i], s=24, alpha=0.75,
                        edgecolor="none", label=glyphs.OP_GLYPHS[i])
    axes[1].set_title("same scatter, coloured by operator")
    axes[1].set_xlabel("PC1"); axes[1].set_ylabel("PC2")
    axes[1].legend(loc="best", frameon=True)
    for ax in axes: ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout(); plt.savefig(out_path, dpi=160, bbox_inches="tight", facecolor="white"); plt.close()
    print("wrote", out_path.name)


def fig_op_as_shift(grid, out_path: Path) -> None:
    shift = viz.op_as_shift(grid)
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    for ax, (lbl, info) in zip(axes.ravel(), shift.items()):
        ax.bar(range(1, 21), info["var_ratio"][:20], color="#2E7D8A", edgecolor="none")
        ax.set_title(f"{lbl}\ntop-1 = {info['top1']*100:.1f}%   top-3 = {info['top3']*100:.1f}%",
                     fontsize=10)
        ax.set_xlabel("singular rank"); ax.set_ylabel("variance ratio")
        ax.set_ylim(0, max(0.85, info["top1"] + 0.05))
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("operator-as-shift SVD spectrum (10×10 (a,b) grid)", y=1.01, fontsize=12)
    plt.tight_layout(); plt.savefig(out_path, dpi=160, bbox_inches="tight", facecolor="white"); plt.close()
    print("wrote", out_path.name)


def fig_decoder_unit_atlas(model, out_path: Path, alpha: float = 3.0, n_show: int = 24) -> None:
    atlas = viz.decoder_unit_atlas(model, alpha=alpha)
    foot, norms = atlas["footprints"], atlas["norms"]
    top = np.argsort(-norms)[:n_show]
    rows, cols = 4, 6
    fig, axes = plt.subplots(rows, cols, figsize=(13, 4.8))
    vmax = float(np.max(np.abs(foot[top])))
    for i, u in enumerate(top):
        ax = axes[i // cols, i % cols]
        ax.imshow(foot[u], cmap="seismic", vmin=-vmax, vmax=vmax, aspect="equal",
                  interpolation="bilinear")
        ax.set_title(f"unit {int(u)}   ‖Δ‖ = {norms[u]:.2f}", fontsize=9)
        ax.axis("off")
    fig.suptitle(f"decoder unit atlas — top {n_show} trunk units by L2 footprint\n"
                 f"red = adds bright pixels  ·  blue = subtracts (α = {alpha})",
                 fontsize=11, y=1.03)
    plt.tight_layout(); plt.savefig(out_path, dpi=160, bbox_inches="tight", facecolor="white"); plt.close()
    print("wrote", out_path.name)


def fig_prototype_gallery(model, bins, out_path: Path, n_show: int = 16) -> None:
    # Build a symbol library: encoder embeddings for digits + operator glyphs.
    lib_x, lib_lbl, lib_imgs = [], [], []
    rng_lib = random.Random(0)
    for d in range(10):
        for n in range(min(40, bins[d].shape[0])):
            lib_x.append(bins[d][n][None, None])
            lib_lbl.append(d)
            lib_imgs.append(bins[d][n])
    for op_idx in range(4):
        for k in range(20):
            if k < 5: pil = glyphs.render_glyph_clean(op_idx)
            else:     pil = glyphs.render_glyph_augmented(op_idx, rng_lib)
            arr = (np.asarray(pil, np.float32) / 255.0 - data.MNIST_MEAN) / data.MNIST_STD
            lib_x.append(arr[None, None])
            lib_lbl.append(glyphs.OP_LABEL_OFFSET + op_idx)
            lib_imgs.append(arr)
    lib_batch = jnp.concatenate([jnp.asarray(x) for x in lib_x], axis=0)
    @nnx.jit
    def encode(model_, x): return model_.encoder(x)
    lib_emb = np.asarray(encode(model, lib_batch))
    lib_labels = np.array(lib_lbl)
    lib_imgs = np.stack(lib_imgs, axis=0)

    win = viz.prototype_winners(model, lib_emb, lib_labels, k=15)
    # Operator-specialised = op-class winner in middle slot AND purity ≥ 0.5,
    # matching the paper's §6 tagging criterion.
    PURITY_THRESHOLD = 0.5
    is_op_winner = win["win_op"] >= glyphs.OP_LABEL_OFFSET
    is_pure = win["pur_op"] >= PURITY_THRESHOLD
    is_op = is_op_winner & is_pure
    n_op = int(is_op.sum())
    n_op_unfiltered = int(is_op_winner.sum())
    print(f"  {n_op} / {len(win['win_op'])} units have op-class winner AT purity ≥ {PURITY_THRESHOLD}")
    print(f"  (raw op-winners ignoring purity threshold: {n_op_unfiltered})")

    # Rank op-specialised units by op-slot purity.
    op_idxs = np.where(is_op)[0]
    op_purs = win["pur_op"][op_idxs]
    top_op = op_idxs[np.argsort(-op_purs)[:n_show]]

    # For each top unit, find which library image best matches each slot.
    # We use the indices stored implicitly by prototype_winners — recompute
    # the argmax per slot for display.
    from painting_arithmetic.viz import _yat_score
    state = nnx.state(model.h1, nnx.Param)
    flat = dict(nnx.to_flat_state(state))
    W = None
    for k_path, v in flat.items():
        arr = np.asarray(v.value)
        if arr.ndim == 2:
            W = arr
            break
    if W.shape[1] != lib_emb.shape[-1] * 3:
        W = W.T
    W_a, W_op, W_b = W[:, :64], W[:, 64:128], W[:, 128:]
    sc_a = _yat_score(lib_emb, W_a); idx_a = sc_a.argmax(axis=0)
    sc_op = _yat_score(lib_emb, W_op); idx_op = sc_op.argmax(axis=0)
    sc_b = _yat_score(lib_emb, W_b); idx_b = sc_b.argmax(axis=0)

    def _name(idx):
        l = lib_labels[idx]
        return glyphs.OP_GLYPHS[l - glyphs.OP_LABEL_OFFSET] if l >= glyphs.OP_LABEL_OFFSET else str(l)

    fig, axes = plt.subplots(4, n_show, figsize=(1.0 * n_show, 5.0),
                             gridspec_kw={"height_ratios": [1, 1, 1, 0.35]})
    for j, u in enumerate(top_op):
        for r, slot, winner_idx in [(0, "a", idx_a[u]),
                                    (1, "op", idx_op[u]),
                                    (2, "b", idx_b[u])]:
            ax = axes[r, j]
            ax.imshow(lib_imgs[winner_idx], cmap="gray", vmin=-2, vmax=2)
            ax.axis("off")
        ax = axes[3, j]
        ax.text(0.5, 0.8, f"u {u}", ha="center", va="top", fontsize=8, family="monospace")
        ax.text(0.5, 0.25, f"({_name(idx_a[u])}, {_name(idx_op[u])}, {_name(idx_b[u])})",
                ha="center", va="top", fontsize=9, color="#3A3A3A")
        ax.axis("off")
    for r, lbl_text in [(0, "slot a"), (1, "slot op"), (2, "slot b")]:
        axes[r, 0].text(-0.18, 0.5, lbl_text, transform=axes[r, 0].transAxes,
                        ha="right", va="center", fontsize=10, family="monospace",
                        color="#3A3A3A")
    fig.suptitle(f"prototype gallery — top {n_show} operator-specialised trunk units\n"
                 f"({n_op} / {len(win['win_op'])} units total have an operator-class winner in the op slot)",
                 fontsize=11, y=1.005)
    plt.tight_layout(); plt.savefig(out_path, dpi=160, bbox_inches="tight", facecolor="white"); plt.close()
    print("wrote", out_path.name)


def main() -> None:
    print("loading model + MNIST test bins…")
    model = load_model()
    bins = load_mnist_bins()
    print("collecting trunk grid (390 points)…")
    grid = viz.collect_trunk_grid(model, bins)
    print("trunks:", grid.trunks.shape, "results range:", grid.results.min(), "..", grid.results.max())

    fig_trunk_pca(grid, OUT / "trunk_pca_v3.png")
    fig_op_as_shift(grid, OUT / "op_as_shift.png")
    fig_decoder_unit_atlas(model, OUT / "decoder_unit_atlas.png")
    fig_prototype_gallery(model, bins, OUT / "prototype_gallery_v3.png")
    print("done.")


if __name__ == "__main__":
    main()
