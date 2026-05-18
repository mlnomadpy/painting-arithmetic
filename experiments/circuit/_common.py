"""Shared infrastructure for circuit-mapping experiments.

Loads the v3-thin checkpoint and pre-computes the canonical grid
(every $(a, op, b)$ with $0\le a, b \le 9$, dropping $b=0$ for //) once.
Each downstream experiment imports :func:`grid_cache` and works from
that single batch.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

# Ensure the painting_arithmetic package is importable.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, "/Users/tahabsn/conductor/workspaces/nmn/islamabad/src")
# viz_style for figures
sys.path.insert(0, str(ROOT.parent / "visual_arithmetic"))

import viz_style as S
from painting_arithmetic import data, glyphs
from painting_arithmetic.eval import load_checkpoint
from painting_arithmetic.model import YatArithmeticGen

CKPT = ROOT / "ckpts_thin" / "model.npz"
DATA_ROOT = str(ROOT / "data")
OUT_BASE = ROOT / "experiments" / "circuit"


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


def load_thin() -> YatArithmeticGen:
    m = YatArithmeticGen(use_yat_encoder=False, single_yat=True, rngs=nnx.Rngs(0))
    load_checkpoint(m, str(CKPT))
    return m


# ---------------------------------------------------------------------------
# Canonical grid: one MNIST exemplar per digit, four canonical operators
# ---------------------------------------------------------------------------


@dataclass
class Grid:
    """Pre-computed canonical (a, op, b) grid.

    Attributes
    ----------
    img_a, img_op, img_b : (N, 1, 28, 28) float32 — MNIST-normalised
    a, b, op, r : (N,) int32 — labels (op in 0..3, r in [-9, 81])
    e_a, e_op, e_b : (N, 64) — encoder outputs
    t : (N, 256) — trunk vector
    img_out : (N, 1, 28, 84) — painted answer
    op_e_clean : (4, 64) — encoded canonical operator glyphs (one per op)
    """

    img_a: np.ndarray; img_op: np.ndarray; img_b: np.ndarray
    a: np.ndarray; b: np.ndarray; op: np.ndarray; r: np.ndarray
    e_a: np.ndarray; e_op: np.ndarray; e_b: np.ndarray
    t: np.ndarray
    img_out: np.ndarray
    op_e_clean: np.ndarray


def build_grid(model: YatArithmeticGen, *, n_per_digit: int = 1, seed: int = 0) -> Grid:
    """Build the canonical grid. ``n_per_digit`` controls how many MNIST
    exemplars are paired with each digit slot (default 1 = single exemplar
    per digit, deterministic)."""
    bins = data.load_mnist_binned(DATA_ROOT, train=False)
    rng = np.random.default_rng(seed)
    by_d = {d: bins[d][rng.choice(bins[d].shape[0], size=n_per_digit, replace=False)] for d in range(10)}

    # Pre-render the 4 canonical operator glyphs.
    op_arrs = []
    for op_idx in range(4):
        pil = glyphs.render_glyph_clean(op_idx)
        arr = (np.asarray(pil, np.float32) / 255.0 - data.MNIST_MEAN) / data.MNIST_STD
        op_arrs.append(arr)
    op_arrs = np.stack(op_arrs)  # (4, 28, 28)

    ias, ios, ibs, a_lst, b_lst, op_lst, r_lst = [], [], [], [], [], [], []
    for a in range(10):
        for b in range(10):
            for op in range(4):
                if glyphs.OPS[op] == "//" and b == 0:
                    continue
                for k in range(n_per_digit):
                    ias.append(by_d[a][k])
                    ios.append(op_arrs[op])
                    ibs.append(by_d[b][k])
                    a_lst.append(a); b_lst.append(b); op_lst.append(op)
                    if   glyphs.OPS[op] == "+":  r = a + b
                    elif glyphs.OPS[op] == "-":  r = a - b
                    elif glyphs.OPS[op] == "*":  r = a * b
                    else:                         r = a // b
                    r_lst.append(r)

    img_a = np.stack(ias)[:, None].astype(np.float32)
    img_op = np.stack(ios)[:, None].astype(np.float32)
    img_b = np.stack(ibs)[:, None].astype(np.float32)

    # Forward pass.
    @nnx.jit
    def fwd(m, ia, io, ib):
        ea = m.encoder(ia); eo = m.encoder(io); eb = m.encoder(ib)
        h = jnp.concatenate([ea, eo, eb], axis=-1)
        t = m.h1(h)
        # In v3-thin, t2 = t1; decoder reads t1.
        img = m.decoder(t)
        return ea, eo, eb, t, img

    e_a, e_op, e_b, t, img_out = fwd(model, jnp.asarray(img_a), jnp.asarray(img_op), jnp.asarray(img_b))

    # Clean per-operator embeddings (no batch).
    op_e_clean = np.asarray(model.encoder(jnp.asarray(op_arrs)[:, None].astype(np.float32)))

    return Grid(
        img_a=img_a, img_op=img_op, img_b=img_b,
        a=np.array(a_lst, dtype=np.int32),
        b=np.array(b_lst, dtype=np.int32),
        op=np.array(op_lst, dtype=np.int32),
        r=np.array(r_lst, dtype=np.int32),
        e_a=np.asarray(e_a), e_op=np.asarray(e_op), e_b=np.asarray(e_b),
        t=np.asarray(t),
        img_out=np.asarray(img_out),
        op_e_clean=op_e_clean,
    )


def get_h1_kernel(model: YatArithmeticGen) -> np.ndarray:
    """Return the (256, 192) h1 kernel with rows = unit prototypes."""
    state = nnx.state(model.h1, nnx.Param)
    flat = dict(nnx.to_flat_state(state))
    for k, v in flat.items():
        arr = np.asarray(v[...])
        if arr.ndim == 2:
            # NNX stores (in, out); transpose so rows are units.
            return arr.T if arr.shape[1] == 192 else arr.T  # (out, in) form
    raise RuntimeError("no 2-D weight in h1")


# ---------------------------------------------------------------------------
# Per-unit operator tagging (prototype winners on a library of glyphs)
# ---------------------------------------------------------------------------


def operator_tags(model: YatArithmeticGen, *, purity_threshold: float = 0.5, seed: int = 0) -> dict:
    """Return per-unit operator tag based on the prototype-gallery scheme."""
    from painting_arithmetic.viz import prototype_winners
    import random as _r
    rng = _r.Random(seed)

    bins = data.load_mnist_binned(DATA_ROOT, train=False)
    tensors, labels = [], []
    for d in range(10):
        idx = rng.sample(range(bins[d].shape[0]), 120)
        for i in idx:
            tensors.append(bins[d][i]); labels.append(d)
    for op in range(4):
        for k in range(60):
            pil = (glyphs.render_glyph_clean(op) if k < 15
                   else glyphs.render_glyph_augmented(op, rng))
            arr = (np.asarray(pil, np.float32) / 255.0 - data.MNIST_MEAN) / data.MNIST_STD
            tensors.append(arr); labels.append(glyphs.OP_LABEL_OFFSET + op)
    stack = np.stack(tensors)[:, None].astype(np.float32)
    emb = np.asarray(model.encoder(jnp.asarray(stack)))
    winners = prototype_winners(model, emb, np.array(labels), k=15)
    win_op = np.asarray(winners["win_op"])
    pur_op = np.asarray(winners["pur_op"])
    is_op = (win_op >= glyphs.OP_LABEL_OFFSET) & (pur_op >= purity_threshold)
    return {
        "win_op": win_op,
        "pur_op": pur_op,
        "op_of_unit": np.where(is_op, win_op - glyphs.OP_LABEL_OFFSET, -1),
    }


__all__ = ["Grid", "load_thin", "build_grid", "get_h1_kernel", "operator_tags", "OUT_BASE", "S"]
