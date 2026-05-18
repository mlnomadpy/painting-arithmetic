"""Interpretability views — trunk PCA, operator-as-shift SVD, decoder unit
atlas, prototype gallery, layer trajectories.

Each function takes a trained ``YatArithmeticGen`` model + a binned MNIST
test set and returns numpy arrays the caller can plot. No matplotlib
dependency at the analysis layer; callers compose their own figures.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from . import glyphs
from .model import YatArithmeticGen


# ---------------------------------------------------------------------------
# Trunk grid — every (a, op, b) on the 10×10×4 grid
# ---------------------------------------------------------------------------


@dataclass
class TrunkGrid:
    trunks: np.ndarray   # (4, 10, 10, hidden) but valid mask for // when b==0
    results: np.ndarray  # (4, 10, 10) signed integer truth
    ops: np.ndarray      # (4, 10, 10) op id
    a_idx: np.ndarray    # (4, 10, 10)
    b_idx: np.ndarray    # (4, 10, 10)


def collect_trunk_grid(model: YatArithmeticGen, bins_test: list[np.ndarray]) -> TrunkGrid:
    """For every (a, op, b) on the canonical grid, compute the post-h2 trunk
    vector (using one representative MNIST digit per class and one canonical
    operator glyph)."""
    by_digit = [bins_test[d][0][None, None, :, :] for d in range(10)]  # (1, 1, 28, 28)
    op_imgs = []
    for i in range(4):
        op_arr = (np.asarray(glyphs.render_glyph_clean(i), dtype=np.float32) / 255.0)
        op_arr = (op_arr - 0.1307) / 0.3081
        op_imgs.append(op_arr[None, None, :, :])

    # Build the full grid as a single batch — 4*10*10 entries (// b=0 dropped post-hoc).
    ias, ios, ibs, ops, results, a_idx, b_idx = [], [], [], [], [], [], []
    for op in range(4):
        for a in range(10):
            for b in range(10):
                if glyphs.OPS[op] == "//" and b == 0:
                    continue
                ias.append(by_digit[a])
                ios.append(op_imgs[op])
                ibs.append(by_digit[b])
                ops.append(op)
                a_idx.append(a); b_idx.append(b)
                if   glyphs.OPS[op] == "+": results.append(a + b)
                elif glyphs.OPS[op] == "-": results.append(a - b)
                elif glyphs.OPS[op] == "*": results.append(a * b)
                else:                       results.append(a // b)
    ia = jnp.concatenate([jnp.asarray(x) for x in ias], axis=0)
    io = jnp.concatenate([jnp.asarray(x) for x in ios], axis=0)
    ib = jnp.concatenate([jnp.asarray(x) for x in ibs], axis=0)

    @nnx.jit
    def get_t2(model_, ia, io, ib):
        _, _, _, _, _, t = model_.trunk(ia, io, ib)
        return t

    t2 = np.asarray(get_t2(model, ia, io, ib))
    return TrunkGrid(
        trunks=t2,
        results=np.array(results),
        ops=np.array(ops),
        a_idx=np.array(a_idx),
        b_idx=np.array(b_idx),
    )


# ---------------------------------------------------------------------------
# Trunk PCA — value line + operator regions
# ---------------------------------------------------------------------------


def trunk_pca(grid: TrunkGrid) -> dict:
    Xc = grid.trunks - grid.trunks.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    coords = Xc @ Vt[:2].T
    var_ratio = (S**2 / (S**2).sum())[:2]
    return {"coords": coords, "var_ratio": var_ratio, "basis": Vt[:2]}


# ---------------------------------------------------------------------------
# Operator-as-shift SVD — is each operator transition one direction?
# ---------------------------------------------------------------------------


def op_as_shift(grid: TrunkGrid) -> dict:
    """For every pair of operators, SVD the per-(a, b) trunk differences."""
    by_op = {i: grid.trunks[grid.ops == i] for i in range(4)}
    pairs = [(0, 1, "+ − −"), (0, 2, "+ − ×"), (0, 3, "+ − ÷"),
             (1, 2, "− − ×"), (1, 3, "− − ÷"), (2, 3, "× − ÷")]
    out = {}
    for i, j, label in pairs:
        n = min(by_op[i].shape[0], by_op[j].shape[0])
        diff = by_op[i][:n] - by_op[j][:n]
        U, S, Vt = np.linalg.svd(diff, full_matrices=False)
        vr = (S ** 2) / (S ** 2).sum()
        out[label] = {"var_ratio": vr, "top1": float(vr[0]), "top3": float(vr[:3].sum()),
                       "direction": Vt[0]}
    return out


# ---------------------------------------------------------------------------
# Decoder unit atlas — what each trunk unit paints
# ---------------------------------------------------------------------------


def decoder_unit_atlas(model: YatArithmeticGen, alpha: float = 3.0) -> dict:
    """For each of the H trunk units, compute ``decoder(α·e_u) - decoder(0)``."""
    # Probe the decoder via a dummy forward to discover the trunk dim.
    test = jnp.zeros((1, 1, 28, 28))
    t = model.trunk(test, test, test)[-1]   # (1, H)
    H = t.shape[-1]

    @nnx.jit
    def decode(model_, vec):
        return model_.decoder(vec)

    baseline = np.asarray(decode(model, jnp.zeros((1, H))))[0, 0]
    footprints = np.zeros((H, 28, 84), dtype=np.float32)
    chunk = 64
    for s in range(0, H, chunk):
        e = s + min(chunk, H - s)
        v = np.zeros((e - s, H), dtype=np.float32)
        for i in range(e - s):
            v[i, s + i] = alpha
        img = np.asarray(decode(model, jnp.asarray(v)))[:, 0]   # (chunk, 28, 84)
        footprints[s:e] = img - baseline
    norms = np.sqrt((footprints ** 2).sum(axis=(1, 2)))
    return {"footprints": footprints, "baseline": baseline, "norms": norms}


# ---------------------------------------------------------------------------
# Yat prototype matching — closest library symbol per slot
# ---------------------------------------------------------------------------


def _yat_score(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Replicate the Yat numerator/denominator scoring used by the trunk."""
    dot = x @ w.T
    x_sq = (x ** 2).sum(axis=-1, keepdims=True)
    w_sq = (w ** 2).sum(axis=-1)[None, :]
    dist = x_sq + w_sq - 2 * dot
    return (dot ** 2) / (dist + 1e-3)


def prototype_winners(
    model: YatArithmeticGen,
    lib_emb: np.ndarray,
    lib_labels: np.ndarray,
    k: int = 15,
) -> dict:
    """For each h1 unit, identify the symbol the unit's prototype prefers in
    each of the three input slots (a, op, b)."""
    state = nnx.state(model.h1, nnx.Param)
    flat = dict(nnx.to_flat_state(state))
    # The YatNMN weight is stored under ('kernel',) in NNX; use the first
    # match that's 2-D.
    W = None
    for k_path, v in flat.items():
        arr = np.asarray(v.value)
        if arr.ndim == 2:
            W = arr
            break
    if W is None:
        raise RuntimeError("could not find h1 kernel weight")
    # NNX stores (in, out); transpose so each row is a unit's prototype.
    if W.shape[1] != lib_emb.shape[-1] * 3:
        W = W.T
    assert W.shape[1] == lib_emb.shape[-1] * 3, (
        f"unexpected h1 kernel shape {W.shape}; expected last dim {lib_emb.shape[-1] * 3}"
    )
    H = W.shape[0]

    W_a, W_op, W_b = W[:, :64], W[:, 64:128], W[:, 128:]
    sc_a = _yat_score(lib_emb, W_a)
    sc_op = _yat_score(lib_emb, W_op)
    sc_b = _yat_score(lib_emb, W_b)

    def winners(scores):
        top_idx = np.argsort(scores, axis=0)[-k:]
        win = np.zeros(H, dtype=int); pur = np.zeros(H)
        for u in range(H):
            counts = np.bincount(lib_labels[top_idx[:, u]], minlength=glyphs.NUM_SYMBOLS)
            win[u] = int(counts.argmax()); pur[u] = counts.max() / k
        return win, pur

    win_a, pur_a = winners(sc_a)
    win_op, pur_op = winners(sc_op)
    win_b, pur_b = winners(sc_b)
    return {
        "win_a": win_a, "win_op": win_op, "win_b": win_b,
        "pur_a": pur_a, "pur_op": pur_op, "pur_b": pur_b,
        "n_op_specialised": int((win_op >= glyphs.OP_LABEL_OFFSET).sum()),
    }


__all__ = [
    "TrunkGrid",
    "collect_trunk_grid",
    "trunk_pca",
    "op_as_shift",
    "decoder_unit_atlas",
    "prototype_winners",
]
