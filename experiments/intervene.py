"""Operator-ablation interventions: weight-edits in h₁.

We use the prototype gallery's assignment (``prototype_winners``) to identify
the h₁ units whose *middle-slot prototype* is a given operator class. Then
we run three weight-edit interventions, one operator at a time, and measure
per-operator OCR on a held-out test set.

Interventions (applied to a fresh copy of the model each time):

* ``row``   — zero the full kernel column + bias for every targeted unit
              (i.e. that h₁ unit outputs 0 regardless of input).
* ``slot``  — zero only the middle 64-d slot of the targeted units; the
              digit pathways for those units are untouched.
* ``random``— same number of units chosen uniformly at random; baseline that
              isolates "how much does losing N units hurt by itself."

Headline output: a 4 × 4 specificity matrix per intervention
(rows = killed op, cols = evaluated op, values = OCR %). A clean
intervention shows the targeted operator's column dropping while the others
stay near the baseline; entanglement shows as off-diagonal damage.

Run::
    PYTHONPATH=/path/to/nmn/src:src python experiments/intervene.py
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from painting_arithmetic import data, glyphs
from painting_arithmetic.eval import OCRMetric, evaluate, load_checkpoint
from painting_arithmetic.model import YatArithmeticGen
from painting_arithmetic.viz import prototype_winners

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Library construction for prototype matching
# ---------------------------------------------------------------------------


def _norm(x: np.ndarray) -> np.ndarray:
    return (x - data.MNIST_MEAN) / data.MNIST_STD


def build_library(
    model: YatArithmeticGen,
    data_root: str,
    k_digit: int = 120,
    k_op: int = 60,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Encode a balanced library of digit + operator glyphs."""
    bins = data.load_mnist_binned(data_root, train=False)
    rng = random.Random(seed)
    tensors, labels = [], []
    for d in range(10):
        idx = rng.sample(range(bins[d].shape[0]), min(k_digit, bins[d].shape[0]))
        for i in idx:
            tensors.append(bins[d][i])               # already normalised
            labels.append(d)
    for op in range(4):
        for k in range(k_op):
            pil = (
                glyphs.render_glyph_clean(op) if k < max(1, k_op // 4)
                else glyphs.render_glyph_augmented(op, rng)
            )
            arr = _norm(np.asarray(pil, np.float32) / 255.0)
            tensors.append(arr)
            labels.append(glyphs.OP_LABEL_OFFSET + op)
    stack = np.stack(tensors)[:, None, :, :].astype(np.float32)        # (N, 1, 28, 28)
    emb = np.asarray(model.encoder(jnp.asarray(stack)))                # (N, 64)
    return emb, np.array(labels)


# ---------------------------------------------------------------------------
# Interventions
# ---------------------------------------------------------------------------


def _zero_kernel_columns(model: YatArithmeticGen, cols: np.ndarray) -> None:
    """Zero the listed output columns of ``h1.kernel``.

    The Yat bias sits *inside* the kernel numerator
    (``y = α (x·W + b)² / (‖x − W‖² + ε)``) and is therefore part of the
    kernel function itself rather than a separable additive feature.
    Zeroing it would edit the kernel definition for the targeted units
    rather than remove the landmark, so we leave the bias untouched and
    only zero the prototype rows.
    """
    if cols.size == 0:
        return
    k = np.array(model.h1.kernel[...])     # writable copy
    k[:, cols] = 0.0
    model.h1.kernel[...] = jnp.asarray(k)


def _zero_kernel_slice(
    model: YatArithmeticGen,
    cols: np.ndarray,
    slot: slice = slice(64, 128),
) -> None:
    """Zero only a slot range of the listed kernel output columns. Bias kept."""
    if cols.size == 0:
        return
    k = np.array(model.h1.kernel[...])     # writable copy
    k[slot.start:slot.stop, cols] = 0.0
    model.h1.kernel[...] = jnp.asarray(k)


def fresh_model(ckpt: str, use_yat_encoder: bool, single_yat: bool = False) -> YatArithmeticGen:
    m = YatArithmeticGen(use_yat_encoder=use_yat_encoder, single_yat=single_yat, rngs=nnx.Rngs(0))
    load_checkpoint(m, ckpt)
    return m


# ---------------------------------------------------------------------------
# Per-op evaluation
# ---------------------------------------------------------------------------


def per_op_ocr(
    model: YatArithmeticGen,
    bins: list[np.ndarray],
    test_size: int,
    batch_size: int,
    seed: int,
) -> dict:
    ds = data.build_dataset(bins, length=test_size, seed=seed, augment_op=False)
    loader = data.build_loader(ds, batch_size=batch_size, shuffle=False, num_epochs=1, seed=seed)
    metric = OCRMetric(glyphs.render_all_results())
    return evaluate(model, loader, metric)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run(
    ckpt: str,
    out_dir: str,
    *,
    data_root: str = "./data",
    test_size: int = 6000,
    batch_size: int = 256,
    purity_threshold: float = 0.5,
    seed: int = 0,
    single_yat: bool = False,
) -> dict:
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)

    print(f"loading {ckpt}")
    base = fresh_model(ckpt, use_yat_encoder=False, single_yat=single_yat)
    bins_test = data.load_mnist_binned(data_root, train=False)

    # ----- assignments -----
    print("building prototype library + assigning h₁ units to operator classes …")
    lib_emb, lib_lbl = build_library(base, data_root, seed=seed)
    winners = prototype_winners(base, lib_emb, lib_lbl, k=15)
    win_op = np.asarray(winners["win_op"])
    pur_op = np.asarray(winners["pur_op"])
    is_op = (win_op >= glyphs.OP_LABEL_OFFSET) & (pur_op >= purity_threshold)
    op_of_unit = np.where(is_op, win_op - glyphs.OP_LABEL_OFFSET, -1)
    counts = {op: int((op_of_unit == op).sum()) for op in range(4)}
    print(f"  units specialised per operator (purity ≥ {purity_threshold}): {counts}")
    print(f"  total operator-specialised units: {int(is_op.sum())} / 256")

    # ----- baseline -----
    print("\nbaseline (no intervention):")
    base_metrics = per_op_ocr(base, bins_test, test_size, batch_size, seed)
    baseline_row = {glyphs.OPS[i]: float(base_metrics["per_op_acc"][glyphs.OPS[i]]) for i in range(4)}
    print(f"  OCR={base_metrics['ocr_acc']*100:.2f}%  per-op={ {k: f'{v*100:.1f}%' for k, v in baseline_row.items()} }")

    # ----- interventions -----
    rng = np.random.default_rng(seed)
    results: dict[str, dict] = {"baseline": baseline_row,
                                 "counts": counts,
                                 "interventions": {}}

    for mode in ("row", "slot", "random"):
        mat: dict[str, dict] = {}
        print(f"\n=== intervention: {mode} ===")
        for op_to_kill in range(4):
            target_units = np.where(op_of_unit == op_to_kill)[0]
            if mode == "random":
                # Match the per-op target count so the comparison is fair.
                n = max(counts[op_to_kill], 1)
                target_units = rng.choice(256, size=n, replace=False)

            m = fresh_model(ckpt, use_yat_encoder=False, single_yat=single_yat)
            if mode == "row":
                _zero_kernel_columns(m, target_units)
            elif mode == "slot":
                _zero_kernel_slice(m, target_units, slot=slice(64, 128))
            elif mode == "random":
                _zero_kernel_columns(m, target_units)

            metrics = per_op_ocr(m, bins_test, test_size, batch_size, seed)
            row = {glyphs.OPS[i]: float(metrics["per_op_acc"][glyphs.OPS[i]]) for i in range(4)}
            row["_overall"] = float(metrics["ocr_acc"])
            mat[glyphs.OPS[op_to_kill]] = row
            print(f"  kill={glyphs.OPS[op_to_kill]:>2}  units_zeroed={len(target_units):>3}  "
                  f"+={row['+']*100:5.1f}  -={row['-']*100:5.1f}  *={row['*']*100:5.1f}  "
                  f"//={row['//']*100:5.1f}  overall={row['_overall']*100:5.1f}")
        results["interventions"][mode] = mat

    (out / "results.json").write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out / 'results.json'}")
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="ckpts_phased_v2/model.npz")
    p.add_argument("--out", default="experiments/intervene_out")
    p.add_argument("--data-root", default="./data")
    p.add_argument("--test-size", type=int, default=6000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--purity-threshold", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--single-yat", action="store_true",
                   help="The ckpt was trained with single_yat=True (h2=None).")
    args = p.parse_args()
    run(args.ckpt, args.out,
        data_root=args.data_root,
        test_size=args.test_size,
        batch_size=args.batch_size,
        purity_threshold=args.purity_threshold,
        seed=args.seed,
        single_yat=args.single_yat)


if __name__ == "__main__":
    main()
