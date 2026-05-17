"""Evaluation utilities.

The headline metric is **OCR accuracy**: nearest-neighbour pixel distance
from the predicted 28×84 image to each of the 91 canonical result images,
counted correct when the nearest neighbour is the true result. This metric
never inspects the model's training-time auxiliary heads.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from . import data, glyphs
from .model import YatArithmeticGen


# ---------------------------------------------------------------------------
# OCR metric
# ---------------------------------------------------------------------------


@dataclass
class OCRMetric:
    targets: np.ndarray  # (91, 28, 84) float32 in [0, 1]

    def __post_init__(self):
        self._t = jnp.asarray(self.targets.reshape(self.targets.shape[0], -1))

    def pred_to_int(self, img: jnp.ndarray) -> jnp.ndarray:
        """``img``: (B, 1, 28, 84). Returns (B,) int32 in [-9, 81]."""
        flat = img.reshape(img.shape[0], -1)
        # Squared L2 distance to every target.
        diff = flat[:, None, :] - self._t[None, :, :]
        d2 = (diff * diff).sum(axis=-1)
        idx = jnp.argmin(d2, axis=-1)
        return idx + glyphs.RESULT_MIN


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------


def evaluate(
    model: YatArithmeticGen,
    loader: Iterable[dict[str, np.ndarray]],
    ocr: OCRMetric,
) -> dict:
    """Run the model over ``loader`` and return aggregate + per-op metrics."""

    @nnx.jit
    def forward(model_, ia, io, ib):
        return model_(ia, io, ib).img

    n = 0
    correct = 0
    per_op_n = [0] * 4
    per_op_ok = [0] * 4
    for batch in loader:
        ia = jnp.asarray(batch["img_a"]); io = jnp.asarray(batch["img_op"]); ib = jnp.asarray(batch["img_b"])
        img = forward(model, ia, io, ib)
        pred_r = np.asarray(ocr.pred_to_int(img))
        true_r = np.asarray(batch["result"])
        ok = pred_r == true_r
        correct += int(ok.sum())
        n += pred_r.size
        op_idx = np.asarray(batch["op_idx"])
        for i in range(4):
            mask = op_idx == i
            per_op_n[i] += int(mask.sum())
            per_op_ok[i] += int((ok & mask).sum())

    return {
        "ocr_acc": correct / max(1, n),
        "per_op_acc": {
            glyphs.OPS[i]: per_op_ok[i] / max(1, per_op_n[i]) for i in range(4)
        },
        "per_op_n": {glyphs.OPS[i]: per_op_n[i] for i in range(4)},
    }


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------


def load_checkpoint(model: YatArithmeticGen, path: str | Path):
    """Load a numpy-archive checkpoint produced by :mod:`painting_arithmetic.train`."""
    npz = np.load(path)
    state = nnx.state(model, nnx.Param)
    flat = dict(nnx.to_flat_state(state))
    for key_str, arr in npz.items():
        path_tuple = tuple(int(p) if p.isdigit() else p for p in key_str.split("."))
        if path_tuple in flat:
            flat[path_tuple].value = jnp.asarray(arr)
    new_state = nnx.from_flat_state(list(flat.items()))
    nnx.update(model, new_state)
    return model


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="./ckpts/model.npz")
    p.add_argument("--data-root", default="./data")
    p.add_argument("--test-size", type=int, default=10_000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--use-yat-encoder", action="store_true", default=True)
    args = p.parse_args()

    bins = data.load_mnist_binned(args.data_root, train=False)
    test_ds = data.build_dataset(bins, length=args.test_size, seed=999, augment_op=False)
    loader = data.build_loader(test_ds, batch_size=args.batch_size, shuffle=False, num_epochs=1)

    model = YatArithmeticGen(use_yat_encoder=args.use_yat_encoder, rngs=nnx.Rngs(0))
    load_checkpoint(model, args.ckpt)
    ocr = OCRMetric(glyphs.render_all_results())
    metrics = evaluate(model, loader, ocr)
    print(metrics)


if __name__ == "__main__":
    main()


__all__ = ["OCRMetric", "evaluate", "load_checkpoint"]
