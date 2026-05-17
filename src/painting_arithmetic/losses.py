"""Loss functions for the Painting Arithmetic model.

The headline loss is BCE on the predicted 28×84 image with a per-pixel
weight (2× on the tens-slot column).

Three auxiliary, training-only loss terms shape the trunk:

* ``aux_sym``  — 14-way cross-entropy on each input embedding
* ``aux_mod``  — modular CRT cross-entropy (mod 2 / 5 / 11 / sign) on ``t₂``
* ``aux_slot`` — per-slot cross-entropy (sign / tens / units) on ``t₂``

The per-slot CE is the load-bearing fix for the multi-digit collapse: when
removed, the decoder defaults to leaving the tens column blank (~72% of
training results fit in one digit).
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import optax

from .glyphs import IMG_H, IMG_W
from .model import TrainOut


@dataclass(frozen=True)
class LossWeights:
    aux_sym: float = 0.2
    aux_mod: float = 0.05
    aux_slot: float = 0.5
    tens_weight: float = 2.0


def build_pixel_weight(tens_weight: float = 2.0) -> jnp.ndarray:
    """Per-pixel multiplier for the BCE loss — 2× over the tens-slot column."""
    w = jnp.ones((1, 1, IMG_H, IMG_W), dtype=jnp.float32)
    w = w.at[:, :, :, 28:56].set(tens_weight)
    return w


def _bce(pred: jnp.ndarray, target: jnp.ndarray, eps: float = 1e-7) -> jnp.ndarray:
    """Element-wise binary cross-entropy."""
    pred = jnp.clip(pred, eps, 1.0 - eps)
    return -(target * jnp.log(pred) + (1.0 - target) * jnp.log1p(-pred))


def _ce(logits: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
    """Sparse cross-entropy mean over batch."""
    return optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()


def compute_loss(
    out: TrainOut,
    batch: dict[str, jnp.ndarray],
    weights: LossWeights,
    pix_weight: jnp.ndarray,
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    """Total loss + a dict of scalar metrics for logging."""
    target = batch["target"]
    pix_map = _bce(out.img, target)
    loss_pix = (pix_map * pix_weight).mean()

    loss_sym = (
        _ce(out.aux_a, batch["sym_a"]) + _ce(out.aux_op, batch["sym_op"]) + _ce(out.aux_b, batch["sym_b"])
    ) / 3.0
    loss_mod = (
        _ce(out.head_mod2, batch["mod2"])
        + _ce(out.head_mod5, batch["mod5"])
        + _ce(out.head_mod11, batch["mod11"])
        + _ce(out.head_sign, batch["sign"])
    )
    loss_slot = (
        _ce(out.slot_sign, batch["slot_sign"])
        + _ce(out.slot_tens, batch["slot_tens"])
        + _ce(out.slot_units, batch["slot_units"])
    ) / 3.0

    total = (
        loss_pix
        + weights.aux_sym * loss_sym
        + weights.aux_mod * loss_mod
        + weights.aux_slot * loss_slot
    )
    metrics = {
        "loss":      total,
        "loss_pix":  loss_pix,
        "loss_sym":  loss_sym,
        "loss_mod":  loss_mod,
        "loss_slot": loss_slot,
    }
    return total, metrics


__all__ = ["LossWeights", "build_pixel_weight", "compute_loss"]
