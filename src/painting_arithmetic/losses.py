"""Loss functions for the Painting Arithmetic model.

Two training regimes are supported:

* **Joint** (legacy). One forward pass, BCE on the painted image plus the
  three auxiliary heads (symbol / modular CRT / per-slot). Gradients flow
  end-to-end. See :func:`compute_loss`.

* **Phased** (recommended). Three sequential stages with strict
  ``stop_gradient`` cuts so pixel-space supervision cannot leak back into
  the encoder or the trunk:

    1. ``phase1_loss`` — encoder + ``aux_sym`` head. CE on each of the three
       inputs. The encoder learns symbol recognition with no arithmetic
       context.
    2. ``phase2_loss`` — frozen encoder. Train ``h1``, ``h2`` and the
       modular / per-slot heads on stop-gradiented embeddings. The trunk
       learns arithmetic in latent space; the decoder is never touched.
    3. ``phase3_loss`` — frozen encoder + trunk. Train only the decoder
       with BCE on the painted target image. ``stop_gradient`` on ``t₂``
       guarantees no decoder gradient can sneak back into the math.

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
from .model import TrainOut, YatArithmeticGen


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


# ---------------------------------------------------------------------------
# Phased losses — strict stop_gradient between stages
# ---------------------------------------------------------------------------


def phase1_loss(
    model: YatArithmeticGen,
    batch: dict[str, jnp.ndarray],
    weights: LossWeights,
    pix_weight: jnp.ndarray,
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    """Encoder-only: 14-way symbol CE on each of the three inputs.

    Pixel and arithmetic supervision are absent; the encoder learns to
    discriminate the 14 symbols (10 digits + 4 operators) with no
    arithmetic context.
    """
    del pix_weight  # unused
    ea = model.encoder(batch["img_a"])
    eo = model.encoder(batch["img_op"])
    eb = model.encoder(batch["img_b"])
    la = model.aux_sym(ea); lo = model.aux_sym(eo); lb = model.aux_sym(eb)
    loss_sym = (_ce(la, batch["sym_a"]) + _ce(lo, batch["sym_op"]) + _ce(lb, batch["sym_b"])) / 3.0

    logits = jnp.concatenate([la, lo, lb], axis=0)
    truth = jnp.concatenate([batch["sym_a"], batch["sym_op"], batch["sym_b"]], axis=0)
    sym_acc = (logits.argmax(-1) == truth).mean()

    return loss_sym, {"loss": loss_sym, "loss_sym": loss_sym, "sym_acc": sym_acc}


def phase2_loss(
    model: YatArithmeticGen,
    batch: dict[str, jnp.ndarray],
    weights: LossWeights,
    pix_weight: jnp.ndarray,
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    """Trunk-only: stop_gradient on encoder outputs, train ``h1``/``h2`` +
    modular + per-slot heads. The decoder is never called."""
    del pix_weight  # unused
    ea = jax.lax.stop_gradient(model.encoder(batch["img_a"]))
    eo = jax.lax.stop_gradient(model.encoder(batch["img_op"]))
    eb = jax.lax.stop_gradient(model.encoder(batch["img_b"]))
    h = jnp.concatenate([ea, eo, eb], axis=-1)
    t1 = model.h1(h)
    t2 = t1 if model.h2 is None else model.h2(t1)

    loss_mod = (
        _ce(model.head_mod2(t2), batch["mod2"])
        + _ce(model.head_mod5(t2), batch["mod5"])
        + _ce(model.head_mod11(t2), batch["mod11"])
        + _ce(model.head_sign(t2), batch["sign"])
    )
    s_sign = model.head_slot_sign(t2)
    s_tens = model.head_slot_tens(t2)
    s_units = model.head_slot_units(t2)
    loss_slot = (_ce(s_sign, batch["slot_sign"]) + _ce(s_tens, batch["slot_tens"]) + _ce(s_units, batch["slot_units"])) / 3.0

    total = weights.aux_mod * loss_mod + weights.aux_slot * loss_slot

    ok_sign = s_sign.argmax(-1) == batch["slot_sign"]
    ok_tens = s_tens.argmax(-1) == batch["slot_tens"]
    ok_units = s_units.argmax(-1) == batch["slot_units"]
    slot_acc = (ok_sign & ok_tens & ok_units).mean()

    return total, {
        "loss": total,
        "loss_mod": loss_mod,
        "loss_slot": loss_slot,
        "slot_acc": slot_acc,
    }


def phase3_loss(
    model: YatArithmeticGen,
    batch: dict[str, jnp.ndarray],
    weights: LossWeights,
    pix_weight: jnp.ndarray,
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    """Decoder-only: stop_gradient on the trunk vector, train ``decoder``
    on BCE against the 28×84 target image."""
    del weights  # unused
    ea = model.encoder(batch["img_a"])
    eo = model.encoder(batch["img_op"])
    eb = model.encoder(batch["img_b"])
    h = jnp.concatenate([ea, eo, eb], axis=-1)
    t1 = model.h1(h)
    t2 = t1 if model.h2 is None else model.h2(t1)
    t = jax.lax.stop_gradient(t2)
    img = model.decoder(t)

    pix_map = _bce(img, batch["target"])
    loss_pix = (pix_map * pix_weight).mean()
    return loss_pix, {"loss": loss_pix, "loss_pix": loss_pix}


__all__ = [
    "LossWeights",
    "build_pixel_weight",
    "compute_loss",
    "phase1_loss",
    "phase2_loss",
    "phase3_loss",
]
