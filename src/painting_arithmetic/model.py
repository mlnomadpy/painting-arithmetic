"""The Yat-arithmetic model, written in Flax NNX.

Encoder → Yat trunk → ConvTranspose decoder. All training-only auxiliary
heads (sym, mod CRT, slot) live on the trunk and are only used inside
``forward_train``; they don't appear in ``__call__``.

Two encoder variants are available behind the ``use_yat_encoder`` flag:

* ``False``  — stock ``nnx.Conv + GELU`` blocks.
* ``True``   — ``YatConv`` everywhere. Same parameter count (YatConv adds only
  per-channel ``alpha``), no extra activation needed because the rational
  kernel is already nonlinear.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
from flax import nnx

from nmn.nnx.layers.conv.yat_conv import YatConv
from nmn.nnx.layers.nmn import YatNMN

from .glyphs import NUM_SYMBOLS


# ---------------------------------------------------------------------------
# Encoder variants
# ---------------------------------------------------------------------------


class StockEncoder(nnx.Module):
    """4-layer Conv + GELU CNN, ends in a 64-d embedding."""

    def __init__(self, emb_dim: int = 64, *, rngs: nnx.Rngs):
        self.c1 = nnx.Conv(1, 32, kernel_size=(3, 3), padding="SAME", rngs=rngs)
        self.c2 = nnx.Conv(32, 32, kernel_size=(3, 3), strides=(2, 2), padding="SAME", rngs=rngs)
        self.c3 = nnx.Conv(32, 64, kernel_size=(3, 3), padding="SAME", rngs=rngs)
        self.c4 = nnx.Conv(64, 64, kernel_size=(3, 3), strides=(2, 2), padding="SAME", rngs=rngs)
        self.proj = nnx.Linear(64, emb_dim, rngs=rngs)

    def __call__(self, x):
        # NNX/JAX conv expects (B, H, W, C); we accept (B, 1, 28, 28) and transpose.
        if x.ndim == 4 and x.shape[1] == 1:
            x = jnp.transpose(x, (0, 2, 3, 1))
        x = nnx.gelu(self.c1(x))
        x = nnx.gelu(self.c2(x))
        x = nnx.gelu(self.c3(x))
        x = nnx.gelu(self.c4(x))
        x = jnp.mean(x, axis=(1, 2))   # adaptive average pool to (B, C)
        return self.proj(x)


class YatEncoder(nnx.Module):
    """Same shape as ``StockEncoder``, every conv is a ``YatConv`` rational
    kernel. No GELU — the kernel is already nonlinear."""

    def __init__(self, emb_dim: int = 64, *, rngs: nnx.Rngs):
        self.c1 = YatConv(1, 32, kernel_size=(3, 3), padding="SAME", rngs=rngs)
        self.c2 = YatConv(32, 32, kernel_size=(3, 3), strides=(2, 2), padding="SAME", rngs=rngs)
        self.c3 = YatConv(32, 64, kernel_size=(3, 3), padding="SAME", rngs=rngs)
        self.c4 = YatConv(64, 64, kernel_size=(3, 3), strides=(2, 2), padding="SAME", rngs=rngs)
        self.proj = nnx.Linear(64, emb_dim, rngs=rngs)

    def __call__(self, x):
        if x.ndim == 4 and x.shape[1] == 1:
            x = jnp.transpose(x, (0, 2, 3, 1))
        x = self.c1(x)
        x = self.c2(x)
        x = self.c3(x)
        x = self.c4(x)
        x = jnp.mean(x, axis=(1, 2))
        return self.proj(x)


# ---------------------------------------------------------------------------
# Decoder — small ConvTranspose stack: ℝ²⁵⁶ → 28×84 image
# ---------------------------------------------------------------------------


class Decoder(nnx.Module):
    """Linear projection → reshape → 2× ConvTranspose → sigmoid 28×84."""

    def __init__(self, hidden: int = 256, *, rngs: nnx.Rngs):
        self.fc = nnx.Linear(hidden, 16 * 7 * 21, rngs=rngs)
        self.ct1 = nnx.ConvTranspose(
            16, 16, kernel_size=(4, 4), strides=(2, 2), padding="SAME", rngs=rngs
        )
        self.c1 = nnx.Conv(16, 16, kernel_size=(3, 3), padding="SAME", rngs=rngs)
        self.ct2 = nnx.ConvTranspose(
            16, 8, kernel_size=(4, 4), strides=(2, 2), padding="SAME", rngs=rngs
        )
        self.c2 = nnx.Conv(8, 1, kernel_size=(3, 3), padding="SAME", rngs=rngs)

    def __call__(self, t):
        x = self.fc(t).reshape(-1, 7, 21, 16)            # (B, H, W, C) NHWC
        x = nnx.gelu(self.ct1(x))                         # 14 × 42
        x = nnx.gelu(self.c1(x))
        x = nnx.gelu(self.ct2(x))                         # 28 × 84
        x = nnx.sigmoid(self.c2(x))
        return jnp.transpose(x, (0, 3, 1, 2))             # back to (B, 1, 28, 84)


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------


class TrainOut(NamedTuple):
    """Bundle of forward_train outputs — image plus every auxiliary head."""

    img: jnp.ndarray
    aux_a: jnp.ndarray
    aux_op: jnp.ndarray
    aux_b: jnp.ndarray
    head_mod2: jnp.ndarray
    head_mod5: jnp.ndarray
    head_mod11: jnp.ndarray
    head_sign: jnp.ndarray
    slot_sign: jnp.ndarray
    slot_tens: jnp.ndarray
    slot_units: jnp.ndarray


class InferOut(NamedTuple):
    img: jnp.ndarray
    aux_a: jnp.ndarray
    aux_op: jnp.ndarray
    aux_b: jnp.ndarray


class YatArithmeticGen(nnx.Module):
    """Shared CNN encoder + 2× Yat layers + ConvTranspose decoder.

    Auxiliary heads (training-only): per-input 14-way symbol classifier,
    modular CRT classifiers (mod 2, mod 5, mod 11, sign), per-slot
    classifiers (sign, tens, units).
    """

    def __init__(
        self,
        emb_dim: int = 64,
        hidden: int = 256,
        use_yat_encoder: bool = True,
        *,
        rngs: nnx.Rngs,
    ):
        self.encoder = (YatEncoder if use_yat_encoder else StockEncoder)(emb_dim, rngs=rngs)
        fused = 3 * emb_dim
        self.h1 = YatNMN(fused, hidden, rngs=rngs)
        self.h2 = YatNMN(hidden, hidden, rngs=rngs)
        self.decoder = Decoder(hidden, rngs=rngs)

        self.aux_sym = nnx.Linear(emb_dim, NUM_SYMBOLS, rngs=rngs)
        # Training-only modular CRT heads on the trunk.
        self.head_mod2 = nnx.Linear(hidden, 2, rngs=rngs)
        self.head_mod5 = nnx.Linear(hidden, 5, rngs=rngs)
        self.head_mod11 = nnx.Linear(hidden, 11, rngs=rngs)
        self.head_sign = nnx.Linear(hidden, 2, rngs=rngs)
        # Training-only per-slot classification heads (the LOAD-BEARING fix
        # for the multi-digit collapse failure mode).
        self.head_slot_sign = nnx.Linear(hidden, 2, rngs=rngs)
        self.head_slot_tens = nnx.Linear(hidden, 10, rngs=rngs)
        self.head_slot_units = nnx.Linear(hidden, 10, rngs=rngs)

    # ------------------------------------------------------------------ trunk
    def trunk(self, img_a, img_op, img_b):
        ea = self.encoder(img_a)
        eo = self.encoder(img_op)
        eb = self.encoder(img_b)
        h = jnp.concatenate([ea, eo, eb], axis=-1)
        t1 = self.h1(h)
        t2 = self.h2(t1)
        return ea, eo, eb, h, t1, t2

    # ------------------------------------------------------------------ infer
    def __call__(self, img_a, img_op, img_b) -> InferOut:
        ea, eo, eb, _, _, t = self.trunk(img_a, img_op, img_b)
        img = self.decoder(t)
        return InferOut(img=img,
                        aux_a=self.aux_sym(ea),
                        aux_op=self.aux_sym(eo),
                        aux_b=self.aux_sym(eb))

    # ------------------------------------------------------------------ train
    def forward_train(self, img_a, img_op, img_b) -> TrainOut:
        ea, eo, eb, _, _, t = self.trunk(img_a, img_op, img_b)
        img = self.decoder(t)
        return TrainOut(
            img=img,
            aux_a=self.aux_sym(ea),
            aux_op=self.aux_sym(eo),
            aux_b=self.aux_sym(eb),
            head_mod2=self.head_mod2(t),
            head_mod5=self.head_mod5(t),
            head_mod11=self.head_mod11(t),
            head_sign=self.head_sign(t),
            slot_sign=self.head_slot_sign(t),
            slot_tens=self.head_slot_tens(t),
            slot_units=self.head_slot_units(t),
        )


__all__ = ["YatArithmeticGen", "TrainOut", "InferOut", "StockEncoder", "YatEncoder", "Decoder"]
