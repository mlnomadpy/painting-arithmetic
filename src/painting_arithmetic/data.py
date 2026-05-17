"""Data pipeline using Grain.

We sample `(a, op, b)` triples on the fly, materialise the operator glyph
(augmented or canonical), pair them with random MNIST digit crops, and emit
all the auxiliary labels the loss needs.

Grain handles batching, shuffling, and multi-process loading. We expose a
``build_dataset`` factory and a ``build_loader`` helper that returns an
iterator yielding numpy/jax-friendly batches.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import grain.python as grain
import numpy as np
from torchvision import datasets, transforms

from . import glyphs

MNIST_MEAN, MNIST_STD = 0.1307, 0.3081


# ---------------------------------------------------------------------------
# MNIST loader (one-shot, into memory)
# ---------------------------------------------------------------------------


def load_mnist_binned(root: str | Path, train: bool = True) -> list[np.ndarray]:
    """Return ``bins[d]`` = (N_d, 28, 28) float32 MNIST tensors normalised
    with the standard MNIST mean/std."""
    tf = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
        ]
    )
    ds = datasets.MNIST(str(root), train=train, download=True, transform=tf)
    bins: list[list[np.ndarray]] = [[] for _ in range(10)]
    for img, lbl in ds:
        bins[int(lbl)].append(img.numpy().squeeze(0).astype(np.float32))
    return [np.stack(b) for b in bins]


# ---------------------------------------------------------------------------
# Per-sample record produced by the data pipeline
# ---------------------------------------------------------------------------


@dataclass
class Sample:
    """One training sample. All arrays are numpy on host."""

    img_a: np.ndarray           # (1, 28, 28) float32, MNIST-normalised
    img_op: np.ndarray          # (1, 28, 28) float32, MNIST-normalised
    img_b: np.ndarray           # (1, 28, 28) float32, MNIST-normalised
    target: np.ndarray          # (1, 28, 84) float32 in [0, 1]
    sym_a: np.int32             # 14-way symbol label for a
    sym_op: np.int32            # 14-way symbol label for op (10..13)
    sym_b: np.int32             # 14-way symbol label for b
    mod2: np.int32              # |r| mod 2
    mod5: np.int32              # |r| mod 5
    mod11: np.int32             # |r| mod 11
    sign: np.int32              # 1 if r < 0 else 0
    slot_sign: np.int32         # same as sign — for the per-slot head
    slot_tens: np.int32         # |r| // 10
    slot_units: np.int32        # |r| % 10
    result: np.int32            # signed integer truth (for eval only)
    op_idx: np.int32            # 0..3


# ---------------------------------------------------------------------------
# Grain source — synthetic, infinite (length-defined per epoch)
# ---------------------------------------------------------------------------


class ArithmeticSource(grain.RandomAccessDataSource):
    """A Grain ``RandomAccessDataSource`` that materialises one
    ``(img_a, img_op, img_b, ...)`` sample at index ``idx``.

    Despite the index, all sampling is fully randomised — ``idx`` only
    seeds a per-sample RNG so iteration is deterministic and reproducible.
    """

    def __init__(
        self,
        bins: list[np.ndarray],
        length: int,
        seed: int,
        augment_op: bool = True,
    ):
        self._bins = bins
        self._length = int(length)
        self._seed = seed
        self._augment_op = augment_op

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> Sample:
        rng = random.Random((self._seed * 2_147_483_647) ^ idx)
        nprng = np.random.default_rng((self._seed << 16) | idx)

        a = rng.randrange(10)
        b = rng.randrange(10)
        op = rng.randrange(4)
        if glyphs.OPS[op] == "//" and b == 0:
            b = rng.randrange(1, 10)
        if   glyphs.OPS[op] == "+":  r = a + b
        elif glyphs.OPS[op] == "-":  r = a - b
        elif glyphs.OPS[op] == "*":  r = a * b
        else:                         r = a // b

        # Operator glyph (augmented or canonical).
        op_pil = (
            glyphs.render_glyph_augmented(op, rng)
            if self._augment_op
            else glyphs.render_glyph_clean(op)
        )
        op_arr = np.asarray(op_pil, dtype=np.float32) / 255.0
        op_arr = (op_arr - MNIST_MEAN) / MNIST_STD

        # Operand digit crops.
        ia = self._bins[a][int(nprng.integers(0, self._bins[a].shape[0]))]
        ib = self._bins[b][int(nprng.integers(0, self._bins[b].shape[0]))]

        # Target image as float32 [0, 1].
        target = glyphs.render_result_image(r).astype(np.float32) / 255.0

        sign_lbl, tens_lbl, units_lbl = glyphs.slot_labels(r)
        absr = abs(r)
        return Sample(
            img_a=ia[None, :, :],
            img_op=op_arr[None, :, :],
            img_b=ib[None, :, :],
            target=target[None, :, :],
            sym_a=np.int32(a),
            sym_op=np.int32(glyphs.OP_LABEL_OFFSET + op),
            sym_b=np.int32(b),
            mod2=np.int32(absr % 2),
            mod5=np.int32(absr % 5),
            mod11=np.int32(absr % 11),
            sign=np.int32(sign_lbl),
            slot_sign=np.int32(sign_lbl),
            slot_tens=np.int32(tens_lbl),
            slot_units=np.int32(units_lbl),
            result=np.int32(r),
            op_idx=np.int32(op),
        )


# ---------------------------------------------------------------------------
# Batch collation
# ---------------------------------------------------------------------------


def _collate(samples: list[Sample]) -> dict[str, np.ndarray]:
    """Stack a list of ``Sample`` into a dict of batched numpy arrays.
    Grain calls this via ``MapOperation`` after batching."""
    return {
        "img_a":     np.stack([s.img_a     for s in samples]),
        "img_op":    np.stack([s.img_op    for s in samples]),
        "img_b":     np.stack([s.img_b     for s in samples]),
        "target":    np.stack([s.target    for s in samples]),
        "sym_a":     np.stack([s.sym_a     for s in samples]),
        "sym_op":    np.stack([s.sym_op    for s in samples]),
        "sym_b":     np.stack([s.sym_b     for s in samples]),
        "mod2":      np.stack([s.mod2      for s in samples]),
        "mod5":      np.stack([s.mod5      for s in samples]),
        "mod11":     np.stack([s.mod11     for s in samples]),
        "sign":      np.stack([s.sign      for s in samples]),
        "slot_sign": np.stack([s.slot_sign for s in samples]),
        "slot_tens": np.stack([s.slot_tens for s in samples]),
        "slot_units": np.stack([s.slot_units for s in samples]),
        "result":    np.stack([s.result    for s in samples]),
        "op_idx":    np.stack([s.op_idx    for s in samples]),
    }


# ---------------------------------------------------------------------------
# Public factories
# ---------------------------------------------------------------------------


def build_dataset(
    bins: list[np.ndarray],
    length: int,
    seed: int,
    augment_op: bool = True,
) -> ArithmeticSource:
    return ArithmeticSource(bins=bins, length=length, seed=seed, augment_op=augment_op)


def build_loader(
    source: ArithmeticSource,
    *,
    batch_size: int,
    shuffle: bool = True,
    num_epochs: int = 1,
    seed: int = 0,
    num_workers: int = 0,
) -> Iterator[dict[str, np.ndarray]]:
    """Wrap a source in a Grain DataLoader.

    Grain handles sharded reading, shuffling, and parallel preprocessing.
    We then batch via ``BatchOperation`` and collate to a dict-of-arrays.
    """
    sampler = grain.IndexSampler(
        num_records=len(source),
        num_epochs=num_epochs,
        shard_options=grain.NoSharding(),
        shuffle=shuffle,
        seed=seed,
    )
    class _Collate(grain.MapTransform):
        def map(self, samples):
            return _collate(samples)

    operations: list = [
        grain.Batch(batch_size=batch_size, drop_remainder=True),
        _Collate(),
    ]
    return grain.DataLoader(
        data_source=source,
        sampler=sampler,
        operations=operations,
        worker_count=num_workers,
    )


__all__ = [
    "Sample",
    "ArithmeticSource",
    "load_mnist_binned",
    "build_dataset",
    "build_loader",
    "MNIST_MEAN",
    "MNIST_STD",
]
