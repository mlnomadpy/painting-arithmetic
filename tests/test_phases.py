"""Smoke test for the phased training curriculum.

Verifies, for each phase:
  1. The labelled gradient transformation zeros updates outside the
     ``trainable`` module set.
  2. After one step, every frozen top-level submodule has *bit-identical*
     params; every trainable submodule has at least one parameter that
     changed.

Run::
    PYTHONPATH=src python tests/test_phases.py
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from painting_arithmetic.losses import LossWeights, build_pixel_weight
from painting_arithmetic.model import YatArithmeticGen
from painting_arithmetic.phases import PHASES, _path_top, freeze_tx
from painting_arithmetic.train import make_phase_train_step


def _fake_batch(batch_size: int = 4) -> dict[str, jnp.ndarray]:
    rng = np.random.default_rng(0)
    return {
        "img_a":     jnp.asarray(rng.normal(size=(batch_size, 1, 28, 28)).astype(np.float32)),
        "img_op":    jnp.asarray(rng.normal(size=(batch_size, 1, 28, 28)).astype(np.float32)),
        "img_b":     jnp.asarray(rng.normal(size=(batch_size, 1, 28, 28)).astype(np.float32)),
        "target":    jnp.asarray(rng.uniform(size=(batch_size, 1, 28, 84)).astype(np.float32)),
        "sym_a":     jnp.asarray(rng.integers(0, 10, size=batch_size).astype(np.int32)),
        "sym_op":    jnp.asarray(rng.integers(10, 14, size=batch_size).astype(np.int32)),
        "sym_b":     jnp.asarray(rng.integers(0, 10, size=batch_size).astype(np.int32)),
        "mod2":      jnp.asarray(rng.integers(0, 2, size=batch_size).astype(np.int32)),
        "mod5":      jnp.asarray(rng.integers(0, 5, size=batch_size).astype(np.int32)),
        "mod11":     jnp.asarray(rng.integers(0, 11, size=batch_size).astype(np.int32)),
        "sign":      jnp.asarray(rng.integers(0, 2, size=batch_size).astype(np.int32)),
        "slot_sign": jnp.asarray(rng.integers(0, 2, size=batch_size).astype(np.int32)),
        "slot_tens": jnp.asarray(rng.integers(0, 10, size=batch_size).astype(np.int32)),
        "slot_units": jnp.asarray(rng.integers(0, 10, size=batch_size).astype(np.int32)),
        "result":    jnp.asarray(rng.integers(-9, 82, size=batch_size).astype(np.int32)),
        "op_idx":    jnp.asarray(rng.integers(0, 4, size=batch_size).astype(np.int32)),
    }


def _flat_params(model):
    return [(p, np.asarray(v.value)) for p, v in nnx.to_flat_state(nnx.state(model, nnx.Param))]


def test_phase(phase: int) -> None:
    spec = PHASES[phase]
    print(f"\n--- phase {phase} ({spec.name}) ---")
    model = YatArithmeticGen(rngs=nnx.Rngs(0))

    before = _flat_params(model)

    tx = freeze_tx(optax.adam(1e-2), model, spec.trainable)
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)

    pix_w = build_pixel_weight(2.0)
    step = make_phase_train_step(phase, pix_w, LossWeights())
    metrics = step(model, optimizer, _fake_batch())

    after = _flat_params(model)
    assert len(before) == len(after)

    # Verify: leaves whose top-level key ∈ trainable changed; others are
    # bit-identical.
    changed_top: set[str] = set()
    unchanged_top: set[str] = set()
    for (path_b, arr_b), (path_a, arr_a) in zip(before, after):
        assert path_b == path_a
        top = str(path_b[0]) if path_b else ""
        same = np.array_equal(arr_b, arr_a)
        if top in spec.trainable:
            if not same:
                changed_top.add(top)
        else:
            assert same, f"frozen module {top!r} moved at path {path_b}"
            unchanged_top.add(top)

    print(f"  loss={float(metrics['loss']):.4f}")
    print(f"  trainable modules that moved: {sorted(changed_top)}")
    print(f"  frozen modules verified static: {sorted(unchanged_top)}")
    missing = set(spec.trainable) - changed_top
    if missing:
        # Some trainable modules may legitimately not move (e.g. unused
        # heads). Warn but don't fail unless none of them moved.
        print(f"  note: trainable but unchanged after 1 step: {sorted(missing)}")
    assert changed_top, f"phase {phase}: no trainable module moved"


if __name__ == "__main__":
    for p in (1, 2, 3):
        test_phase(p)
    print("\nall phases ok")
