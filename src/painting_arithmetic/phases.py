"""Three-phase training curriculum.

Each phase declares (a) which top-level modules of :class:`YatArithmeticGen`
are trainable and (b) the loss function used. Modules outside the trainable
set receive ``optax.set_to_zero`` updates via :func:`freeze_tx`, and the
loss functions add ``stop_gradient`` so the wasted compute never happens
in the first place.

Phase summary::

    1  encoder      ← encoder + aux_sym                   (CE on symbols)
    2  trunk        ← h1, h2, mod heads, slot heads       (CE on mod + slot)
    3  decoder      ← decoder only                        (BCE on painted image)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from . import losses as L
from .model import YatArithmeticGen


# ---------------------------------------------------------------------------
# Phase registry
# ---------------------------------------------------------------------------


LossFn = Callable[
    [YatArithmeticGen, dict, L.LossWeights, jnp.ndarray],
    tuple[jnp.ndarray, dict],
]


@dataclass(frozen=True)
class PhaseSpec:
    name: str
    trainable: frozenset[str]
    loss_fn: LossFn
    headline_metric: str   # key into the metrics dict to print each epoch


_PHASE_2_MODULES = frozenset({
    "h1", "h2",
    "head_mod2", "head_mod5", "head_mod11", "head_sign",
    "head_slot_sign", "head_slot_tens", "head_slot_units",
})


PHASES: dict[int, PhaseSpec] = {
    1: PhaseSpec(
        name="encoder",
        trainable=frozenset({"encoder", "aux_sym"}),
        loss_fn=L.phase1_loss,
        headline_metric="sym_acc",
    ),
    2: PhaseSpec(
        name="trunk",
        trainable=_PHASE_2_MODULES,
        loss_fn=L.phase2_loss,
        headline_metric="slot_acc",
    ),
    3: PhaseSpec(
        name="decoder",
        trainable=frozenset({"decoder"}),
        loss_fn=L.phase3_loss,
        headline_metric="loss_pix",
    ),
}


# ---------------------------------------------------------------------------
# Freezing — wrap an optax tx with multi_transform so frozen leaves get 0s
# ---------------------------------------------------------------------------


def _path_top(path: tuple) -> str:
    """First-level key of a ``jax.tree_util`` key path."""
    if not path:
        return ""
    first = path[0]
    if hasattr(first, "key"):
        return str(first.key)
    if hasattr(first, "name"):
        return str(first.name)
    if hasattr(first, "idx"):
        return str(first.idx)
    return str(first)


def freeze_labels_for(params, trainable: frozenset[str]):
    """Return a pytree matching ``params`` whose leaves are ``"train"`` or
    ``"frozen"`` based on the top-level submodule name in each path."""
    return jax.tree_util.tree_map_with_path(
        lambda path, _v: "train" if _path_top(path) in trainable else "frozen",
        params,
    )


def freeze_tx(
    real_tx: optax.GradientTransformation,
    model: YatArithmeticGen,
    trainable: frozenset[str],
) -> optax.GradientTransformation:
    """Wrap ``real_tx`` so only leaves under ``trainable`` top-level modules
    receive updates; everything else gets ``optax.set_to_zero``.

    ``param_labels`` is supplied as a callable so optax invokes it on its
    own view of the params tree (plain arrays, no nnx ``Variable`` nodes).
    """
    del model  # only the trainable set drives the labeling
    return optax.multi_transform(
        {"train": real_tx, "frozen": optax.set_to_zero()},
        lambda params: freeze_labels_for(params, trainable),
    )


__all__ = ["PHASES", "PhaseSpec", "freeze_labels", "freeze_tx"]
