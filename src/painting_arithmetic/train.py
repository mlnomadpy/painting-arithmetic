"""Training script for the Painting Arithmetic model.

Two CLIs share this module:

* **Phased** (recommended). Trains in three strict stages with
  ``stop_gradient`` cuts so pixel supervision can't leak into the
  encoder or trunk::

      # Run the full curriculum end-to-end:
      python -m painting_arithmetic.train --phase all --epochs-1 8 --epochs-2 15 --epochs-3 8

      # Or one stage at a time (each phase auto-loads the prior ckpt):
      python -m painting_arithmetic.train --phase 1 --epochs 8
      python -m painting_arithmetic.train --phase 2 --epochs 15
      python -m painting_arithmetic.train --phase 3 --epochs 8

* **Joint** (legacy). Single-pass training of all heads at once::

      python -m painting_arithmetic.train --phase joint --epochs 15

Programmatic use::

    from painting_arithmetic.train import TrainConfig, train_all_phases
    out = train_all_phases(TrainConfig(epochs_per_phase=(8, 15, 8)))
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from . import data, glyphs
from .eval import OCRMetric, evaluate, load_checkpoint
from .losses import LossWeights, build_pixel_weight, compute_loss
from .model import YatArithmeticGen
from .phases import PHASES, freeze_tx


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class TrainConfig:
    epochs: int = 15                                  # used by joint + single-phase
    epochs_per_phase: tuple[int, int, int] = (8, 15, 8)
    train_size: int = 30_000
    test_size: int = 6_000
    batch_size: int = 256
    lr: float = 2e-3
    weight_decay: float = 1e-4
    warmup_frac: float = 0.05
    seed: int = 0
    use_yat_encoder: bool = False
    single_yat: bool = False
    loss: LossWeights = field(default_factory=LossWeights)
    data_root: str = "./data"
    ckpt_dir: str = "./ckpts"
    log_every: int = 50

    def steps_for(self, epochs: int) -> int:
        return epochs * (self.train_size // self.batch_size)


# ---------------------------------------------------------------------------
# Optimiser + LR schedule
# ---------------------------------------------------------------------------


def build_schedule(cfg: TrainConfig, epochs: int) -> optax.Schedule:
    total = cfg.steps_for(epochs)
    warmup = max(1, int(cfg.warmup_frac * total))
    return optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=cfg.lr,
        warmup_steps=warmup,
        decay_steps=total - warmup,
        end_value=cfg.lr * 1e-2,
    )


def build_base_tx(cfg: TrainConfig, epochs: int) -> optax.GradientTransformation:
    return optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=build_schedule(cfg, epochs),
                    weight_decay=cfg.weight_decay),
    )


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------


def save_checkpoint(model: YatArithmeticGen, path: str | Path) -> Path:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    state = nnx.state(model, nnx.Param)
    flat = nnx.to_flat_state(state)
    np.savez(
        path,
        **{".".join(map(str, k)): np.asarray(v.value) for k, v in flat},
    )
    return path


# ---------------------------------------------------------------------------
# Joint (legacy) train step — kept for back-compat
# ---------------------------------------------------------------------------


def make_joint_train_step(pix_weight: jnp.ndarray, loss_weights: LossWeights):
    @nnx.jit(donate_argnames=("optimizer",))
    def train_step(model: YatArithmeticGen, optimizer: nnx.Optimizer, batch):
        def loss_fn(m):
            out = m.forward_train(batch["img_a"], batch["img_op"], batch["img_b"])
            return compute_loss(out, batch, loss_weights, pix_weight)

        (_, metrics), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
        optimizer.update(model, grads)
        return metrics

    return train_step


# ---------------------------------------------------------------------------
# Phased train step
# ---------------------------------------------------------------------------


def make_phase_train_step(phase: int, pix_weight: jnp.ndarray, loss_weights: LossWeights):
    spec = PHASES[phase]

    @nnx.jit(donate_argnames=("optimizer",))
    def train_step(model: YatArithmeticGen, optimizer: nnx.Optimizer, batch):
        def loss_fn(m):
            return spec.loss_fn(m, batch, loss_weights, pix_weight)
        (_, metrics), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
        optimizer.update(model, grads)
        return metrics

    return train_step


def make_phase_eval_step(phase: int, pix_weight: jnp.ndarray, loss_weights: LossWeights):
    """JIT'd forward-only pass that returns the phase's loss + metrics on a batch."""
    spec = PHASES[phase]

    @nnx.jit
    def eval_step(model: YatArithmeticGen, batch):
        return spec.loss_fn(model, batch, loss_weights, pix_weight)

    return eval_step


def evaluate_phase(model, loader, eval_step, headline_key: str) -> dict:
    """Average phase-specific metrics across the test loader. Returns the
    headline value plus the loss for logging."""
    sums: dict[str, float] = {}
    n_batches = 0
    for batch in loader:
        batch_jax = {k: jnp.asarray(v) for k, v in batch.items()}
        _, metrics = eval_step(model, batch_jax)
        for k, v in metrics.items():
            sums[k] = sums.get(k, 0.0) + float(v)
        n_batches += 1
    avg = {k: v / max(1, n_batches) for k, v in sums.items()}
    return {"headline": avg.get(headline_key, 0.0), **avg}


# ---------------------------------------------------------------------------
# Shared epoch driver
# ---------------------------------------------------------------------------


def _device_str() -> str:
    return ", ".join(str(d) for d in jax.devices())


def _run_epochs(
    model: YatArithmeticGen,
    optimizer: nnx.Optimizer,
    train_step,
    train_loader,
    *,
    epochs: int,
    steps_per_epoch: int,
    on_epoch_end,
    label: str,
):
    """Common loop: pull batches, call train_step, call ``on_epoch_end``."""
    history: list[dict] = []
    t0 = time.time()
    step = 0
    epoch_loss = 0.0
    epoch_seen = 0

    for batch in train_loader:
        batch_jax = {k: jnp.asarray(v) for k, v in batch.items()}
        metrics = train_step(model, optimizer, batch_jax)
        epoch_loss += float(metrics["loss"]) * batch_jax["target"].shape[0]
        epoch_seen += batch_jax["target"].shape[0]
        step += 1

        if step % steps_per_epoch == 0:
            epoch = step // steps_per_epoch
            train_loss = epoch_loss / max(1, epoch_seen)
            log = on_epoch_end(epoch, train_loss, time.time() - t0)
            log["label"] = label
            history.append(log)
            epoch_loss = 0.0
            epoch_seen = 0
            if epoch >= epochs:
                break

    return history


# ---------------------------------------------------------------------------
# Per-phase training driver
# ---------------------------------------------------------------------------


def train_phase(
    cfg: TrainConfig,
    phase: int,
    *,
    model: YatArithmeticGen | None = None,
    epochs: int | None = None,
    from_ckpt: str | Path | None = None,
    save_to: str | Path | None = None,
) -> tuple[YatArithmeticGen, list[dict]]:
    """Train a single phase of the curriculum.

    Args:
        cfg:        Training config.
        phase:      1, 2, or 3.
        model:      If given, reuse this model; otherwise create a fresh one
                    and optionally restore from ``from_ckpt``.
        epochs:     Overrides ``cfg.epochs_per_phase[phase - 1]``.
        from_ckpt:  Path to load before training.
        save_to:    Path to write the resulting npz; default is
                    ``cfg.ckpt_dir / f"phase{phase}.npz"``.
    """
    spec = PHASES[phase]
    epochs = epochs or cfg.epochs_per_phase[phase - 1]
    save_to = Path(save_to) if save_to else Path(cfg.ckpt_dir) / f"phase{phase}.npz"

    print(f"\n=== phase {phase} — {spec.name} ({epochs} epochs) ===")
    print(f"trainable modules: {sorted(spec.trainable)}")
    print(f"devices: {_device_str()}")

    bins_train = data.load_mnist_binned(cfg.data_root, train=True)
    bins_test = data.load_mnist_binned(cfg.data_root, train=False)
    train_ds = data.build_dataset(bins_train, length=cfg.train_size, seed=cfg.seed + phase, augment_op=True)
    test_ds = data.build_dataset(bins_test, length=cfg.test_size, seed=cfg.seed + 999 + phase, augment_op=False)
    train_loader = data.build_loader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, num_epochs=epochs, seed=cfg.seed + phase
    )
    ocr_metric = OCRMetric(glyphs.render_all_results())

    if model is None:
        rngs = nnx.Rngs(cfg.seed)
        model = YatArithmeticGen(use_yat_encoder=cfg.use_yat_encoder, single_yat=cfg.single_yat, rngs=rngs)
        if from_ckpt is not None:
            print(f"loading prior-phase ckpt: {from_ckpt}")
            load_checkpoint(model, from_ckpt)
    n_params = sum(p.size for p in jax.tree.leaves(nnx.state(model, nnx.Param)))
    print(f"params: {n_params/1e6:.3f} M (only trainable subset gets updates)")

    base_tx = build_base_tx(cfg, epochs)
    tx = freeze_tx(base_tx, model, spec.trainable)
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)

    pix_weight = build_pixel_weight(cfg.loss.tens_weight)
    train_step = make_phase_train_step(phase, pix_weight, cfg.loss)
    eval_step = make_phase_eval_step(phase, pix_weight, cfg.loss)

    steps_per_epoch = cfg.train_size // cfg.batch_size

    def on_epoch_end(epoch: int, train_loss: float, elapsed: float) -> dict:
        # Phase-relevant metrics from the phase's own loss_fn.
        eval_loader = data.build_loader(
            test_ds, batch_size=cfg.batch_size, shuffle=False, num_epochs=1, seed=cfg.seed + 1 + epoch
        )
        phase_metrics = evaluate_phase(model, eval_loader, eval_step, spec.headline_metric)

        log = {
            "phase": phase,
            "epoch": epoch,
            "train_loss": train_loss,
            "test_loss": phase_metrics.get("loss", 0.0),
            spec.headline_metric: phase_metrics.get("headline", 0.0),
            "wall_s": elapsed,
        }

        line = (
            f"phase {phase} epoch {epoch:>2}/{epochs}  "
            f"train_loss={train_loss:.4f}  test_loss={phase_metrics.get('loss', 0.0):.4f}"
        )
        # Phase 1 & 2 print their classification headline (sym_acc / slot_acc).
        # Phase 3 uses OCR instead (computed below) — its headline is a loss, not an accuracy.
        if phase != 3:
            line += f"  {spec.headline_metric}={phase_metrics.get('headline', 0.0) * 100:.2f}%"

        # OCR is only meaningful in phase 3 (decoder has been trained).
        if phase == 3:
            ocr_loader = data.build_loader(
                test_ds, batch_size=cfg.batch_size, shuffle=False, num_epochs=1, seed=cfg.seed + 1001 + epoch
            )
            ocr_metrics = evaluate(model, ocr_loader, ocr_metric)
            log.update(ocr_metrics)
            line += f"  ocr={ocr_metrics['ocr_acc'] * 100:.2f}%"

        line += f"  [{elapsed:.1f}s]"
        print(line)
        return log

    history = _run_epochs(
        model, optimizer, train_step, train_loader,
        epochs=epochs, steps_per_epoch=steps_per_epoch,
        on_epoch_end=on_epoch_end, label=f"phase{phase}",
    )

    save_checkpoint(model, save_to)
    print(f"wrote {save_to}")
    return model, history


# ---------------------------------------------------------------------------
# Full curriculum
# ---------------------------------------------------------------------------


def train_all_phases(cfg: TrainConfig) -> dict:
    """Run phases 1 → 2 → 3 sequentially on a single shared model."""
    rngs = nnx.Rngs(cfg.seed)
    model = YatArithmeticGen(use_yat_encoder=cfg.use_yat_encoder, single_yat=cfg.single_yat, rngs=rngs)
    all_history: list[dict] = []

    for phase in (1, 2, 3):
        _, history = train_phase(cfg, phase, model=model,
                                 epochs=cfg.epochs_per_phase[phase - 1])
        all_history.extend(history)

    save_checkpoint(model, Path(cfg.ckpt_dir) / "model.npz")
    print(f"\nwrote final {Path(cfg.ckpt_dir) / 'model.npz'}")

    return {"history": all_history, "config": {**asdict(cfg), "loss": asdict(cfg.loss)}}


# ---------------------------------------------------------------------------
# Joint (legacy) driver — single pass, all heads at once
# ---------------------------------------------------------------------------


def train_joint(cfg: TrainConfig) -> dict:
    print(f"=== joint training ({cfg.epochs} epochs) ===")
    print(f"devices: {_device_str()}")
    print(f"font:    {glyphs.selected_font_path()}")
    print("config:", json.dumps({**asdict(cfg), "loss": asdict(cfg.loss)}, indent=2))

    bins_train = data.load_mnist_binned(cfg.data_root, train=True)
    bins_test = data.load_mnist_binned(cfg.data_root, train=False)
    train_ds = data.build_dataset(bins_train, length=cfg.train_size, seed=cfg.seed, augment_op=True)
    test_ds = data.build_dataset(bins_test, length=cfg.test_size, seed=cfg.seed + 999, augment_op=False)
    train_loader = data.build_loader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, num_epochs=cfg.epochs, seed=cfg.seed
    )

    rngs = nnx.Rngs(cfg.seed)
    model = YatArithmeticGen(use_yat_encoder=cfg.use_yat_encoder, single_yat=cfg.single_yat, rngs=rngs)
    n_params = sum(p.size for p in jax.tree.leaves(nnx.state(model, nnx.Param)))
    print(f"params: {n_params/1e6:.3f} M")

    tx = build_base_tx(cfg, cfg.epochs)
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)

    pix_weight = build_pixel_weight(cfg.loss.tens_weight)
    train_step = make_joint_train_step(pix_weight, cfg.loss)
    ocr_metric = OCRMetric(glyphs.render_all_results())
    steps_per_epoch = cfg.train_size // cfg.batch_size

    def on_epoch_end(epoch: int, train_loss: float, elapsed: float) -> dict:
        loader = data.build_loader(
            test_ds, batch_size=cfg.batch_size, shuffle=False, num_epochs=1, seed=cfg.seed + 1 + epoch
        )
        eval_metrics = evaluate(model, loader, ocr_metric)
        log = {"epoch": epoch, "train_loss": train_loss, **eval_metrics, "wall_s": elapsed}
        print(
            f"epoch {epoch:>2}/{cfg.epochs}  "
            f"loss={train_loss:.3f}  "
            f"ocr={eval_metrics['ocr_acc'] * 100:.2f}%  "
            f"+={eval_metrics['per_op_acc']['+'] * 100:.1f}  "
            f"-={eval_metrics['per_op_acc']['-'] * 100:.1f}  "
            f"*={eval_metrics['per_op_acc']['*'] * 100:.1f}  "
            f"//={eval_metrics['per_op_acc']['//'] * 100:.1f}  "
            f"[{elapsed:.1f}s]"
        )
        return log

    history = _run_epochs(
        model, optimizer, train_step, train_loader,
        epochs=cfg.epochs, steps_per_epoch=steps_per_epoch,
        on_epoch_end=on_epoch_end, label="joint",
    )

    save_checkpoint(model, Path(cfg.ckpt_dir) / "model.npz")
    print(f"wrote {Path(cfg.ckpt_dir) / 'model.npz'}")
    return {"history": history, "config": {**asdict(cfg), "loss": asdict(cfg.loss)}}


# ---------------------------------------------------------------------------
# Back-compat alias
# ---------------------------------------------------------------------------


def train(cfg: TrainConfig) -> dict:
    """Legacy entry point — equivalent to ``train_joint(cfg)``."""
    return train_joint(cfg)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _parse_phase(value: str) -> str:
    v = value.lower()
    if v in {"1", "2", "3", "all", "joint"}:
        return v
    raise argparse.ArgumentTypeError(f"--phase must be 1/2/3/all/joint, got {value}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--phase", type=_parse_phase, default="all",
                   help="1, 2, 3 (one stage), all (run the full curriculum), "
                        "or joint (legacy single-pass training).")
    p.add_argument("--epochs", type=int, default=15,
                   help="Used by single-phase or --phase joint runs.")
    p.add_argument("--epochs-1", type=int, default=8)
    p.add_argument("--epochs-2", type=int, default=15)
    p.add_argument("--epochs-3", type=int, default=8)
    p.add_argument("--train-size", type=int, default=30_000)
    p.add_argument("--test-size", type=int, default=6_000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--yat-encoder", action="store_true",
                   help="EXPERIMENTAL: use YatConv encoder. Underperforms stock at the same recipe.")
    p.add_argument("--single-yat", action="store_true",
                   help="Collapse the trunk to a single YatNMN layer (h2=None). "
                        "Used for interpretability experiments where prototype↔unit "
                        "mapping needs to be directly observable at the decoder's input.")
    p.add_argument("--from-ckpt", default=None,
                   help="When --phase is 1/2/3, restore params from this npz before training. "
                        "Defaults to ckpts/phase{n-1}.npz when n > 1.")
    p.add_argument("--data-root", default="./data")
    p.add_argument("--ckpt-dir", default="./ckpts")
    args = p.parse_args()

    cfg = TrainConfig(
        epochs=args.epochs,
        epochs_per_phase=(args.epochs_1, args.epochs_2, args.epochs_3),
        train_size=args.train_size,
        test_size=args.test_size,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        use_yat_encoder=args.yat_encoder,
        single_yat=args.single_yat,
        data_root=args.data_root,
        ckpt_dir=args.ckpt_dir,
    )

    if args.phase == "all":
        out = train_all_phases(cfg)
    elif args.phase == "joint":
        out = train_joint(cfg)
    else:
        phase = int(args.phase)
        from_ckpt = args.from_ckpt
        if from_ckpt is None and phase > 1:
            candidate = Path(cfg.ckpt_dir) / f"phase{phase - 1}.npz"
            if candidate.exists():
                from_ckpt = str(candidate)
        _, history = train_phase(cfg, phase, epochs=args.epochs, from_ckpt=from_ckpt)
        out = {"history": history, "config": {**asdict(cfg), "loss": asdict(cfg.loss)}}

    (Path(cfg.ckpt_dir) / "history.json").write_text(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
