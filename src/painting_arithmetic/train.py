"""Training script for the Painting Arithmetic model.

Run from anywhere in the repo:

    python -m painting_arithmetic.train --epochs 15 --train-size 30000

Or programmatically:

    from painting_arithmetic.train import TrainConfig, train
    history = train(TrainConfig(epochs=15))
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from . import data, glyphs
from .eval import OCRMetric, evaluate
from .losses import LossWeights, build_pixel_weight, compute_loss
from .model import YatArithmeticGen


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class TrainConfig:
    epochs: int = 15
    train_size: int = 30_000
    test_size: int = 6_000
    batch_size: int = 256
    lr: float = 2e-3
    weight_decay: float = 1e-4
    warmup_frac: float = 0.05
    seed: int = 0
    # Stock Conv+GELU encoder is the validated path (96.91% OCR at 25
    # epochs × 60k samples). YatConv encoder (use_yat_encoder=True) is
    # experimental and currently underperforms — see model.py docstring.
    use_yat_encoder: bool = False
    loss: LossWeights = field(default_factory=LossWeights)
    data_root: str = "./data"
    ckpt_dir: str = "./ckpts"
    log_every: int = 50

    def steps(self) -> int:
        return self.epochs * (self.train_size // self.batch_size)


# ---------------------------------------------------------------------------
# Optimiser + LR schedule
# ---------------------------------------------------------------------------


def build_optimizer(cfg: TrainConfig) -> optax.GradientTransformation:
    total = cfg.steps()
    warmup = max(1, int(cfg.warmup_frac * total))
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=cfg.lr,
        warmup_steps=warmup,
        decay_steps=total - warmup,
        end_value=cfg.lr * 1e-2,
    )
    return optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=schedule, weight_decay=cfg.weight_decay),
    )


# ---------------------------------------------------------------------------
# JIT-compiled train step
# ---------------------------------------------------------------------------


def make_train_step(pix_weight: jnp.ndarray, loss_weights: LossWeights):
    @nnx.jit(donate_argnames=("optimizer",))
    def train_step(model: YatArithmeticGen, optimizer: nnx.Optimizer, batch):
        def loss_fn(model_):
            out = model_.forward_train(batch["img_a"], batch["img_op"], batch["img_b"])
            return compute_loss(out, batch, loss_weights, pix_weight)

        (_loss, metrics), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
        optimizer.update(model, grads)
        return metrics

    return train_step


# ---------------------------------------------------------------------------
# Training driver
# ---------------------------------------------------------------------------


def _device_str() -> str:
    return ", ".join(str(d) for d in jax.devices())


def train(cfg: TrainConfig) -> dict:
    """Run training and return a history dict."""
    print(f"devices: {_device_str()}")
    print(f"font:    {glyphs.selected_font_path()}")
    print("config:", json.dumps({**asdict(cfg), "loss": asdict(cfg.loss)}, indent=2))

    bins_train = data.load_mnist_binned(cfg.data_root, train=True)
    bins_test = data.load_mnist_binned(cfg.data_root, train=False)

    train_ds = data.build_dataset(bins_train, length=cfg.train_size, seed=cfg.seed, augment_op=True)
    test_ds = data.build_dataset(
        bins_test, length=cfg.test_size, seed=cfg.seed + 999, augment_op=False
    )
    train_loader = data.build_loader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, num_epochs=cfg.epochs, seed=cfg.seed
    )
    test_loader = data.build_loader(
        test_ds, batch_size=cfg.batch_size, shuffle=False, num_epochs=1, seed=cfg.seed
    )

    rngs = nnx.Rngs(cfg.seed)
    model = YatArithmeticGen(use_yat_encoder=cfg.use_yat_encoder, rngs=rngs)
    n_params = sum(p.size for p in jax.tree.leaves(nnx.state(model, nnx.Param)))
    print(f"params:  {n_params/1e6:.3f} M")

    tx = build_optimizer(cfg)
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)

    pix_weight = build_pixel_weight(cfg.loss.tens_weight)
    train_step = make_train_step(pix_weight, cfg.loss)
    ocr_metric = OCRMetric(glyphs.render_all_results())

    history: list[dict] = []
    t0 = time.time()
    steps_per_epoch = cfg.train_size // cfg.batch_size
    step = 0
    epoch_loss = 0.0
    epoch_seen = 0

    for batch in train_loader:
        batch_jax = {k: jnp.asarray(v) for k, v in batch.items()}
        metrics = train_step(model, optimizer, batch_jax)
        epoch_loss += float(metrics["loss"]) * batch_jax["target"].shape[0]
        epoch_seen += batch_jax["target"].shape[0]
        step += 1

        # End of epoch — evaluate.
        if step % steps_per_epoch == 0:
            epoch = step // steps_per_epoch
            train_loss = epoch_loss / max(1, epoch_seen)
            eval_metrics = evaluate(model, test_loader, ocr_metric)
            elapsed = time.time() - t0
            log = {
                "epoch": epoch,
                "step": step,
                "train_loss": train_loss,
                **eval_metrics,
                "wall_s": elapsed,
            }
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
            history.append(log)
            # Rebuild the test loader for the next epoch.
            test_loader = data.build_loader(
                test_ds, batch_size=cfg.batch_size, shuffle=False, num_epochs=1, seed=cfg.seed + 1 + epoch
            )
            epoch_loss = 0.0
            epoch_seen = 0
            if epoch >= cfg.epochs:
                break

    # Save final checkpoint as a numpy archive (no Orbax required for this size).
    Path(cfg.ckpt_dir).mkdir(parents=True, exist_ok=True)
    state = nnx.state(model, nnx.Param)
    flat = nnx.to_flat_state(state)
    np.savez(
        Path(cfg.ckpt_dir) / "model.npz",
        **{".".join(map(str, k)): np.asarray(v.value) for k, v in flat},
    )
    print(f"wrote {Path(cfg.ckpt_dir) / 'model.npz'}  ({n_params/1e6:.3f} M params)")

    return {"history": history, "config": {**asdict(cfg), "loss": asdict(cfg.loss)},
            "n_params": int(n_params)}


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--train-size", type=int, default=30_000)
    p.add_argument("--test-size", type=int, default=6_000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--yat-encoder", action="store_true",
                   help="EXPERIMENTAL: use YatConv encoder instead of stock "
                        "Conv+GELU. Currently underperforms (~29%% OCR vs "
                        "~97%% with the stock encoder at the same recipe).")
    p.add_argument("--data-root", default="./data")
    p.add_argument("--ckpt-dir", default="./ckpts")
    args = p.parse_args()

    cfg = TrainConfig(
        epochs=args.epochs,
        train_size=args.train_size,
        test_size=args.test_size,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        use_yat_encoder=args.yat_encoder,
        data_root=args.data_root,
        ckpt_dir=args.ckpt_dir,
    )
    out = train(cfg)
    (Path(cfg.ckpt_dir) / "history.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
