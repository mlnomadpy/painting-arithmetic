"""Build painting_arithmetic.ipynb — JAX / Flax NNX / Grain tutorial.

Run:
    python notebooks/build_notebook.py
"""

from __future__ import annotations

import json
import nbformat as nbf
from pathlib import Path

OUT = Path(__file__).resolve().parent / "painting_arithmetic.ipynb"

nb = nbf.v4.new_notebook()
cells: list = []


def md(text: str):
    cells.append(nbf.v4.new_markdown_cell(text.strip("\n")))


def code(src: str):
    cells.append(nbf.v4.new_code_cell(src.strip("\n")))


# =============================================================================

md(r"""
# Painting Arithmetic — JAX / Flax NNX / Grain 🎨

Three image inputs (`img_a`, `img_op`, `img_b`), one image output (28×84). The model has to learn what each operator means visually, do the math in latent space, and **paint** the answer — pixel by pixel — without any softmax over result classes.

This notebook is a full walkthrough. It installs the [`painting_arithmetic`](https://github.com/mlnomadpy/painting-arithmetic) package, downloads MNIST, builds the model end-to-end, trains it for a few minutes on the Kaggle GPU, and opens up the trained latent space with four interpretability views.

> ⚠️ **Kaggle GPU note.** Set the accelerator to **GPU T4 x1** (right sidebar). The P100 SKU is `sm_60`; the default JAX wheel needs `sm_70+`. On CPU this notebook still works; it just takes ~25 min instead of ~3.
""")

md("## 0. Setup")

code(r"""
import sys, subprocess
# Install the painting_arithmetic package + its deps (nmn, grain).
try:
    import painting_arithmetic  # noqa: F401
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
        "git+https://github.com/mlnomadpy/painting-arithmetic.git"])
    import painting_arithmetic  # noqa: F401
""")

code(r"""
import time, json, random
from dataclasses import asdict
from pathlib import Path

import jax, jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from flax import nnx

from painting_arithmetic import data, glyphs, viz
from painting_arithmetic.losses import LossWeights, build_pixel_weight, compute_loss
from painting_arithmetic.model import YatArithmeticGen
from painting_arithmetic.eval import OCRMetric, evaluate
from painting_arithmetic.train import TrainConfig, build_optimizer, make_train_step

print("jax     :", jax.__version__)
print("devices :", jax.devices())
print("font    :", glyphs.selected_font_path())
""")

md(r"""
## 1. The data

We sample `(a, op, b)` triples on the fly. MNIST gives us digit images; we render operator glyphs with augmentation; the target image is a deterministic 28×84 rendering with three slots `[sign | tens | units]`.
""")

code(r"""
# Show one MNIST sample per class.
bins_train = data.load_mnist_binned("./data", train=True)
bins_test  = data.load_mnist_binned("./data", train=False)

fig, axes = plt.subplots(1, 10, figsize=(16, 2.2))
for d, ax in enumerate(axes):
    ax.imshow(bins_test[d][0] * data.MNIST_STD + data.MNIST_MEAN,
              cmap="gray", interpolation="bilinear")
    ax.set_title(str(d), fontsize=14, fontweight="bold")
    ax.axis("off")
plt.suptitle("MNIST — one sample per digit", fontsize=13); plt.tight_layout(); plt.show()
""")

code(r"""
# Operator glyphs — canonical + augmented.
rng = random.Random(0)
fig, axes = plt.subplots(4, 5, figsize=(14, 11))
for i in range(4):
    axes[i, 0].imshow(glyphs.render_glyph_clean(i), cmap="gray", interpolation="bilinear")
    axes[i, 0].set_title(f"{glyphs.OP_GLYPHS[i]}  (canonical)", fontsize=13, fontweight="bold")
    for j in range(1, 5):
        axes[i, j].imshow(glyphs.render_glyph_augmented(i, rng),
                          cmap="gray", interpolation="bilinear")
        axes[i, j].set_title(f"{glyphs.OP_GLYPHS[i]}  (augmented)", fontsize=11)
    for ax in axes[i]: ax.axis("off")
plt.suptitle("operator glyphs", fontsize=13); plt.tight_layout(); plt.show()
""")

code(r"""
# Target images — what the decoder must learn to paint.
samples = [-9, -1, 0, 5, 7, 10, 18, 42, 56, 81]
fig, axes = plt.subplots(len(samples), 1, figsize=(12, len(samples)*0.9))
for ax, r in zip(axes, samples):
    ax.imshow(glyphs.render_result_image(r), cmap="gray",
              aspect="auto", interpolation="bilinear")
    ax.set_ylabel(f"r = {r:>3}", rotation=0, ha="right", va="center",
                  fontsize=12, family="monospace")
    ax.set_xticks([]); ax.set_yticks([])
plt.suptitle("target image per result (28×84, three slots)", fontsize=13, y=1.01)
plt.tight_layout(); plt.show()
""")

md(r"""
## 2. The Yat rational kernel — math first

Each Yat unit computes

$$
y_u = \alpha_u \cdot \frac{(x \cdot W_u + b_u)^2}{\lVert x - W_u \rVert^2 + \varepsilon}.
$$

Two facts make this useful here:

1. **Each $W_u$ is a prototype point in input space.** The denominator goes to zero when $x \to W_u$, so $y_u$ fires when the input is *close to* the prototype.
2. **The numerator additionally rewards alignment.** Each unit has a localised but directional receptive field.

We stack two Yat layers as the trunk, and we replace every conv in the encoder with a `YatConv` rational kernel by default — so the prototype-matching interpretability we use on the trunk also applies to every encoder layer.
""")

md(r"""
## 3. Build the model
""")

code(r"""
USE_YAT_ENCODER = True   # toggle: False reverts to stock Conv+GELU
EMB_DIM, HIDDEN = 64, 256

model = YatArithmeticGen(emb_dim=EMB_DIM, hidden=HIDDEN,
                         use_yat_encoder=USE_YAT_ENCODER,
                         rngs=nnx.Rngs(0))
n_params = sum(p.size for p in jax.tree.leaves(nnx.state(model, nnx.Param)))
print(f"params: {n_params/1e6:.3f} M  ·  encoder: {'YatConv' if USE_YAT_ENCODER else 'Conv+GELU'}")

# Quick sanity forward.
ia = jnp.zeros((2, 1, 28, 28))
out = model(ia, ia, ia)
print("img output shape:", out.img.shape)
""")

md(r"""
## 4. The loss

```
L = BCE(img_pred, target) × tens_pixel_weight   ← headline, BCE with 2× weight on tens column
  + 0.5  · CE(sign-slot, tens-slot, units-slot) ← LOAD-BEARING — the multi-digit fix
  + 0.05 · CE(mod 2, mod 5, mod 11, sign)       ← Chinese-Remainder-Theorem prior
  + 0.2  · CE(sym_a, sym_op, sym_b)             ← shared-encoder regulariser
```

Without the slot-CE block, the decoder defaults to leaving the tens column blank (~72% of training results fit in one digit). Adding it alone jumped accuracy from 62% to 97%.
""")

md(r"""
## 5. Train
""")

code(r"""
# Config — edit these. Default is a ~3-minute run on a Kaggle T4.
cfg = TrainConfig(
    epochs=10,          # bump to 25 for the full paper number (~96.9% OCR)
    train_size=20_000,  # bump to 60_000 for paper number
    test_size=4_000,
    batch_size=256,
    lr=2e-3,
    use_yat_encoder=USE_YAT_ENCODER,
    seed=0,
)
print(json.dumps({**asdict(cfg), "loss": asdict(cfg.loss)}, indent=2))
""")

code(r"""
# Build dataloaders.
train_ds = data.build_dataset(bins_train, length=cfg.train_size, seed=cfg.seed, augment_op=True)
test_ds  = data.build_dataset(bins_test,  length=cfg.test_size,  seed=cfg.seed + 999, augment_op=False)
train_loader = data.build_loader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                                 num_epochs=cfg.epochs, seed=cfg.seed)

# Optimiser + JIT-compiled train step.
tx = build_optimizer(cfg)
optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)
pix_weight = build_pixel_weight(cfg.loss.tens_weight)
train_step = make_train_step(pix_weight, cfg.loss)
ocr_metric = OCRMetric(glyphs.render_all_results())
""")

code(r"""
# Training loop.
steps_per_epoch = cfg.train_size // cfg.batch_size
step = 0; epoch_loss = 0.0; epoch_seen = 0; history = []
t0 = time.time()
for batch in train_loader:
    bj = {k: jnp.asarray(v) for k, v in batch.items()}
    metrics = train_step(model, optimizer, bj)
    epoch_loss += float(metrics["loss"]) * bj["target"].shape[0]
    epoch_seen += bj["target"].shape[0]
    step += 1
    if step % steps_per_epoch == 0:
        epoch = step // steps_per_epoch
        train_loss = epoch_loss / max(1, epoch_seen)
        # Build a fresh test loader for the epoch.
        tl = data.build_loader(test_ds, batch_size=cfg.batch_size, shuffle=False,
                               num_epochs=1, seed=cfg.seed + 1 + epoch)
        m = evaluate(model, tl, ocr_metric)
        elapsed = time.time() - t0
        print(f"epoch {epoch:>2}/{cfg.epochs}  loss={train_loss:.3f}  "
              f"ocr={m['ocr_acc']*100:.2f}%  "
              f"+={m['per_op_acc']['+']*100:.1f}  "
              f"-={m['per_op_acc']['-']*100:.1f}  "
              f"*={m['per_op_acc']['*']*100:.1f}  "
              f"//={m['per_op_acc']['//']*100:.1f}  "
              f"[{elapsed:.1f}s]")
        history.append({"epoch": epoch, "train_loss": train_loss, **m})
        epoch_loss = 0.0; epoch_seen = 0
        if epoch >= cfg.epochs: break
""")

md(r"""
## 6. Does it work? — visualise predictions
""")

code(r"""
# Helper: run model on a chosen (a, op, b).
def predict(model, a, op_idx, b):
    op_arr = (np.asarray(glyphs.render_glyph_clean(op_idx), np.float32) / 255.0)
    op_arr = (op_arr - data.MNIST_MEAN) / data.MNIST_STD
    ia = bins_test[a][0]; ib = bins_test[b][0]
    ia_t = jnp.asarray(ia)[None, None]
    io_t = jnp.asarray(op_arr)[None, None]
    ib_t = jnp.asarray(ib)[None, None]
    out = model(ia_t, io_t, ib_t)
    return np.asarray(out.img)[0, 0]

probes = [(3, 2, 5), (7, 2, 6), (9, 0, 9), (4, 0, 7),
          (8, 1, 3), (2, 3, 4), (6, 2, 7), (5, 0, 8)]
fig, axes = plt.subplots(len(probes), 2, figsize=(14, len(probes)*1.3))
for k, (a, op, b) in enumerate(probes):
    if   glyphs.OPS[op] == "+":  r = a + b
    elif glyphs.OPS[op] == "-":  r = a - b
    elif glyphs.OPS[op] == "*":  r = a * b
    else:                          r = a // b
    pred = predict(model, a, op, b)
    axes[k, 0].imshow(pred, cmap="gray", aspect="auto", interpolation="bilinear")
    axes[k, 0].set_ylabel(f"{a} {glyphs.OP_GLYPHS[op]} {b}",
                          rotation=0, ha="right", va="center",
                          fontsize=14, family="monospace")
    axes[k, 1].imshow(glyphs.render_result_image(r), cmap="gray",
                      aspect="auto", interpolation="bilinear")
    axes[k, 1].set_ylabel(f"= {r}", rotation=0, ha="right", va="center",
                          fontsize=14, family="monospace")
    for ax in axes[k]:
        ax.set_xticks([]); ax.set_yticks([])
axes[0, 0].set_title("model's painted answer", fontsize=13, pad=12)
axes[0, 1].set_title("canonical target",      fontsize=13, pad=12)
plt.tight_layout(); plt.show()
""")

md(r"""
## 7. Interpretability — what is the model doing?
""")

code(r"""
# Collect every trunk vector on the 10×10×4 grid.
grid = viz.collect_trunk_grid(model, bins_test)
print("trunk grid:", grid.trunks.shape, "  results range:", grid.results.min(), grid.results.max())
""")

md(r"""
### 7.1 Trunk PCA — does the model lay results on a value line?
""")

code(r"""
pca = viz.trunk_pca(grid)
coords, vr = pca["coords"], pca["var_ratio"]
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
sc = axes[0].scatter(coords[:, 0], coords[:, 1], c=grid.results, cmap="turbo",
                     s=24, alpha=0.85, edgecolor="none")
axes[0].set_title(f"trunk PCA — coloured by result\nPC1={vr[0]*100:.1f}%  PC2={vr[1]*100:.1f}%")
plt.colorbar(sc, ax=axes[0], label="result")
op_colors = ["#ffb86b", "#a5e075", "#8be9fd", "#ff7a7a"]
for i in range(4):
    m = grid.ops == i
    axes[1].scatter(coords[m, 0], coords[m, 1], c=op_colors[i], s=24, alpha=0.7,
                    edgecolor="none", label=glyphs.OP_GLYPHS[i])
axes[1].legend(); axes[1].set_title("same scatter, coloured by operator")
plt.tight_layout(); plt.show()
""")

md(r"""
### 7.2 Operator-as-shift SVD

For every pair of operators, SVD the per-(a, b) trunk differences. Top-1 var ratio captures "how much of the operator transition is a single direction."
""")

code(r"""
shift = viz.op_as_shift(grid)
fig, axes = plt.subplots(2, 3, figsize=(13, 6.5))
for ax, (lbl, info) in zip(axes.ravel(), shift.items()):
    ax.bar(range(1, 21), info["var_ratio"][:20], color="#8be9fd")
    ax.set_title(f"{lbl}\ntop-1 {info['top1']*100:.1f}%   top-3 {info['top3']*100:.1f}%",
                 fontsize=10)
    ax.set_xlabel("singular rank"); ax.set_ylabel("var ratio")
plt.suptitle("operator-as-shift SVD spectrum", y=1.02); plt.tight_layout(); plt.show()
""")

md(r"""
### 7.3 Decoder unit atlas — what each trunk unit paints
""")

code(r"""
atlas = viz.decoder_unit_atlas(model)
foot, norms = atlas["footprints"], atlas["norms"]
top = np.argsort(-norms)[:16]
fig, axes = plt.subplots(4, 4, figsize=(14, 6))
vmax = float(np.max(np.abs(foot[top])))
for i, u in enumerate(top):
    ax = axes[i // 4, i % 4]
    ax.imshow(foot[u], cmap="seismic", vmin=-vmax, vmax=vmax,
              aspect="equal", interpolation="bilinear")
    ax.set_title(f"unit {int(u)}  ‖Δ‖ = {norms[u]:.2f}", fontsize=10)
    ax.axis("off")
plt.suptitle("decoder unit atlas — top 16 trunk units by footprint magnitude\n"
             "red = paints light pixels · blue = paints dark pixels",
             fontsize=12, y=1.02)
plt.tight_layout(); plt.show()
""")

md(r"""
### 7.4 Yat prototype matching
""")

code(r"""
# Build a symbol library: encoder embeddings for many digits + operator glyphs.
lib_x, lib_lbl = [], []
rng_lib = random.Random(0)
for d in range(10):
    for n in range(min(40, bins_test[d].shape[0])):
        lib_x.append(bins_test[d][n][None, None]); lib_lbl.append(d)
for op_idx in range(4):
    for k in range(20):
        if k < 5: pil = glyphs.render_glyph_clean(op_idx)
        else:     pil = glyphs.render_glyph_augmented(op_idx, rng_lib)
        arr = (np.asarray(pil, np.float32) / 255.0 - data.MNIST_MEAN) / data.MNIST_STD
        lib_x.append(arr[None, None]); lib_lbl.append(glyphs.OP_LABEL_OFFSET + op_idx)
lib_batch = jnp.concatenate([jnp.asarray(x) for x in lib_x], axis=0)
@nnx.jit
def encode(model_, x): return model_.encoder(x)
lib_emb = np.asarray(encode(model, lib_batch))
lib_lbl = np.array(lib_lbl)

win = viz.prototype_winners(model, lib_emb, lib_lbl, k=15)
print(f"{win['n_op_specialised']} of {len(win['win_a'])} h1 units have an operator-class prototype.")
""")

md(r"""
## 8. What to try next

1. **Bump epochs / data.** `cfg.epochs = 25`, `cfg.train_size = 60_000` → ~96.9% OCR (paper number).
2. **Flip the encoder.** Re-run with `USE_YAT_ENCODER = False` and compare numbers.
3. **Steering experiment.** Use `shift[...]["direction"]` to push trunk vectors along the dominant operator-shift axis and watch the painted image drift.
4. **Closed-loop arithmetic.** Train so the decoder's output is in the same distribution as the operand pads, and you have a multi-step world model.

## References

- This notebook's [GitHub repo](https://github.com/mlnomadpy/painting-arithmetic) — README, paper, source code.
- [Goodfire AI — A Geometric Calculator](https://www.goodfire.ai/research/a-geometric-calculator)
- [Nanda et al. — Grokking via mechanistic interpretability](https://arxiv.org/abs/2301.05217) (ICLR 2023)
- [`nmn`](https://github.com/azettaai/nmn) — the Yat / rational kernel library.
""")


# =============================================================================

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.11"},
    "kaggle": {
        "accelerator": "gpu",
        "isInternetEnabled": True,
        "language": "python",
        "sourceType": "notebook",
    },
}
OUT.write_text(json.dumps(nb, indent=1))
print(f"wrote {OUT}  ({len(cells)} cells)")
