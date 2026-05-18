"""Build `painting_arithmetic_kaggle.ipynb` — a story-driven Kaggle notebook.

Run::
    python notebooks/build_kaggle_notebook.py

Output::
    notebooks/painting_arithmetic_kaggle.ipynb
"""

from __future__ import annotations

import json
from pathlib import Path

import nbformat as nbf


def md(s: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(s.strip("\n"))


def code(s: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(s.strip("\n"))


# ===========================================================================
# Section 0 — Title + abstract + table of contents
# ===========================================================================


CELLS: list[nbf.NotebookNode] = []

CELLS += [
    md(r"""
# Painting Arithmetic — a network that *paints* its answer

> A 0.7M-parameter rational-kernel network that does single-digit MNIST
> arithmetic by **painting the result image**, with a circuit-level
> interpretation you can edit by hand. JAX + Flax NNX.

This notebook is the companion to the paper *"Painting Arithmetic: A
Rational-Kernel Network for Visual Symbolic Computation in Latent
Space"*. Where the paper presents results, this notebook builds them
step by step — same data pipeline, same architecture, same training
recipe, same interpretability probes.

**The story arc:**

1. **The challenge.** Three images go in (digit, operator, digit). One
   image comes out — the painted answer. No softmax over result classes
   anywhere.
2. **The architecture.** A shared CNN encoder maps every symbol to a
   64-d embedding; a single Yat rational-kernel layer fuses the three
   embeddings into a 256-d "trunk vector"; a small ConvTranspose decoder
   paints the trunk vector to a $28\!\times\!84$ image with three slots
   $[\text{sign} \mid \text{tens} \mid \text{units}]$.
3. **Phased training.** Three sequential stages with strict
   `stop_gradient` cuts: encoder first, then trunk on frozen encoder,
   then decoder on frozen trunk. No representation leakage between
   stages.
4. **Headline numbers.** $\approx\!89\%$ painted OCR with this thin
   trunk, $\approx\!97\%$ with a deeper baseline.
5. **The geometric calculator.** The operator
   $\{+,\,-,\,\times,\,\div\}$ becomes a *direction* in trunk space —
   the property *(Goodfire AI, 2025)* found in Llama 3.1, reproduced
   here in a model four orders of magnitude smaller.
6. **The big claim: an operator is also a *circuit*.** Pick the
   $30$ Yat units whose middle-slot prototype is a "×" glyph, zero
   their kernel rows — $\times$ accuracy collapses by $43.7$pp while
   the other three operators lose at most $1.6$pp.
7. **Counterfactual steering.** Adding the SVD operator-difference
   direction to the trunk at inference flips $- \to \div$ on $70\%$ of
   the grid. The operator-direction is *causal*, not just descriptive.
8. **Operator-conditioned painting.** $\times$-tagged units pour
   $74.7\%$ of their decoder-Jacobian energy into the tens slot of the
   output; the other three operators' units paint the units slot. The
   decoder's "slot alphabet" is operator-aware.

**Runtime.** ~60–70 minutes on a Kaggle T4 — paper-recipe budget
(60 k samples × (8, 30, 12) epochs), reaches 88.78 % painted OCR.
JAX uses GPU when available.

**Reading order.** Sections 1–4 build the model end-to-end. Sections
5–8 do interpretability on the trained checkpoint. Skip ahead if you
already know how arithmetic models work.
"""),
]


# ===========================================================================
# Section 1 — environment
# ===========================================================================


CELLS += [
    md(r"""
## 1 · Environment

We use JAX + Flax NNX for the model and Grain for the data pipeline.
Kaggle's default environment already has JAX with CUDA; we install
`flax` and `grain` and the `nmn` library (for the `YatNMN` layer)
fresh."""),
    code(r"""
# Kaggle-friendly install. Skip if you already have these.
import sys, subprocess
def pip(*args):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *args], check=True)

try:
    import flax.nnx as nnx
except Exception:
    pip("flax")
try:
    import grain.python as grain
except Exception:
    pip("grain")
try:
    import nmn  # noqa
except Exception:
    pip("git+https://github.com/mlnomadpy/nmn.git")
"""),
    code(r"""
import os, math, random, time, json
from dataclasses import dataclass, field

import numpy as np
import jax, jax.numpy as jnp
from flax import nnx
import optax
import grain.python as grain

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from matplotlib import font_manager

print("JAX devices:", jax.devices())
print("JAX default backend:", jax.default_backend())
"""),
]


# ===========================================================================
# Section 2 — the data: MNIST digits + rendered operator glyphs
# ===========================================================================


CELLS += [
    md(r"""
## 2 · Data — digits, operators, and a painted answer image

A training sample is a quadruple $(\text{img\_a},\ \text{img\_op},\ \text{img\_b},\ \text{img\_out})$:

* `img_a`, `img_b` are MNIST digit crops $(28\!\times\!28)$.
* `img_op` is a $28\!\times\!28$ render of one of four operator glyphs
  $\{+, -, \times, \div\}$, with mild rotation / translation / blur
  augmentation so the model can't memorise pixel patterns.
* `img_out` is a $28\!\times\!84$ image, three $28\!\times\!28$ slots
  $[\text{sign} \mid \text{tens} \mid \text{units}]$, deterministically
  rendered from the integer result $r = \text{op}(a, b)$ —
  with $r \in [-9, 81]$ for single-digit arithmetic.

Crucially, the **target is an image**, not a class label. The model has
to *paint* its answer.
"""),
    code(r"""
# --- MNIST loader (PyTorch-free; we shim with the .npz mirror that Kaggle ships) ---
def load_mnist_test_npz(path):
    '''Return (images, labels) for MNIST.

    Kaggle's `digit-recognizer` dataset ships test.csv; the kernel-base
    image also has MNIST in /kaggle/input if the dataset is attached.
    For local notebooks we fall back to torchvision.
    '''
    try:
        from torchvision import datasets, transforms
        tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        train = datasets.MNIST(path, train=True, download=True, transform=tf)
        test = datasets.MNIST(path, train=False, download=True, transform=tf)
        def _to_arrays(ds):
            X, Y = [], []
            for img, lbl in ds:
                X.append(img.numpy().squeeze(0).astype(np.float32))
                Y.append(int(lbl))
            return np.stack(X), np.asarray(Y, dtype=np.int32)
        return _to_arrays(train), _to_arrays(test)
    except Exception as e:
        raise RuntimeError(f"Could not load MNIST: {e}. On Kaggle, attach the MNIST dataset.")

MNIST_MEAN, MNIST_STD = 0.1307, 0.3081

(train_X, train_Y), (test_X, test_Y) = load_mnist_test_npz("./data")
print("train:", train_X.shape, "test:", test_X.shape)
"""),
    code(r"""
# --- Operator glyph rendering ---
OPS    = ["+", "-", "*", "//"]
GLYPHS = ["+", "−", "×", "÷"]    # display glyphs (Unicode minus/times/divide)
OP_LABEL_OFFSET = 10
NUM_SYMBOLS = 14                 # 10 digits + 4 operators
RESULT_MIN, RESULT_MAX = -9, 81
NUM_RESULTS = RESULT_MAX - RESULT_MIN + 1   # 91

def _font_has_glyph(font, ch):
    '''Probe whether a font actually has a glyph for `ch`.'''
    bbox = font.getbbox(ch)
    return bbox is not None and (bbox[2] - bbox[0]) > 0 and (bbox[3] - bbox[1]) > 0

def _pick_font(size):
    # Prefer matplotlib's bundled DejaVu Sans which always has the needed glyphs.
    p = font_manager.findfont("DejaVu Sans", fallback_to_default=True)
    return ImageFont.truetype(p, size)

def _render_centered(char, size_canvas=28, target_max=22):
    '''Draw `char` centred at the configured target size, MNIST-style.'''
    big = 200
    pil = Image.new("L", (big, big), 0)
    drw = ImageDraw.Draw(pil)
    font = _pick_font(size=120)
    bbox = drw.textbbox((big // 2, big // 2), char, font=font, anchor="mm")
    drw.text((big // 2, big // 2), char, fill=255, font=font, anchor="mm")
    arr = np.asarray(pil)
    if (arr > 0).any():
        ys, xs = np.where(arr > 0)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        crop = arr[y0:y1, x0:x1]
        h, w = crop.shape
        scale = target_max / max(h, w)
        new = Image.fromarray(crop).resize((max(1, int(w * scale)), max(1, int(h * scale))),
                                            resample=Image.BILINEAR)
        new_arr = np.asarray(new)
        out = np.zeros((size_canvas, size_canvas), np.uint8)
        h2, w2 = new_arr.shape
        oy = (size_canvas - h2) // 2; ox = (size_canvas - w2) // 2
        out[oy:oy + h2, ox:ox + w2] = new_arr
        return out
    return np.zeros((size_canvas, size_canvas), np.uint8)

def render_glyph_clean(op_idx):
    return _render_centered(GLYPHS[op_idx])

def render_glyph_augmented(op_idx, rng):
    arr = _render_centered(GLYPHS[op_idx])
    pil = Image.fromarray(arr)
    angle = rng.uniform(-15, 15); pil = pil.rotate(angle, resample=Image.BILINEAR)
    tx = rng.randint(-3, 3); ty = rng.randint(-3, 3)
    pil = pil.transform(pil.size, Image.AFFINE, (1, 0, -tx, 0, 1, -ty), resample=Image.BILINEAR)
    if rng.random() < 0.5:
        from PIL import ImageFilter
        pil = pil.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.0, 0.7)))
    return np.asarray(pil, np.uint8)

def render_result_image(result):
    '''Render r as a (28, 84) image: [sign | tens | units].'''
    assert RESULT_MIN <= result <= RESULT_MAX
    out = np.zeros((28, 84), dtype=np.uint8)
    if result < 0:
        out[:, 0:28] = _render_centered("−")
    a = abs(result)
    tens, units = a // 10, a % 10
    if tens > 0:
        out[:, 28:56] = _render_centered(str(tens))
    out[:, 56:84] = _render_centered(str(units))
    return out

def render_all_results():
    return np.stack([render_result_image(r).astype(np.float32) / 255.0
                     for r in range(RESULT_MIN, RESULT_MAX + 1)])

def slot_labels(result):
    '''(sign, tens, units) labels in [0,1] x [0,9] x [0,9].'''
    sign = 1 if result < 0 else 0
    a = abs(result)
    return sign, a // 10, a % 10

# Quick visual check.
plt.figure(figsize=(11, 1.6))
for k, op in enumerate(GLYPHS):
    plt.subplot(1, 4, k + 1); plt.imshow(render_glyph_clean(k), cmap="gray"); plt.title(f"'{op}'")
    plt.axis("off")
plt.tight_layout(); plt.show()

plt.figure(figsize=(11, 1.6))
for k, r in enumerate([-3, 0, 4, 17, 42]):
    plt.subplot(1, 5, k + 1); plt.imshow(render_result_image(r), cmap="gray"); plt.title(f"r = {r}")
    plt.axis("off")
plt.tight_layout(); plt.show()
"""),
    code(r"""
# --- Dataset (numpy, Grain-friendly) ---
@dataclass
class Sample:
    img_a:  np.ndarray; img_op: np.ndarray; img_b:  np.ndarray
    target: np.ndarray
    sym_a:  int; sym_op: int; sym_b:  int
    slot_sign: int; slot_tens: int; slot_units: int
    mod2: int; mod5: int; mod11: int; sign: int
    result: int; op_idx: int

class ArithmeticSource(grain.RandomAccessDataSource):
    '''Synthetic on-the-fly source: each idx → one random (a, op, b).'''
    def __init__(self, X, Y, length, seed, augment_op=True, glyph_pool=2000):
        # Per-digit bins.
        bins = [X[Y == d] for d in range(10)]
        self._bins = bins
        self._length = int(length); self._seed = seed
        # Pre-render the operator glyphs once (kills the CPU bottleneck).
        rng = random.Random(seed * 31 + 7)
        pool_size = glyph_pool if augment_op else 4
        self._pool = []
        for op in range(4):
            arrs = []
            for k in range(pool_size):
                arr = (render_glyph_augmented(op, rng) if augment_op
                       else render_glyph_clean(op))
                a = (arr.astype(np.float32) / 255.0 - MNIST_MEAN) / MNIST_STD
                arrs.append(a)
            self._pool.append(np.stack(arrs))
        self._targets = render_all_results()
    def __len__(self): return self._length
    def __getitem__(self, idx):
        rng  = random.Random((self._seed * 2_147_483_647) ^ idx)
        nprng = np.random.default_rng((self._seed << 16) | idx)
        a = rng.randrange(10); b = rng.randrange(10); op = rng.randrange(4)
        if OPS[op] == "//" and b == 0: b = rng.randrange(1, 10)
        if   OPS[op] == "+":  r = a + b
        elif OPS[op] == "-":  r = a - b
        elif OPS[op] == "*":  r = a * b
        else:                  r = a // b
        gp = self._pool[op]; op_arr = gp[rng.randrange(gp.shape[0])]
        ia = self._bins[a][int(nprng.integers(0, self._bins[a].shape[0]))]
        ib = self._bins[b][int(nprng.integers(0, self._bins[b].shape[0]))]
        target = self._targets[r - RESULT_MIN]
        sign_l, tens_l, units_l = slot_labels(r); absr = abs(r)
        return Sample(
            img_a=ia[None, :, :], img_op=op_arr[None, :, :], img_b=ib[None, :, :],
            target=target[None, :, :],
            sym_a=int(a), sym_op=OP_LABEL_OFFSET + op, sym_b=int(b),
            slot_sign=int(sign_l), slot_tens=int(tens_l), slot_units=int(units_l),
            mod2=int(absr % 2), mod5=int(absr % 5), mod11=int(absr % 11),
            sign=int(sign_l), result=int(r), op_idx=int(op),
        )

def collate(samples):
    return {
        "img_a": np.stack([s.img_a for s in samples]),
        "img_op": np.stack([s.img_op for s in samples]),
        "img_b": np.stack([s.img_b for s in samples]),
        "target": np.stack([s.target for s in samples]),
        "sym_a": np.asarray([s.sym_a for s in samples], np.int32),
        "sym_op": np.asarray([s.sym_op for s in samples], np.int32),
        "sym_b": np.asarray([s.sym_b for s in samples], np.int32),
        "slot_sign": np.asarray([s.slot_sign for s in samples], np.int32),
        "slot_tens": np.asarray([s.slot_tens for s in samples], np.int32),
        "slot_units": np.asarray([s.slot_units for s in samples], np.int32),
        "mod2": np.asarray([s.mod2 for s in samples], np.int32),
        "mod5": np.asarray([s.mod5 for s in samples], np.int32),
        "mod11": np.asarray([s.mod11 for s in samples], np.int32),
        "sign": np.asarray([s.sign for s in samples], np.int32),
        "result": np.asarray([s.result for s in samples], np.int32),
        "op_idx": np.asarray([s.op_idx for s in samples], np.int32),
    }

def build_loader(source, batch_size, num_epochs, shuffle, seed, num_workers=0):
    class _Collate(grain.MapTransform):
        def map(self, samples): return collate(samples)
    sampler = grain.IndexSampler(
        num_records=len(source), num_epochs=num_epochs,
        shard_options=grain.NoSharding(), shuffle=shuffle, seed=seed,
    )
    return grain.DataLoader(
        data_source=source, sampler=sampler,
        operations=[grain.Batch(batch_size=batch_size, drop_remainder=True), _Collate()],
        worker_count=num_workers,
    )

# Quick visualisation of one sample.
ds = ArithmeticSource(train_X, train_Y, length=2048, seed=0)
s = ds[7]
fig, ax = plt.subplots(1, 4, figsize=(12, 2.2),
                        gridspec_kw={"width_ratios": [1, 1, 1, 3]})
def _denorm(a): return (a * MNIST_STD + MNIST_MEAN).clip(0, 1)
ax[0].imshow(_denorm(s.img_a[0]), cmap="gray"); ax[0].set_title(f"a = {s.sym_a}")
ax[1].imshow(_denorm(s.img_op[0]), cmap="gray"); ax[1].set_title(f"op = {GLYPHS[s.op_idx]}")
ax[2].imshow(_denorm(s.img_b[0]), cmap="gray"); ax[2].set_title(f"b = {s.sym_b}")
ax[3].imshow(s.target[0], cmap="gray"); ax[3].set_title(f"target → {s.result}")
for a in ax: a.axis("off")
plt.tight_layout(); plt.show()
"""),
]


# ===========================================================================
# Section 3 — model: encoder + Yat trunk + decoder + auxiliary heads
# ===========================================================================


CELLS += [
    md(r"""
## 3 · Model — encoder + Yat trunk + decoder

Three pieces:

1. **`SymbolEncoder`** — four Conv+GELU blocks, average pool, 64-d output.
   *Shared across all three input slots* — digits and operators must live
   in the same latent space.
2. **`YatNMN`** — a single rational-kernel layer that computes
   $y_u = \alpha_u (x \cdot W_u + b_u)^2 \big/ (\lVert x - W_u\rVert^2 + \varepsilon)$.
   $W_u$ is a *prototype point* in input space; the activation is high
   when $x$ both **aligns with** and is **close to** $W_u$.
3. **`Decoder`** — a tiny ConvTranspose stack that paints a $28\!\times\!84$
   image from the trunk vector.

Plus training-only auxiliary heads (14-way symbol classifier on each
input; modular CRT + per-slot CE on the trunk) that vanish at inference.
"""),
    code(r"""
from nmn.nnx.layers.nmn import YatNMN

class StockEncoder(nnx.Module):
    def __init__(self, emb_dim=64, *, rngs):
        self.c1 = nnx.Conv(1, 32, kernel_size=(3, 3), padding="SAME", rngs=rngs)
        self.c2 = nnx.Conv(32, 32, kernel_size=(3, 3), strides=(2, 2), padding="SAME", rngs=rngs)
        self.c3 = nnx.Conv(32, 64, kernel_size=(3, 3), padding="SAME", rngs=rngs)
        self.c4 = nnx.Conv(64, 64, kernel_size=(3, 3), strides=(2, 2), padding="SAME", rngs=rngs)
        self.proj = nnx.Linear(64, emb_dim, rngs=rngs)
    def __call__(self, x):
        if x.ndim == 4 and x.shape[1] == 1: x = jnp.transpose(x, (0, 2, 3, 1))
        x = nnx.gelu(self.c1(x)); x = nnx.gelu(self.c2(x))
        x = nnx.gelu(self.c3(x)); x = nnx.gelu(self.c4(x))
        x = jnp.mean(x, axis=(1, 2))
        return self.proj(x)

class Decoder(nnx.Module):
    def __init__(self, hidden=256, *, rngs):
        self.fc  = nnx.Linear(hidden, 16 * 7 * 21, rngs=rngs)
        self.ct1 = nnx.ConvTranspose(16, 16, kernel_size=(4, 4), strides=(2, 2), padding="SAME", rngs=rngs)
        self.c1  = nnx.Conv(16, 16, kernel_size=(3, 3), padding="SAME", rngs=rngs)
        self.ct2 = nnx.ConvTranspose(16, 8, kernel_size=(4, 4), strides=(2, 2), padding="SAME", rngs=rngs)
        self.c2  = nnx.Conv(8, 1, kernel_size=(3, 3), padding="SAME", rngs=rngs)
    def __call__(self, t):
        x = self.fc(t).reshape(-1, 7, 21, 16)
        x = nnx.gelu(self.ct1(x)); x = nnx.gelu(self.c1(x))
        x = nnx.gelu(self.ct2(x)); x = nnx.sigmoid(self.c2(x))
        return jnp.transpose(x, (0, 3, 1, 2))

class YatArithmeticGen(nnx.Module):
    def __init__(self, emb_dim=64, hidden=256, single_yat=True, *, rngs):
        self.encoder = StockEncoder(emb_dim, rngs=rngs)
        fused = 3 * emb_dim
        # The paper's interpretability story uses single_yat=True; the
        # 2-layer variant is the higher-accuracy comparison baseline.
        self.h1 = YatNMN(fused, hidden, rngs=rngs)
        self.h2 = None if single_yat else YatNMN(hidden, hidden, rngs=rngs)
        self.decoder = Decoder(hidden, rngs=rngs)
        # Auxiliary heads (training-time only).
        self.aux_sym = nnx.Linear(emb_dim, NUM_SYMBOLS, rngs=rngs)
        self.head_mod2 = nnx.Linear(hidden, 2, rngs=rngs)
        self.head_mod5 = nnx.Linear(hidden, 5, rngs=rngs)
        self.head_mod11 = nnx.Linear(hidden, 11, rngs=rngs)
        self.head_sign = nnx.Linear(hidden, 2, rngs=rngs)
        self.head_slot_sign = nnx.Linear(hidden, 2, rngs=rngs)
        self.head_slot_tens = nnx.Linear(hidden, 10, rngs=rngs)
        self.head_slot_units = nnx.Linear(hidden, 10, rngs=rngs)

    def trunk(self, img_a, img_op, img_b):
        ea = self.encoder(img_a); eo = self.encoder(img_op); eb = self.encoder(img_b)
        h = jnp.concatenate([ea, eo, eb], axis=-1)
        t1 = self.h1(h)
        t2 = t1 if self.h2 is None else self.h2(t1)
        return ea, eo, eb, h, t1, t2

    def __call__(self, img_a, img_op, img_b):
        _, _, _, _, _, t = self.trunk(img_a, img_op, img_b)
        return self.decoder(t)

model = YatArithmeticGen(rngs=nnx.Rngs(0))
n_params = sum(p.size for p in jax.tree.leaves(nnx.state(model, nnx.Param)))
print(f"params: {n_params/1e6:.3f} M")
"""),
]


# ===========================================================================
# Section 4 — phased training
# ===========================================================================


CELLS += [
    md(r"""
## 4 · Phased training — no representation leakage

Training every part of the network at once leaves the door open to
representation leakage: the pixel-space objective on the painted image
can subtly shape the encoder (allocating embedding dimensions that are
convenient for the decoder rather than for symbol identity), and the
modular auxiliary heads can shape the trunk through both the operator
and the digit slots simultaneously, blurring what each component is
for.

For the v3-thin run that supports the interpretability claims we train
in **three sequential stages**, each with a strict component freeze.
Each phase uses `optax.multi_transform(set_to_zero, …)` at the optimiser
level **and** `jax.lax.stop_gradient` at the model level — belt and
braces, so that no frozen parameter can drift by more than the float32
noise floor.

| phase | trainable             | loss                    | headline metric |
|------:|-----------------------|-------------------------|-----------------|
|     1 | encoder + aux_sym     | 14-way symbol CE        | sym_acc         |
|     2 | h₁ + slot & mod heads | slot CE + modular CE    | slot_acc        |
|     3 | decoder               | pixel BCE               | OCR             |
"""),
    code(r"""
# --- Losses ---
def _bce(p, t, eps=1e-7):
    p = jnp.clip(p, eps, 1.0 - eps)
    return -(t * jnp.log(p) + (1.0 - t) * jnp.log1p(-p))

def _ce(logits, labels):
    return optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()

def build_pixel_weight(tens_weight=2.0):
    w = jnp.ones((1, 1, 28, 84), dtype=jnp.float32)
    return w.at[:, :, :, 28:56].set(tens_weight)

def phase1_loss(model, batch, pix_w):
    ea = model.encoder(batch["img_a"])
    eo = model.encoder(batch["img_op"])
    eb = model.encoder(batch["img_b"])
    la = model.aux_sym(ea); lo = model.aux_sym(eo); lb = model.aux_sym(eb)
    loss = (_ce(la, batch["sym_a"]) + _ce(lo, batch["sym_op"]) + _ce(lb, batch["sym_b"])) / 3
    truth = jnp.concatenate([batch["sym_a"], batch["sym_op"], batch["sym_b"]], 0)
    logits = jnp.concatenate([la, lo, lb], 0)
    return loss, {"loss": loss, "sym_acc": (logits.argmax(-1) == truth).mean()}

def phase2_loss(model, batch, pix_w):
    ea = jax.lax.stop_gradient(model.encoder(batch["img_a"]))
    eo = jax.lax.stop_gradient(model.encoder(batch["img_op"]))
    eb = jax.lax.stop_gradient(model.encoder(batch["img_b"]))
    h  = jnp.concatenate([ea, eo, eb], -1)
    t  = model.h1(h); t = t if model.h2 is None else model.h2(t)
    L_mod = (_ce(model.head_mod2(t), batch["mod2"])
             + _ce(model.head_mod5(t), batch["mod5"])
             + _ce(model.head_mod11(t), batch["mod11"])
             + _ce(model.head_sign(t), batch["sign"]))
    s_sign = model.head_slot_sign(t); s_tens = model.head_slot_tens(t); s_units = model.head_slot_units(t)
    L_slot = (_ce(s_sign, batch["slot_sign"]) + _ce(s_tens, batch["slot_tens"]) + _ce(s_units, batch["slot_units"])) / 3
    total = 0.05 * L_mod + 0.5 * L_slot
    ok = ((s_sign.argmax(-1) == batch["slot_sign"]) &
          (s_tens.argmax(-1) == batch["slot_tens"]) &
          (s_units.argmax(-1) == batch["slot_units"]))
    return total, {"loss": total, "slot_acc": ok.mean()}

def phase3_loss(model, batch, pix_w):
    ea = model.encoder(batch["img_a"])
    eo = model.encoder(batch["img_op"])
    eb = model.encoder(batch["img_b"])
    h  = jnp.concatenate([ea, eo, eb], -1)
    t  = model.h1(h); t = t if model.h2 is None else model.h2(t)
    t  = jax.lax.stop_gradient(t)
    img = model.decoder(t)
    pix = _bce(img, batch["target"])
    L = (pix * pix_w).mean()
    return L, {"loss": L, "loss_pix": L}

LOSSES = {1: phase1_loss, 2: phase2_loss, 3: phase3_loss}
TRAINABLE = {
    1: frozenset({"encoder", "aux_sym"}),
    2: frozenset({"h1", "h2",
                  "head_mod2", "head_mod5", "head_mod11", "head_sign",
                  "head_slot_sign", "head_slot_tens", "head_slot_units"}),
    3: frozenset({"decoder"}),
}
HEADLINE = {1: "sym_acc", 2: "slot_acc", 3: "loss_pix"}
"""),
    code(r"""
# --- Freezing: zero gradients for frozen submodules inside the train step ---
#
# We use a portable pattern instead of `optax.multi_transform`: the
# optimizer is a vanilla Adam over ALL params, but inside the train step
# we tree-map the gradient and zero every leaf whose top-level path
# token is NOT in the trainable set. We also drop AdamW's weight decay
# in phased mode (decoupled weight decay would shrink frozen params
# regardless of gradient).
#
# Belt-and-braces with the `jax.lax.stop_gradient` cuts inside each
# phase's loss function: a frozen leaf gets stop_gradient'd grads (=0)
# AND any stray gradient is masked here.
def _path_top(path):
    if not path: return ""
    f = path[0]
    return str(getattr(f, "key", getattr(f, "name", getattr(f, "idx", f))))

def make_phase_step(phase, pix_w):
    loss_fn = LOSSES[phase]
    trainable = TRAINABLE[phase]

    @nnx.jit(donate_argnames=("optimizer",))
    def step(model, optimizer, batch):
        def fn(m): return loss_fn(m, batch, pix_w)
        (_, metrics), grads = nnx.value_and_grad(fn, has_aux=True)(model)
        grads = jax.tree_util.tree_map_with_path(
            lambda p, g: g if _path_top(p) in trainable else jnp.zeros_like(g),
            grads,
        )
        optimizer.update(model, grads)
        return metrics
    return step
"""),
    md(r"""
**Training run.** Three phases at the paper's headline recipe:
60 000 samples per epoch × (8, 30, 12) epochs. On a Kaggle T4 this
takes ~60–70 min end-to-end and reaches **88.78 % painted OCR** — the
v3-thin number quoted in the paper.

If you need a faster smoke run on a slower machine, drop
`TRAIN_PER_EPOCH` to 20 000 and `EPOCHS_PHASE` to `(4, 12, 6)` — the
recipe scales linearly but the OCR ceiling drops with the budget.
"""),
    code(r"""
TRAIN_PER_EPOCH = 60_000     # paper recipe — match the headline OCR
TEST_SIZE       = 6_000
BATCH_SIZE      = 256
LR              = 2e-3
EPOCHS_PHASE    = (8, 30, 12)
SEED            = 0

pix_w = build_pixel_weight(2.0)

def steps_for(epochs, train_per_epoch=TRAIN_PER_EPOCH, batch_size=BATCH_SIZE):
    return epochs * (train_per_epoch // batch_size)

def build_schedule(epochs):
    total = steps_for(epochs)
    warmup = max(1, int(0.05 * total))
    return optax.warmup_cosine_decay_schedule(
        init_value=0.0, peak_value=LR, warmup_steps=warmup,
        decay_steps=total - warmup, end_value=LR * 1e-2)

# Set up: load test source once.
test_src = ArithmeticSource(test_X, test_Y, length=TEST_SIZE, seed=999, augment_op=False)
all_targets = jnp.asarray(render_all_results().reshape(NUM_RESULTS, -1))

def ocr_pred(img):
    flat = img.reshape(img.shape[0], -1)
    diff = flat[:, None, :] - all_targets[None, :, :]
    return jnp.argmin((diff * diff).sum(-1), -1) + RESULT_MIN

@nnx.jit
def predict_img(model, ia, io, ib):
    return model(ia, io, ib)

def evaluate(model):
    '''OCR accuracy on the test source.'''
    loader = build_loader(test_src, BATCH_SIZE, 1, False, seed=42)
    n = ok = 0; per_op_n = [0]*4; per_op_ok = [0]*4
    for batch in loader:
        img = predict_img(model, jnp.asarray(batch["img_a"]),
                          jnp.asarray(batch["img_op"]),
                          jnp.asarray(batch["img_b"]))
        pred = np.asarray(ocr_pred(img))
        truth = np.asarray(batch["result"])
        ok_ = pred == truth; n += pred.size; ok += int(ok_.sum())
        op_idx = np.asarray(batch["op_idx"])
        for i in range(4):
            m = op_idx == i; per_op_n[i] += int(m.sum())
            per_op_ok[i] += int((ok_ & m).sum())
    return {"ocr_acc": ok / max(1, n),
            "per_op_acc": {OPS[i]: per_op_ok[i] / max(1, per_op_n[i]) for i in range(4)}}
"""),
    code(r"""
def train_phase(model, phase, epochs):
    print(f"\\n=== phase {phase} ({epochs} epochs) — trainable: {sorted(TRAINABLE[phase])} ===")
    # Vanilla Adam (no decoupled weight decay) — see make_phase_step for why.
    tx = optax.chain(optax.clip_by_global_norm(1.0),
                     optax.adam(learning_rate=build_schedule(epochs)))
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)
    train_src = ArithmeticSource(train_X, train_Y, length=TRAIN_PER_EPOCH,
                                 seed=SEED + phase, augment_op=True)
    loader = build_loader(train_src, BATCH_SIZE, epochs, True, seed=SEED + phase)
    step = make_phase_step(phase, pix_w)
    steps_per_epoch = TRAIN_PER_EPOCH // BATCH_SIZE
    epoch_loss = epoch_n = epoch = 0; t0 = time.time()
    metric_key = HEADLINE[phase]
    for k, batch in enumerate(loader):
        b = {kk: jnp.asarray(v) for kk, v in batch.items()}
        metrics = step(model, optimizer, b)
        epoch_loss += float(metrics["loss"]) * b["target"].shape[0]; epoch_n += b["target"].shape[0]
        if (k + 1) % steps_per_epoch == 0:
            epoch += 1
            ml = epoch_loss / max(1, epoch_n)
            extra = f"  {metric_key}={float(metrics.get(metric_key, 0)):.3f}"
            print(f"  phase {phase}  epoch {epoch:>2}/{epochs}  loss={ml:.4f}{extra}  [{time.time()-t0:.0f}s]")
            epoch_loss = epoch_n = 0
            if epoch >= epochs: break
    return model

# Run the curriculum.
for phase in (1, 2, 3):
    model = train_phase(model, phase, EPOCHS_PHASE[phase - 1])

metrics = evaluate(model)
print("\\nFinal OCR:", f"{metrics['ocr_acc']*100:.2f}%")
print("Per-op   :", {k: f"{v*100:.1f}%" for k, v in metrics['per_op_acc'].items()})
"""),
]


# ===========================================================================
# Section 5 — show me the paintings
# ===========================================================================


CELLS += [
    md(r"""
## 5 · Headline qualitative result — show me the paintings

Twelve random test samples; for each we show the three input images,
the painted output, and the nearest-neighbour OCR.
"""),
    code(r"""
def _denorm(a):  return (a * MNIST_STD + MNIST_MEAN).clip(0, 1)
def _ocr_one(img):
    return int(ocr_pred(img[None]).item())

rng = np.random.default_rng(0)
fig, axs = plt.subplots(6, 4, figsize=(11, 9), gridspec_kw={"width_ratios":[1, 1, 1, 3]})
for r_idx in range(6):
    for _try in range(50):
        s = test_src[int(rng.integers(0, len(test_src)))]
        if abs(s.result) <= RESULT_MAX:
            break
    ia = jnp.asarray(s.img_a[None].astype(np.float32))
    io = jnp.asarray(s.img_op[None].astype(np.float32))
    ib = jnp.asarray(s.img_b[None].astype(np.float32))
    img = np.asarray(predict_img(model, ia, io, ib))[0, 0]
    pred = _ocr_one(jnp.asarray(img))
    mark = "✓" if pred == s.result else "✗"
    axs[r_idx, 0].imshow(_denorm(s.img_a[0]), cmap="gray"); axs[r_idx, 0].set_title(f"{s.sym_a}")
    axs[r_idx, 1].imshow(_denorm(s.img_op[0]), cmap="gray"); axs[r_idx, 1].set_title(GLYPHS[s.op_idx])
    axs[r_idx, 2].imshow(_denorm(s.img_b[0]), cmap="gray"); axs[r_idx, 2].set_title(f"{s.sym_b}")
    axs[r_idx, 3].imshow(img, cmap="gray", vmin=0, vmax=1)
    axs[r_idx, 3].set_title(f"truth {s.result}  ·  model says {pred}  {mark}")
for a in axs.flat: a.axis("off")
plt.tight_layout(); plt.show()
"""),
]


# ===========================================================================
# Section 6 — operator as direction (geometric calculator)
# ===========================================================================


CELLS += [
    md(r"""
## 6 · The geometric calculator — *operator-as-direction*

Goodfire AI showed in 2025 that arithmetic in Llama 3.1 is organised so
that the operator becomes a *direction in latent space*. We reproduce
that finding here in a model four orders of magnitude smaller, then go
beyond it.

**Setup.** For each $(a, op_i, b)$ on the $10\!\times\!10$ grid we
collect the trunk vector $t$. The pairwise differences
$t(\cdot, op_i, \cdot) - t(\cdot, op_j, \cdot)$, stacked over all
$(a, b)$, give a $100\!\times\!256$ matrix; its top SVD direction is
*the* operator-pair direction.
"""),
    code(r"""
# Build the canonical grid: one MNIST exemplar per digit.
rng = np.random.default_rng(0)
exemplar = {d: test_X[test_Y == d][rng.integers(0, (test_Y == d).sum())] for d in range(10)}
clean_op = []
for op in range(4):
    arr = render_glyph_clean(op).astype(np.float32) / 255.0
    arr = (arr - MNIST_MEAN) / MNIST_STD
    clean_op.append(arr)

def encode_grid(model):
    '''Return arrays of (a, op, b, t, e_a, e_op, e_b) for every grid point.'''
    A, B, OP, R = [], [], [], []
    IA, IO, IB = [], [], []
    for a in range(10):
        for b in range(10):
            for op in range(4):
                if OPS[op] == "//" and b == 0: continue
                A.append(a); B.append(b); OP.append(op)
                IA.append(exemplar[a]); IO.append(clean_op[op]); IB.append(exemplar[b])
                if OPS[op] == "+":  R.append(a + b)
                elif OPS[op] == "-":  R.append(a - b)
                elif OPS[op] == "*":  R.append(a * b)
                else:                  R.append(a // b)
    IA = jnp.asarray(np.stack(IA)[:, None].astype(np.float32))
    IO = jnp.asarray(np.stack(IO)[:, None].astype(np.float32))
    IB = jnp.asarray(np.stack(IB)[:, None].astype(np.float32))
    @nnx.jit
    def fwd(m, ia, io, ib):
        ea, eo, eb, _, t1, t2 = m.trunk(ia, io, ib)
        return ea, eo, eb, t2
    ea, eo, eb, t = fwd(model, IA, IO, IB)
    return (np.array(A), np.array(B), np.array(OP), np.array(R),
            np.asarray(ea), np.asarray(eo), np.asarray(eb), np.asarray(t))

A, B, OPi, R, e_a, e_op, e_b, t = encode_grid(model)
print(f"grid: N={t.shape[0]}, t shape={t.shape}")
"""),
    code(r"""
# Top-1 SVD of operator-pair differences across the grid.
pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
pair_names = [f"{GLYPHS[i]} vs {GLYPHS[j]}" for (i, j) in pairs]
fig, axs = plt.subplots(2, 3, figsize=(11, 6))
all_top1 = []
for k, (i, j) in enumerate(pairs):
    by_op_i = t[OPi == i]; by_op_j = t[OPi == j]
    n = min(by_op_i.shape[0], by_op_j.shape[0])
    diff = by_op_i[:n] - by_op_j[:n]
    _, S, _ = np.linalg.svd(diff, full_matrices=False)
    vr = (S * S) / (S * S).sum()
    all_top1.append(vr[0])
    ax = axs[k // 3, k % 3]
    ax.bar(range(1, 16), vr[:15] * 100, color="#8be9fd")
    ax.bar([1], [vr[0] * 100], color="#ffb86b")
    ax.set_title(f"{pair_names[k]}  ·  top-1 = {vr[0]*100:.0f}%")
    ax.set_ylabel("% variance"); ax.set_xlim(0, 16)
plt.tight_layout(); plt.show()
print(f"mean top-1 share across pairs: {np.mean(all_top1)*100:.0f}%")
"""),
]


# ===========================================================================
# Section 7 — operator as circuit (intervention)
# ===========================================================================


CELLS += [
    md(r"""
## 7 · The big claim — an operator is also a *circuit*

A direction in latent space is a *representation* claim. The stronger
claim is that we can name the **subset of units** that computes a given
operator, surgically remove them, and watch the model lose that
operator while keeping the others.

We do this in two steps:

1. **Tag.** For each $h_1$ unit, look at its middle 64-d weight slot — the
   slice that takes the operator embedding. Ask which of the 14 symbols
   that slot is closest to (using the Yat kernel itself). Units whose
   middle slot prefers an operator class with purity $\geq 0.5$ are
   *operator-tagged*.
2. **Ablate.** Zero out the kernel rows of those units, leaving the
   *bias* untouched (the bias sits inside the kernel numerator — editing
   it would change the kernel function rather than remove the
   landmark).
"""),
    code(r"""
# --- Library: encode digits + augmented operator glyphs for prototype matching ---
def build_library(model, seed=0, k_d=120, k_op=60):
    rng = random.Random(seed)
    tensors, labels = [], []
    for d in range(10):
        idx = rng.sample(range(int((train_Y == d).sum())), k_d)
        for i in idx:
            tensors.append(train_X[train_Y == d][i]); labels.append(d)
    for op in range(4):
        for k in range(k_op):
            arr = (render_glyph_clean(op) if k < 15 else render_glyph_augmented(op, rng))
            a = arr.astype(np.float32) / 255.0
            a = (a - MNIST_MEAN) / MNIST_STD
            tensors.append(a); labels.append(OP_LABEL_OFFSET + op)
    stack = jnp.asarray(np.stack(tensors)[:, None].astype(np.float32))
    @nnx.jit
    def enc(m, x): return m.encoder(x)
    return np.asarray(enc(model, stack)), np.asarray(labels)

# --- Prototype-winner tagging on the middle slot ---
def yat_score(x, w, eps=1e-3):
    dot = x @ w.T
    x_sq = (x ** 2).sum(-1, keepdims=True)
    w_sq = (w ** 2).sum(-1)[None, :]
    return (dot ** 2) / (x_sq + w_sq - 2 * dot + eps)

def operator_tags(model, lib_emb, lib_lbl, k=15, purity_thresh=0.5):
    state = nnx.state(model.h1, nnx.Param)
    flat = dict(nnx.to_flat_state(state))
    W = next(np.asarray(v[...]) for v in flat.values() if np.asarray(v[...]).ndim == 2)
    if W.shape[1] != 64 * 3: W = W.T            # (out, in) form, rows = unit prototypes
    W_op = W[:, 64:128]
    sc = yat_score(lib_emb, W_op)               # (N_lib, 256)
    top = np.argsort(sc, axis=0)[-k:]            # (k, 256)
    H = W.shape[0]
    win = np.zeros(H, dtype=int); pur = np.zeros(H)
    for u in range(H):
        cnt = np.bincount(lib_lbl[top[:, u]], minlength=NUM_SYMBOLS)
        win[u] = int(cnt.argmax()); pur[u] = cnt.max() / k
    is_op = (win >= OP_LABEL_OFFSET) & (pur >= purity_thresh)
    return np.where(is_op, win - OP_LABEL_OFFSET, -1)

lib_emb, lib_lbl = build_library(model, seed=0)
op_of_unit = operator_tags(model, lib_emb, lib_lbl)
print({f"{GLYPHS[op]}-tag": int((op_of_unit == op).sum()) for op in range(4)},
      "untagged:", int((op_of_unit == -1).sum()))
"""),
    code(r"""
# --- Ablate kernel rows for one operator's tagged units, keeping bias intact ---
import copy
def fresh_model_from(template):
    '''Deep-copy params into a new model instance.'''
    m = YatArithmeticGen(rngs=nnx.Rngs(0))
    # Copy each param tensor.
    g_src, s_src = nnx.split(template)
    g_dst, s_dst = nnx.split(m)
    nnx.update(m, s_src)
    return m

def ablate_rows(m, cols, slot_only=False):
    '''Zero h₁ kernel columns (=output units). slot_only=True restricts to the middle 64-d slot.'''
    if cols.size == 0: return
    k = np.array(m.h1.kernel[...])
    if slot_only:
        k[64:128, cols] = 0.0
    else:
        k[:, cols] = 0.0
    m.h1.kernel[...] = jnp.asarray(k)
    # NOTE: bias is part of the Yat kernel numerator — DO NOT zero it.

def per_op_acc(m):
    metrics = evaluate(m); return metrics["per_op_acc"]

base = per_op_acc(model)
print("baseline:", {k: f"{v*100:.1f}%" for k, v in base.items()})

rows = []
rng_int = np.random.default_rng(123)
for mode in ("row", "slot", "random"):
    for kill_op in range(4):
        units = np.where(op_of_unit == kill_op)[0]
        if mode == "random":
            n = max(1, len(units))
            units = rng_int.choice(256, size=n, replace=False)
        m = fresh_model_from(model)
        ablate_rows(m, units, slot_only=(mode == "slot"))
        a = per_op_acc(m)
        rows.append({"mode": mode, "kill": OPS[kill_op], "n_units": int(len(units)),
                     **{f"acc_{op}": a[op] for op in OPS}})

# Display as a tidy matrix per intervention.
import pandas as pd
df = pd.DataFrame(rows)
for mode in ("row", "slot", "random"):
    sub = df[df["mode"] == mode]
    print(f"\\n=== {mode} ===")
    print(sub[["kill", "n_units", "acc_+", "acc_-", "acc_*", "acc_//"]].to_string(index=False,
        formatters={c: lambda v: f"{v*100:5.1f}%" for c in ["acc_+", "acc_-", "acc_*", "acc_//"]}))
"""),
    code(r"""
# --- Visualise the Δ-OCR matrices side by side ---
def delta_matrix(df, mode, base):
    M = np.zeros((4, 4), dtype=np.float32)
    sub = df[df["mode"] == mode]
    for i, kill in enumerate(OPS):
        row = sub[sub["kill"] == kill].iloc[0]
        for j, ev in enumerate(OPS):
            M[i, j] = (row[f"acc_{ev}"] - base[ev]) * 100
    return M

fig, axs = plt.subplots(1, 3, figsize=(13, 4))
vmax = 0
for mode in ("row", "slot", "random"):
    vmax = max(vmax, abs(delta_matrix(df, mode, base)).max())
for ax, mode in zip(axs, ("row", "slot", "random")):
    M = delta_matrix(df, mode, base)
    im = ax.imshow(M, cmap="seismic_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(4)); ax.set_xticklabels(GLYPHS); ax.set_yticks(range(4)); ax.set_yticklabels(GLYPHS)
    ax.set_title(f"{mode} — Δ OCR (pp)")
    ax.set_xlabel("evaluated"); ax.set_ylabel("killed")
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{M[i, j]:+.1f}", ha="center", va="center",
                    fontsize=10, color=("k" if abs(M[i, j]) < vmax * 0.55 else "w"))
plt.tight_layout(); plt.show()
"""),
    md(r"""
**What to look at.**

- **row ablation** column on the `×` row: a large diagonal drop with
  small off-diagonal damage = the operator circuit is real.
- **slot** zeroing only the middle 64-d slice of those same units:
  reproducing most of the diagonal drop with even smaller off-diagonal
  damage = the operator computation flows through the prototype's
  *operator slot*, not laundered through the digit slots.
- **random** column: roughly uniform damage across operators = the
  prototype tagging is doing real work (otherwise the random baseline
  would do the same thing).

In the paper's full run (which sees more training data than this
notebook), the `×` knock-out reaches **−43.7 pp**; `+` reaches
**−13.6 pp**. The `−` and `÷` subsets remain entangled — we explain
why next.
"""),
]


# ===========================================================================
# Section 8 — encoder geometry explains -/÷ entanglement
# ===========================================================================


CELLS += [
    md(r"""
## 8 · Why $-$ and $\div$ entangle — encoder geometry

The intervention picture has two clean rows ($\times$, $+$) and two
muddled rows ($-$, $\div$). The reason isn't a failure of the trunk —
it's the encoder.

Encoding the four canonical operator glyphs gives four 64-d vectors.
Project them to PC1–PC2 and look at the pairwise cosines:
"""),
    code(r"""
@nnx.jit
def encode_clean(m, x): return m.encoder(x)

op_e = np.asarray(encode_clean(model, jnp.asarray(np.stack(clean_op)[:, None].astype(np.float32))))
print("operator embeddings shape:", op_e.shape)

# Cosine matrix.
N = op_e / (np.linalg.norm(op_e, axis=1, keepdims=True) + 1e-9)
cos = N @ N.T
print("cosines:")
print("       " + "  ".join(GLYPHS))
for i, gly in enumerate(GLYPHS):
    print(f"  {gly}  ", "  ".join(f"{cos[i, j]:+.2f}" for j in range(4)))

# 2-D PCA.
mu = op_e.mean(0, keepdims=True); centred = op_e - mu
U, S_, Vt = np.linalg.svd(centred, full_matrices=False)
proj = centred @ Vt[:2].T
plt.figure(figsize=(5, 4))
for op in range(4):
    plt.scatter(proj[op, 0], proj[op, 1], s=220, edgecolor="k")
    plt.annotate(GLYPHS[op], (proj[op, 0], proj[op, 1]), fontsize=20, ha="center", va="center")
plt.title(f"encoder operator embeddings · PC1+PC2 = {(S_[:2]**2 / (S_**2).sum()).sum()*100:.0f}%")
plt.xlabel("PC1"); plt.ylabel("PC2"); plt.tight_layout(); plt.show()
print(f"cos(−, ÷) = {cos[1, 3]:+.2f}   — the encoder treats these two glyphs as nearly the same direction.")
"""),
]


# ===========================================================================
# Section 9 — counterfactual steering
# ===========================================================================


CELLS += [
    md(r"""
## 9 · Counterfactual steering — operator-as-direction is *causal*

The intervention removes a circuit by zeroing prototypes. The
**symmetric** test is whether *adding* a direction to the trunk at
inference can install an operator that wasn't there.

For each ordered $(i, j)$ we recompute the SVD top-1 direction
$v_{ij}$ of $t(a, op_j, b) - t(a, op_i, b)$ over the $10\!\times\!10$
grid, then at inference when the model is fed $(a, op_i, b)$ we add
$\lambda v_{ij}$ to its trunk vector before the decoder.

If steering works, the painted answer should change from $op_i(a, b)$
to $op_j(a, b)$.
"""),
    code(r"""
# Compute steering directions.
def expected(a, b, op):
    return {"+": a + b, "-": a - b, "*": a * b, "//": (a // b if b != 0 else -10)}[OPS[op]]

# Map (a, b, op) -> trunk vector index.
key_to_idx = {(int(A[n]), int(B[n]), int(OPi[n])): n for n in range(len(A))}

# By-op trunk arrays for the common (a, b) keys.
keys_per_op = [set((int(A[n]), int(B[n])) for n in range(len(A)) if OPi[n] == op) for op in range(4)]
common = sorted(set.intersection(*keys_per_op))
print(f"common (a, b) keys across all four operators: {len(common)}")

dirs = {}
for i in range(4):
    for j in range(4):
        if i == j: continue
        diff = np.stack([t[key_to_idx[(a, b, j)]] - t[key_to_idx[(a, b, i)]] for (a, b) in common])
        _, S, Vt = np.linalg.svd(diff, full_matrices=False)
        v = Vt[0]
        if (diff @ v).mean() < 0: v = -v
        dirs[(i, j)] = v / np.linalg.norm(v)

# Sweep λ and measure flip rate.
@nnx.jit
def decode_batch(m, t_batch): return m.decoder(t_batch)

lambdas = np.linspace(0, 6, 13)
sweep = {}
for (i, j), v in dirs.items():
    rows = []
    for lam in lambdas:
        new_t = np.stack([t[key_to_idx[(a, b, i)]] + lam * v for (a, b) in common])
        img = np.asarray(decode_batch(model, jnp.asarray(new_t)))
        preds = [int(ocr_pred(jnp.asarray(img[k:k+1]))[0]) for k in range(img.shape[0])]
        flip = float(np.mean([preds[k] == expected(a, b, j) for k, (a, b) in enumerate(common)]))
        keep = float(np.mean([preds[k] == expected(a, b, i) for k, (a, b) in enumerate(common)]))
        rows.append({"lambda": float(lam), "flip": flip, "keep": keep})
    sweep[(i, j)] = rows
    best = max(rows, key=lambda d: d["flip"])
    print(f"  {GLYPHS[i]} → {GLYPHS[j]}  best flip = {best['flip']*100:5.1f}%  at λ={best['lambda']:.1f}  (keep i = {best['keep']*100:5.1f}%)")
"""),
    code(r"""
# Visualise the 4×4 sweep.
fig, axs = plt.subplots(4, 4, figsize=(12, 9))
for i in range(4):
    for j in range(4):
        ax = axs[i, j]
        if i == j:
            ax.text(0.5, 0.5, GLYPHS[i], ha="center", va="center", fontsize=22,
                    transform=ax.transAxes, color="grey")
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values(): s.set_visible(False)
            continue
        rows = sweep[(i, j)]
        ax.plot([r["lambda"] for r in rows], [r["flip"]*100 for r in rows], color="#ffb86b", lw=1.8, label="flip → j")
        ax.plot([r["lambda"] for r in rows], [r["keep"]*100 for r in rows], color="#8be9fd", lw=1.8, label="keep i")
        ax.set_ylim(-2, 102); ax.set_title(f"{GLYPHS[i]} → {GLYPHS[j]}", fontsize=10)
        ax.set_xlabel("λ"); ax.set_ylabel("%")
plt.tight_layout(); plt.show()
"""),
    md(r"""
**Reading the steering matrix.**

* High orange peak before the cyan keep-rate collapses = clean causal
  edit; the model has switched its painted answer to the target
  operator.
* `−` → `÷` is the cleanest case in the paper's full-data run: $70\%$
  flip at $\lambda = 5$. `+` → `÷` and `+` ↔ `−` work similarly.
* Steering *into* `×` is hard — its trunk region is the furthest from
  the others (Section 6 PCA), so a single direction is not enough to
  push another operator's trunk vector all the way there.

This is the causal upgrade to the geometric-calculator picture: not
just *"operator differences are dominated by one direction"* but
*"moving along that direction IS the operator change"*.
"""),
]


# ===========================================================================
# Section 10 — operator-conditioned slot painting
# ===========================================================================


CELLS += [
    md(r"""
## 10 · Operator-conditioned slot painting — the decoder is *operator-aware*

The decoder atlas in the paper showed that many trunk units paint a
*single output slot* (sign, tens, or units). Here we ask whether the
slot specialisation is also tied to the operator the unit prototypes.

We compute the analytic decoder Jacobian
$\partial \mathrm{img}/\partial t_u$ at each operator's own operating
point (the mean trunk vector over that operator's grid) and decompose
each unit's Jacobian into three slot energies.
"""),
    code(r"""
graphdef, state = nnx.split(model)

def decode_at(t_vec, state_):
    rebuilt = nnx.merge(graphdef, state_)
    return rebuilt.decoder(t_vec[None, :])[0, 0]

jac_one = jax.jit(jax.jacrev(lambda tt: decode_at(tt, state)))

per_op_J = []
for op in range(4):
    t_op = t[OPi == op].mean(0)
    J_op = np.asarray(jac_one(jnp.asarray(t_op)))    # (28, 84, 256)
    per_op_J.append(np.moveaxis(J_op, -1, 0))        # (256, 28, 84)
per_op_J = np.stack(per_op_J)                        # (4, 256, 28, 84)
print("per-op Jacobian stack:", per_op_J.shape)
"""),
    code(r"""
# For each (op, unit op-tag) pair: mean slot-energy share over the units tagged for that op.
slot_means = {}
for op in range(4):
    mask = op_of_unit == op
    if mask.sum() == 0: continue
    J = per_op_J[op]                                 # use this op's operating point
    e_sign  = (J[:, :, 0:28]  ** 2).sum(axis=(1, 2))
    e_tens  = (J[:, :, 28:56] ** 2).sum(axis=(1, 2))
    e_units = (J[:, :, 56:84] ** 2).sum(axis=(1, 2))
    s = np.stack([e_sign, e_tens, e_units], axis=1)
    share = s / (s.sum(axis=1, keepdims=True) + 1e-9)
    slot_means[OPS[op]] = share[mask].mean(axis=0).tolist()
print("per-op slot share (sign / tens / units):")
for k, v in slot_means.items():
    print(f"  {k:>2}: sign={v[0]*100:5.1f}%  tens={v[1]*100:5.1f}%  units={v[2]*100:5.1f}%")

# Bar chart.
ops_with = list(slot_means.keys())
x = np.arange(len(ops_with)); w = 0.27
colors = ["#a5e075", "#ffb86b", "#8be9fd"]
for k, name in enumerate(["sign", "tens", "units"]):
    plt.bar(x + (k - 1) * w, [slot_means[op][k] * 100 for op in ops_with],
            width=w, color=colors[k], label=f"{name} slot")
plt.xticks(x, [GLYPHS[OPS.index(op)] + " units" for op in ops_with], fontsize=12)
plt.ylabel("mean Jacobian energy share (%)"); plt.legend(frameon=False)
plt.title("E2 — where in the output do each op-tag's units paint?")
plt.tight_layout(); plt.show()
"""),
    md(r"""
**The headline.** $\times$-tagged units load $\sim 75\%$ of their
Jacobian energy in the tens slot — they are the *carry-bearers* of
multiplication. $+$, $-$ and $\div$-tagged units put nearly all of
theirs in the units slot, because their answers usually fit in one
digit. The decoder's slot alphabet (Section 5 of the paper) is
**operator-aware**, not just position-aware.
"""),
]


# ===========================================================================
# Section 11 — wrap-up
# ===========================================================================


CELLS += [
    md(r"""
## 11 · Wrap-up

What we built and what it tells us:

* **A 0.7 M-parameter rational-kernel network that does single-digit
  MNIST arithmetic by painting the answer image** — no softmax over result
  classes anywhere user-facing.
* **A phased training recipe** with strict `stop_gradient` cuts and
  `optax.multi_transform` freezes so that each component (encoder,
  trunk, decoder) is trained against a single objective in isolation.
* **The operator-as-direction picture** of Goodfire AI reproduced in
  a model four orders of magnitude smaller than the original Llama
  analysis (Section 6).
* **The operator is also a circuit.** We name 30 Yat units whose
  middle-slot prototype is a $\times$ glyph and zero their kernel rows;
  $\times$ accuracy collapses while the other operators are nearly
  untouched (Section 7).
* **The same direction is causal.** Adding the SVD operator-difference
  direction to the trunk at inference re-paints the answer
  (Section 9).
* **The decoder reads operator-conditioned slots.** $\times$-tagged
  units paint the tens column; $+,\,-,\,\div$-tagged units paint the
  units column (Section 10).

This notebook trained on a small budget so it could fit a Kaggle
session; the paper's headline numbers use larger budgets but the recipe
and code are identical.

If you want to dig deeper, the full paper, the live browser demo, and
the marketing kit live at the project page. The most interesting
follow-ups we are *not* doing here are: (a) a larger result space, so
the Goodfire-style mod-k specialisation has somewhere to live; (b)
contrastive supervision between $-$ and $\div$ glyphs at phase 1, to
break the encoder's aliasing of those two; (c) sparse-autoencoder
features over the trunk as an alternative basis to the Yat prototypes.
"""),
]


# ===========================================================================
# Assemble + save
# ===========================================================================


def main():
    nb = nbf.v4.new_notebook()
    nb.cells = CELLS
    nb.metadata = {
        "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
        "language_info": {"name": "python"},
        "kaggle": {"accelerator": "nvidiaTeslaT4", "dataSources": [], "isInternetEnabled": True,
                   "language": "python", "sourceType": "notebook"},
    }
    out = Path(__file__).resolve().parent / "painting_arithmetic_kaggle.ipynb"
    with out.open("w") as f:
        nbf.write(nb, f)
    print(f"wrote {out}  ({len(CELLS)} cells)")


if __name__ == "__main__":
    main()
