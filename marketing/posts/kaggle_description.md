# Kaggle notebook description

Use this text as the notebook description on Kaggle (top of the page, under the title).

---

## Title

**Painting Arithmetic: A 0.8M-parameter Network That Paints Its Answer**

## Description

A small JAX / Flax NNX / Grain tutorial that builds, trains, and *opens up* a tiny rational-kernel network that does single-digit MNIST arithmetic — from image inputs to image outputs, end-to-end.

You feed it three 28×28 images: a digit, an operator (`+`, `−`, `×`, `÷`), another digit. It paints the integer answer onto a 28×84 canvas with three slots `[sign | tens | units]`. **No softmax over result classes anywhere user-facing.**

### What you'll learn

1. The Yat rational kernel, `α(x·W+b)² / (‖x − W‖² + ε)` — every row of `W` is literally a prototype point in input space, which gives us interpretability for free.
2. The slot-aware training recipe that solves the multi-digit failure mode: pixel BCE + per-slot CE + modular CRT CE + symbol CE.
3. Four interpretability views — trunk PCA (value lines per operator), operator-as-shift SVD (the Goodfire diagnostic, reproduced at 4 orders of magnitude smaller), decoder unit atlas, prototype gallery.

### Runtime

≈ 3 minutes on a Kaggle T4 GPU at the default config (10 epochs × 20k samples). Bump `cfg.epochs = 25, cfg.train_size = 60_000` for the full ~96.9% paper number (~5 min on T4).

> ⚠️ **GPU choice:** set the accelerator to **GPU T4 x1**. The P100 SKU is `sm_60` and the default PyTorch / JAX wheel needs `sm_70+`.

### Code

- Full source / paper / live demo: https://github.com/mlnomadpy/painting-arithmetic
- Yat layer library: https://github.com/azettaai/nmn

If you like this, an upvote helps surface it to other Kaggle users 🙏
