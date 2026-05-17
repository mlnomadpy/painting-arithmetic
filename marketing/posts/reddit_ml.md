# r/MachineLearning — submission draft

**Subreddit:** r/MachineLearning
**Flair:** Research (or Discussion if it doesn't fit Research)
**Best time:** Tuesday or Wednesday, 8–10 AM US Eastern.

---

## Title (≤ 300 chars, r/MachineLearning prefers `[R]` prefix for research)

**[R] Painting Arithmetic: reproducing Goodfire's "geometric calculator" finding in a 0.8M-parameter rational-kernel network (image input → painted image output)**

---

## Body

I built a small open-source artefact that I think is useful as a mech-interp toy: a tiny network that does single-digit MNIST arithmetic from image inputs and outputs the answer **as a painted image**. The full forward pass stays in latent space — no softmax over result classes anywhere user-facing.

**The setup**

Three 28×28 images come in: a handwritten digit, an operator glyph (`+`, `−`, `×`, `÷`), another handwritten digit. The model has never been told what those symbols mean. It must produce — pixel by pixel — a 28×84 image whose three slots read `[sign | tens | units]` of the integer answer.

```
img_a, img_op, img_b  →  shared CNN encoder
                          ↓
                          concat → ℝ¹⁹²
                          ↓
                          YatNMN h1 → ℝ²⁵⁶
                          ↓
                          YatNMN h2 → ℝ²⁵⁶   ("the trunk")
                          ↓
                          ConvTranspose ×2 → sigmoid
                          ↓
                          img_out (28×84)
```

The Yat layer is a rational kernel: `α(x·W+b)² / (‖x − W‖² + ε)`. Each row of `W` is literally a prototype point in input space, which gives us a built-in interpretability hook.

**The numbers**

- 96.91 % OCR test accuracy (nearest-neighbour over 91 canonical result images)
- 13 min training on a single Kaggle T4 / Apple Silicon MPS
- 0.81 M params; the decoder is the bulk
- 96.78 % under augmented operator glyphs at test time (different fonts, rotation, scale)

**The interesting bit**

Goodfire AI's recent post on Llama 3.1 ([link](https://www.goodfire.ai/research/a-geometric-calculator)) showed that arithmetic operations in the residual stream act as **single directions** — top-1 of an SVD over operator-difference vectors captures most of the variance, and you can causally steer one operator into another by adding that direction.

I tested the same diagnostic on this small model:

| op pair | top-1 var ratio | top-3 var ratio |
|---|---:|---:|
| `+ vs −` | 75.8% | 85.2% |
| `+ vs ×` | 63.8% | 80.2% |
| `+ vs ÷` | 69.8% | 79.8% |
| `− vs ×` | 64.9% | 79.3% |
| `− vs ÷` | 70.2% | 80.1% |
| `× vs ÷` | 62.2% | 75.5% |

**Top-1 captures 62–76% of variance for every operator pair.** The same geometric-calculator signature shows up at 4 orders of magnitude smaller scale.

Plus, because the model has a *trained decoder*, you can use it as a fixed lens: feed any latent direction back through it and see what it paints. The resulting unit atlas reveals that the decoder has carved itself into a **slot alphabet** — dedicated sign-slot painters, tens-slot painters, units-slot painters.

**Code / paper / demo**

- Repo: https://github.com/mlnomadpy/painting-arithmetic (JAX / Flax NNX / Grain)
- Live browser demo: https://huggingface.co/spaces/&lt;USER&gt;/painting-arithmetic
- Paper: https://arxiv.org/abs/&lt;ARXIV-ID&gt;
- Kaggle notebook: https://www.kaggle.com/code/&lt;USER&gt;/painting-arithmetic

**Discussion / open questions**

1. The "operator-as-direction" property generalises from billion- to million-parameter models. Does it hold at *every* scale, or is there a threshold below which it fails?
2. The current model is a **one-step** world model — input pads and output paintings live in different distributions. Closing that loop (train so decoder output is ingestible as the next operand) would give a multi-step rollout.
3. A YatConv encoder variant is wired up but currently underperforms at the same training budget (29 % vs 97 %). Probably an LR/init issue; happy for theories.

Honest limitations: single-digit only, ~3 % residual errors are mostly off-by-one on subtraction. Not aiming to set any benchmarks — aiming to be the smallest legible model in this class.

Comments / criticism welcome.
