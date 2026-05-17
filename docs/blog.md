# Painting Arithmetic: A Tiny Geometric Calculator That Paints Its Answer

*A 0.8 M-parameter rational-kernel network does single-digit MNIST arithmetic in pure latent space and renders the result as an image. Along the way it reproduces — at three orders of magnitude smaller — the "geometric calculator" interpretability findings that Goodfire AI recently reported for Llama 3.1.*

---

## 1. The picture, in one paragraph

Hand a small model three 28×28 images: a handwritten digit, an operator glyph (`+`, `−`, `×`, `÷`), another handwritten digit. The model has never been told what any of those symbols mean. It must produce — pixel by pixel — a 28×84 image whose three slots read `[sign | tens | units]` of the integer answer. No softmax over result classes is ever exposed. The number you read is the model's final convolution. We train this model in **JAX / Flax NNX / Grain**, in about three minutes on a single T4 GPU, and we look inside the trained latent space to see what it learned.

We find that the trunk arranges results along a *value line* per operator. We find that the difference between any two operators is dominated by a *single direction* in latent space — the same property Goodfire identified in Llama. We find that the decoder has carved itself into a small library of slot painters. And we find that about 20 % of the Yat-kernel units have learned operator-conditioned prototypes — single neurons that fire on `(7, +, 3)`-shaped triples.

The model is a one-step world model of a calculator, in image space.

## 2. Why bother

Visual arithmetic on MNIST is an old teaching task. The classic recipe is:

1. Two CNNs read the digits.
2. A one-hot operator token is concatenated.
3. A softmax classifier picks one of ~91 result classes.

It is a fine warm-up exercise and it tells us almost nothing about how a network represents *value*. Two things make this version more interesting:

- **Operator-as-image.** We refuse to give the model a token. The operator is also a 28×28 pixel image, encoded by the *same* CNN that reads the digits. The model has to learn fourteen symbols in a single shared latent space — ten digits plus four operators.
- **Output-as-image.** We refuse to give the model a softmax. The answer is a 28×84 painting that the model must render. With a trained decoder lying around, we can use it as an interpretability lens: feed any latent vector through it, see what image comes out.

The point isn't to beat anything. The point is to build a tiny, end-to-end legible artefact and look at it.

## 3. The model in 100 words

```
img_a ─┐
img_op ┼──► [shared CNN encoder] ──► (e_a, e_op, e_b)  each ∈ ℝ⁶⁴
img_b ─┘                                          │
                                  concat → ℝ¹⁹²
                                       ↓
                                   YatNMN h₁ → ℝ²⁵⁶
                                       ↓
                                   YatNMN h₂ → ℝ²⁵⁶     ← "trunk" t₂
                                       ↓
                                 ConvTranspose ×2
                                       ↓
                                 sigmoid · 28×84       ← painted answer
```

The trunk uses **Yat layers**, a rational kernel published in the `nmn` library:

$$
y_u = \alpha_u \cdot \frac{(x \cdot W_u + b_u)^2}{\lVert x - W_u \rVert^2 + \varepsilon}.
$$

Two facts make this useful here. First, each row $W_u$ is literally a *prototype point* in input space: the denominator goes to zero when $x$ approaches $W_u$. Second, the numerator additionally rewards directional alignment, so each unit's receptive field is both localised and directional. Two stacked Yat layers chain prototype-matching.

Every conv layer in the encoder is also a `YatConv` rational kernel by default. The decoder is intentionally small: a linear projection, two `ConvTranspose` upsamples, sigmoid. 0.81 M parameters total.

## 4. Why naïve training fails (and how to fix it)

The first version of the model trained with plain pixel MSE reached 62 % OCR accuracy and failed in a very specific way: it painted single-digit answers for two-digit problems. `3 × 5 = 15` came back as just `5`.

The root cause is data imbalance. With uniform sampling over `(a, op, b)`:

- Subtraction always produces a single-digit result for our operand range.
- Integer division always produces a single-digit result.
- Addition produces single-digit results about half the time.

About **72 %** of training targets have a blank tens slot. Naïve pixel MSE rewards the decoder for defaulting to "leave the tens column empty."

Three changes fix this:

1. **BCE instead of MSE** on the pixel loss. For [0, 1] sigmoid outputs, BCE produces sharper edges than MSE.
2. **Per-pixel weighting.** We multiply the BCE map by a 2× weight on the tens-slot column. The decoder spends as much loss budget on tens as on units, even though tens is blank much of the time.
3. **Per-slot CE auxiliary heads.** Three small classifiers attached to the trunk predict `(sign, tens, units)` independently. They never appear at inference — they exist only to give the trunk a sharp, decomposed gradient on each slot.

Adding (3) alone jumped accuracy from **62.5 % → 96.9 %** in the final training run. The slot-CE block is doing the heavy lifting.

The full loss is the BCE pixel term, the per-slot CE term, a modular CRT term (mod 2 / mod 5 / mod 11 / sign on the trunk — a Chinese-Remainder-Theorem prior the network can read), and a 14-way symbol-classification head on each input embedding (keeps the shared encoder honest).

## 5. The numbers

**Test accuracy** at 25 epochs of 60 k samples each (≈ 13 min on Apple-Silicon MPS, ≈ 5 min on a T4):

| operator | OCR accuracy |
|---|---:|
| `+`  | 96.2 % |
| `−`  | 96.0 % |
| `×`  | **97.1 %** |
| `÷`  | **98.3 %** |
| **overall** | **96.91 %** |

The 14-way auxiliary symbol head sits at **99.3 %** — the encoder is essentially solved. The model retains **96.78 %** when the operator glyph is replaced at test time by a randomly augmented variant (different fonts, rotation, scale).

Multiplication being the *highest*-accuracy operator surprised us. v1 (a softmax baseline) had multiplication as the *worst* one. Plausible reason: with slot-CE supervision, multiplication's wider output range becomes an *advantage* — the model has more bits of discriminative signal per training example.

## 6. What's inside

We trained the model. Now we open it up.

### 6.1 The trunk learns a value line per operator

We project every post-`h₂` trunk vector (one per `(a, op, b)` on the full grid, 390 points total after dropping `÷` by zero) into two dimensions via PCA. Two views: coloured by ground-truth result, and coloured by operator.

![trunk PCA](images/trunk_pca.png)

**The four operators occupy four distinct regions.** Inside each region, the result value moves smoothly along a direction — the colour gradient on the left panel makes it obvious. The model has learned, for each operator, what it means to *increment the answer by one*. PC1 carries 21.5 % of variance, PC2 19.0 %.

### 6.2 The operator is one direction in latent space

This is the Goodfire claim, tested directly. For every pair of operators, we compute the trunk-vector difference at every $(a, b)$:

$$
\Delta_{i,j}(a, b) = t_2(a, \text{op}_i, b) - t_2(a, \text{op}_j, b)
$$

If "the operator is one direction in the trunk," the SVD of the resulting $100 \times 256$ matrix should be concentrated at the top singular value.

| pair | top-1 var ratio | top-3 var ratio |
|---|---:|---:|
| `+ − −` | 75.8 % | 85.2 % |
| `+ − ×` | 63.8 % | 80.2 % |
| `+ − ÷` | 69.8 % | 79.8 % |
| `− − ×` | 64.9 % | 79.3 % |
| `− − ÷` | 70.2 % | 80.1 % |
| `× − ÷` | 62.2 % | 75.5 % |

**Top-1 captures 62 – 76 % for every pair.** A single direction in the 256-dim trunk is responsible for most of every operator transition. That's the geometric-calculator signature, reproduced in 0.81 M parameters.

### 6.3 The decoder is a slot alphabet

Because we have a trained decoder, we can use it as a fixed lens. For each of the 256 trunk units we feed $\alpha \cdot e_u$ — a one-hot peaked at unit $u$ — and look at the image, minus the zero-input baseline. The result is a per-unit "footprint" picture: what does this one neuron contribute to the painted answer?

![decoder unit atlas](images/decoder_unit_atlas.png)

The top units by footprint magnitude show spatial *localisation*. Many units write strongly to a single slot of the output and barely touch the others. The decoder has implicitly carved itself into:

- **Sign-slot painters** (a small set that draws the minus glyph in the leftmost 28 columns).
- **Tens-slot painters** (units that activate the middle 28 columns).
- **Units-slot painters** (the largest group, painting the rightmost 28 columns).

Together they form a *primitive alphabet* the trunk composes via linear combination to draw any of the 91 result images.

### 6.4 The Yat prototypes are visual triples

Each row of $h_1$'s weight matrix $W_u \in \mathbb{R}^{192}$ splits into three 64-d slots that match the three encoder embeddings. We can ask: from a library of real symbol embeddings, which symbol best matches each slot of $W_u$?

For the top units by prototype purity, we then run the full model on the matched triple. The result is "this unit fires on $(7, ×, 3)$ and the model paints $21$" — a complete interpretive story per neuron.

![prototype gallery](images/prototype_gallery.png)

**About 51 of the 256 $h_1$ units** have an operator-class winner in their middle slot. That's roughly **20 % of the trunk specialised to operator-conditioned arithmetic**; the rest are digit-pattern detectors that read $(a, b)$ without conditioning on the operator. This matches Goodfire's observation that a minority of LLM neurons carry the arithmetic logic while the rest carry numeric features.

### 6.5 The trajectory of one operand through latent space

Hold $(\text{op}, b)$ fixed; sweep $a$ from 0 to 9; trace the trunk vector through the same PCA plane. The path tells you how the model *parameterises the answer* as one input varies.

![layer trajectories](images/trajectories.png)

For each operator we see a smooth, nearly-monotonic path — the answer line is real and continuous. The model isn't picking discrete "modes"; it's interpolating along a learned axis.

## 7. Is this a world model?

The framing keeps suggesting itself. We have:

- An **observation space** (28×28 / 28×84 images).
- An **encoder** $\phi : o \mapsto s$ mapping observations to a 64-d latent.
- A **state** $s = t_2$ in 256 dimensions.
- An **action** — the operator — but the action is *also an observation* (an image), encoded by the same encoder.
- A **transition** $T(s, a) \to s'$ implemented by the Yat trunk.
- A **decoder** $\psi : s \to o$ rendering the next observation in pixels.

The unusual property is that the action is itself in the world. There's no "outside" channel through which symbolic information sneaks in — the operator pad is just another image. In that sense yes, this is a closed-loop, pure-latent-space world model.

The caveat: it's a **one-step** world model. The output of the decoder is not in the same distribution as the input pads — `img_out` is a 28×84 glyph-rendered triple, `img_a` is a 28×28 MNIST hand-drawing. You can't natively feed the painted answer back as a new operand. Closing that distributional gap is the v4 follow-up: train so that the decoder's output is in the same space as the encoder's input, and you have a calculator world you can plan in.

## 8. Reproducing

Everything is in the repo. The default training config reaches ~96 % OCR in three minutes on a Kaggle T4:

```bash
pip install -e .
painting-arithmetic-train --epochs 15 --train-size 30000
```

For the full 96.91 % paper number, run 25 epochs of 60 k samples each (≈ 13 min on Apple MPS, ≈ 5 min on a T4):

```bash
painting-arithmetic-train --epochs 25 --train-size 60000
```

The interpretability views are produced by importable functions in `painting_arithmetic.viz` — no notebook required:

```python
from painting_arithmetic import data, viz
from painting_arithmetic.eval import load_checkpoint
from painting_arithmetic.model import YatArithmeticGen
from flax import nnx

bins = data.load_mnist_binned("./data", train=False)
model = YatArithmeticGen(rngs=nnx.Rngs(0))
load_checkpoint(model, "./ckpts/model.npz")

grid = viz.collect_trunk_grid(model, bins)
pca = viz.trunk_pca(grid)
print("PC variance ratios:", pca["var_ratio"])
shift = viz.op_as_shift(grid)
print("operator-as-shift top-1:", {k: f"{v['top1']*100:.1f}%" for k, v in shift.items()})
atlas = viz.decoder_unit_atlas(model)
print("decoder unit footprints:", atlas["footprints"].shape)
```

Or open `notebooks/painting_arithmetic.ipynb` for the Kaggle tutorial.

## 9. What's not in this version

Honest scope limits:

- **Single digits only.** Operands live in $\{0, \dots, 9\}$, results in $[-9, 81]$.
- **The 28×84 output geometry is hard-coded** — three slots, one painter alphabet, no autoregressive extension.
- **No causal interventions.** We don't yet zero out the dominant "+ → ×" direction in $t_2$ and confirm that the painted answer drifts from sum to product. That's the experiment that would convert the qualitative SVD finding into a causal claim, in the spirit of Goodfire's steering experiments.
- **Off-by-one residual.** The ~3 % accuracy gap is dominated by arithmetic errors, not slot-blanking. Examples: `3 × 5` predicted as `25`, `8 − 3` predicted as `6`. The trunk is right about *which slot* needs digits; it's the digit identities that occasionally drift.

## 10. Open follow-ups

1. **Steering experiment.** Pick the top-singular operator-shift direction, scale it, add to a real prediction, decode. Visually verify the painted answer drifts. Causal evidence beats qualitative spectrum analysis.
2. **Closed-loop iteration.** Make the decoder's output ingestible as a new operand. Then nested expressions like `((3 + 5) × 2) ÷ 4` become a multi-step rollout, and you have a *full* world model — not a one-step transition kernel.
3. **Bigger result space.** Two-digit operands push the output to four or five slots and stress the multi-digit story further. Worth checking that the slot-CE recipe scales.
4. **Hand-drawn operators in production.** The model already trains with rotated / blurred / multi-font glyphs; a small fine-tune on actual hand-drawn samples (the live demo already shows this works) closes the gap.

## Acknowledgements

The Yat rational kernel is from the `nmn` library by [Taha Bouhsine](https://github.com/mlnomadpy) (et al., 2025–2026). The geometric-calculator interpretability framing is directly inspired by [Goodfire AI's analysis of Llama 3.1 arithmetic](https://www.goodfire.ai/research/a-geometric-calculator). The grokking-via-modular-Fourier-structure story comes from [Nanda et al. (ICLR 2023)](https://arxiv.org/abs/2301.05217). Activation atlases / feature visualisation: [Olah et al. (Distill 2017)](https://distill.pub/2017/feature-visualization/). Prototype networks: [Snell et al. (NeurIPS 2017)](https://arxiv.org/abs/1703.05175).

## Reproducibility statement

All code, training scripts, the Kaggle notebook, and the live demo are in this repository. A full training run from scratch at the paper config takes ~13 min on Apple MPS or ~5 min on a single T4. The headline numbers are from a single seed (0); rerunning with `--seed 1`, `--seed 2`, etc. tracks within ±0.3 OCR points.

The trained checkpoint (`ckpts/model.npz`, ≈ 3 MB) and the exported ONNX (`demo/model.onnx`) are produced by the training script and the `scripts/export_onnx.py` helper respectively.

---

*Last updated: 2026-05-17.*
