# Twitter / X — "lead with the research finding" variant

**When to use:** when you're optimising for reaches inside the mech-interp community. Slower viral curve, higher long-tail engagement from researchers.

**Asset:** the SVD bar chart (from `social_card.png` or rendered standalone via `viz.op_as_shift`).
**Tag:** `@NeelNanda5` and `@goodfire_ai` are appropriate here — you're testing their claim.

---

## Thread

**1/**
Goodfire showed that arithmetic operations in Llama 3.1 act as single directions in the residual stream.

I wanted to know if that pattern also shows up in a model 4 orders of magnitude smaller.

[ATTACH: social_card.png]

**2/**
Setup: a 0.8M-parameter rational-kernel network does single-digit MNIST arithmetic. Three image inputs (digit, operator, digit). One image output — the painted answer.

96.91% OCR on the test set. Trained in 13 minutes on a single GPU.

[ATTACH: prediction.gif]

**3/**
The SVD test: for every pair of operators, take trunk(a, op_i, b) − trunk(a, op_j, b) across all 100 (a,b) pairs, look at the singular spectrum.

If "operator is one direction," top-1 should dominate.

It does. **62–76% of the variance** captured by the top singular value for every pair.

**4/**
The trunk also arranges results along a value line *within* each operator's region of latent space. Same picture as Goodfire's modular-circle finding for Llama, in 0.81M parameters.

The decoder gives us a free interpretability lens: feed any latent direction back through it and look at what it *paints*.

**5/**
- preprint: https://arxiv.org/abs/<ARXIV-ID>
- code: https://github.com/mlnomadpy/painting-arithmetic
- live demo: https://huggingface.co/spaces/<USER>/painting-arithmetic

a tiny world-model of a calculator, fully legible. JAX · Flax NNX · Grain.
