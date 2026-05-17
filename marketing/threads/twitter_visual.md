# Twitter / X — "lead with the visual" variant

**When to use:** general ML / dev audience. Higher virality ceiling, lower research-credibility floor.

**Asset:** `marketing/prediction.gif` (the painted-answer animation).
**Tag:** none required; consider `@goodfire_ai` only if you have a real reason to invite them in (e.g. the SVD slide).

---

## Thread

**1/**
i trained a 0.8M-parameter network to do MNIST arithmetic.

three image inputs in (a digit, an operator, a digit).
one image out — the answer, painted pixel by pixel.

no softmax over result classes anywhere.

[ATTACH: prediction.gif]

**2/**
the whole forward stays in latent space.

shared CNN encoder reads the three images → two Yat rational-kernel layers do the math → a small ConvTranspose decoder *renders* the answer.

96.91% OCR on the test set after 13 minutes of training on a single GPU.

**3/**
then we opened it up.

the difference between any two operators in the trunk is dominated by *one direction in latent space*. top-1 SVD captures 62–76% of the variance for every pair.

[ATTACH: social_card.png — the SVD chart half is the relevant bit]

**4/**
this is the "geometric calculator" pattern @goodfire_ai found inside Llama 3.1, reproduced in a model 4 orders of magnitude smaller.

every interpretability claim is checked by a tiny script you can run locally.

**5/**
- live demo: https://huggingface.co/spaces/<USER>/painting-arithmetic
- code: https://github.com/mlnomadpy/painting-arithmetic
- kaggle notebook: https://www.kaggle.com/code/<USER>/painting-arithmetic
- paper: https://arxiv.org/abs/<ARXIV-ID>

JAX · Flax NNX · Grain. fully open, fully reproducible.
