# Twitter / X — "hybrid" variant (RECOMMENDED)

**When to use:** default choice. Leads with the visual (good for engagement) but front-loads the research framing (good for retention by the people who can credibly amplify you).

**Asset:** `marketing/prediction.gif` first, `marketing/social_card.png` (SVD half) on tweet 3.

---

## Thread (5 tweets)

**1/**
we trained a 0.8M-parameter network to do MNIST arithmetic

three image inputs in: a digit, an operator, a digit
one image out: the answer, painted pixel by pixel

no softmax. the number you read is the model's last layer.

[GIF: prediction.gif]

**2/**
shared CNN encoder → two Yat rational-kernel layers → ConvTranspose decoder.

96.91% OCR test accuracy. 0.81M params. 13 min on a single T4.

then we opened it up.

**3/**
the difference between any two operators in the trunk is dominated by *one direction* in latent space. top-1 SVD captures 62–76% of variance for every pair.

[ATTACH: social_card.png]

this is exactly the "geometric calculator" pattern @goodfire_ai found in Llama 3.1 — but 4 orders of magnitude smaller.

**4/**
because we have a trained decoder, we can use it as a fixed lens for free.

push one-hot vectors through it and the resulting per-unit footprint images show that the decoder has carved itself into a *slot alphabet* — dedicated sign / tens / units painter neurons.

mech-interp toys, but legible.

**5/**
- live demo (browser): https://huggingface.co/spaces/<USER>/painting-arithmetic
- code: https://github.com/mlnomadpy/painting-arithmetic
- paper: https://arxiv.org/abs/<ARXIV-ID>
- kaggle: https://www.kaggle.com/code/<USER>/painting-arithmetic

JAX · Flax NNX · Grain. one-step world model of a calculator, fully open.
