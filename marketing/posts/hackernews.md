# Hacker News — submission draft

**Best time:** Tuesday or Wednesday, 9–11 AM US Eastern.
**URL field:** point at the Hugging Face Space (the live demo). HN traffic loves "open this and play in 2 seconds." Don't link the GitHub repo as the primary URL — that's a one-click-deeper read.

---

## Title (≤ 80 chars)

**Painting Arithmetic: a tiny network that draws its answer (live demo)**

Alternative titles, ranked:

1. *Painting Arithmetic: a tiny network that draws its answer (live demo)*  ← recommended
2. *A 0.8M-parameter network that paints MNIST arithmetic answers*  ← clearer, less curiosity-gap
3. *Show HN: A network that does MNIST arithmetic by painting the result*  ← if you go Show-HN

---

## First comment (post immediately after submission, signed as you)

Self-explainer for the URL above:

You draw two digits and pick an operator. Three 28×28 images go into a small CNN encoder. Two Yat rational-kernel layers do the math in latent space. A tiny ConvTranspose decoder *paints* the 28×84 answer. There's no softmax over result classes anywhere — the number you read is the model's final convolution.

The reason this is interesting beyond the visual gimmick is that the trained model reproduces — at four orders of magnitude smaller — the "geometric calculator" pattern Goodfire AI recently reported for Llama 3.1: the difference between any two operators in the trunk is dominated by a single direction in latent space (62–76 % of variance for every operator pair, by SVD on the per-(a, b) trunk differences).

Total artefact: 0.81M params, ~13 min training on a single T4, fully open under MIT.

Repo: <REPO-URL>
Paper: <ARXIV-URL>
Kaggle notebook: <KAGGLE-URL>

Built on JAX / Flax NNX / Grain. Happy to discuss the Yat layer, the slot-aware supervision recipe that fixes the multi-digit failure mode (62 % → 97 % OCR), or the "world model in pixel space" framing.
