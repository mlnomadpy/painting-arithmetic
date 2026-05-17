# LinkedIn post draft

**When to post:** Tuesday morning, ~9 AM your local time.
**Attach:** `marketing/social_card.png` (LinkedIn renders 1200×630 inline beautifully).

---

I've open-sourced **Painting Arithmetic** — a 0.8M-parameter rational-kernel network that does single-digit MNIST arithmetic and paints the answer *as an image*.

🎨 Three image inputs in (a digit, an operator, a digit). One image out — the integer answer rendered pixel by pixel. No softmax anywhere user-facing. The number you read is the model's last layer.

📊 The numbers: 96.91% OCR test accuracy, 13 min training on a single GPU, 0.81M parameters.

🔬 The interesting bit — interpretability. Goodfire AI recently showed that arithmetic operations in Llama 3.1 act as single directions in the residual stream. I tested the same diagnostic on this tiny model and found the same pattern at four orders of magnitude smaller: the top-1 SVD direction captures 62–76% of variance for every operator pair.

🧠 Plus, with a trained decoder you get a free interpretability lens — feed any latent direction back through it and see what it paints. The resulting unit atlas reveals that the decoder carved itself into a slot alphabet: dedicated sign-slot, tens-slot, units-slot painter neurons.

Built on JAX / Flax NNX / Grain. Full source, training scripts, Kaggle notebook, paper, and a static browser demo (ONNX-Runtime-Web) all in the repo.

🔗 Repo + paper + demo: https://github.com/mlnomadpy/painting-arithmetic

#machinelearning #interpretability #jax #flax #mnist
