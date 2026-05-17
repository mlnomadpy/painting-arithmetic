# Notebooks

`painting_arithmetic.ipynb` is the JAX / Flax NNX / Grain tutorial — 29 cells covering data, the Yat kernel, the model, training, predictions, and four interpretability views.

## Run on Kaggle

1. <https://kaggle.com/code> → **New notebook** → **File → Import notebook → Upload** `painting_arithmetic.ipynb`.
2. Right sidebar: **Accelerator → GPU T4 x1** (`sm_75`; the stock JAX wheel supports it). **Internet → On**.
3. **Run All**. ~3 minutes at the default config (10 epochs × 20 k samples).

For the paper number (~96.9% OCR), bump the `cfg` cell to `epochs=25, train_size=60_000`.

## Run locally

```bash
cd painting-arithmetic
pip install -e .
jupyter notebook notebooks/painting_arithmetic.ipynb
```

## Rebuild from source

The `.ipynb` is generated from `build_notebook.py` — edit the script, re-run, regenerate. Keeps the notebook diff-reviewable.

```bash
python notebooks/build_notebook.py
```
