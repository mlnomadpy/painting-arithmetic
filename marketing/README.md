# Marketing kit

Everything needed to launch this project across X, Reddit, Hacker News, Kaggle, LinkedIn, and Hugging Face. All the asset generation and copy is committed; you handle the public posts.

## Layout

```
marketing/
├── LAUNCH.md                       ← runbook (week-by-week checklist)
├── README.md                       ← this file
│
├── social_card.png                 ← 1200×630 OpenGraph image
├── prediction.gif                  ← 12-frame painted-answer animation
│
├── make_social_card.py             ← regenerates social_card.png
├── make_prediction_gif.py          ← regenerates prediction.gif from a trained ckpt
│
├── threads/
│   ├── twitter_hybrid.md           ← recommended X thread
│   ├── twitter_visual.md           ← alt: lead with the GIF
│   └── twitter_research.md         ← alt: lead with the Goodfire claim
│
├── posts/
│   ├── reddit_ml.md                ← r/MachineLearning [R] submission
│   ├── hackernews.md               ← HN title + self-explainer
│   ├── linkedin.md                 ← LinkedIn post
│   └── kaggle_description.md       ← description for the Kaggle notebook
│
└── deploy/
    ├── deploy_hf_space.sh          ← one-command HF Space deploy (needs `huggingface-cli login`)
    └── hf_space_readme.md          ← README that lives inside the Space
```

## Quick start

```bash
# 1. Regenerate the assets (no network, ~10 s).
python marketing/make_social_card.py

# 2. Train a model (~5 min on T4 / ~13 min on Apple MPS).
painting-arithmetic-train --epochs 25 --train-size 60000

# 3. Regenerate the prediction GIF (needs the ckpt).
python marketing/make_prediction_gif.py

# 4. Export ONNX and deploy the static HF Space.
python scripts/export_onnx.py --ckpt ./ckpts/model.npz --out demo/model.onnx
huggingface-cli login                          # one-time
./marketing/deploy/deploy_hf_space.sh mlnomadpy painting-arithmetic
```

## Then read `LAUNCH.md`

It has the three-week posting cadence, copy variants, and the answer to "what should I post on which day."
