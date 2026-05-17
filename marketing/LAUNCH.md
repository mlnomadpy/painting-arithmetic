# Launch runbook

A step-by-step checklist for going from "code complete" to "live across five channels in three weeks." Everything that doesn't require your credentials is already automated; you execute the rest.

## Prereqs (do these once, in any order)

- [ ] **Train a real checkpoint.** `painting-arithmetic-train --epochs 25 --train-size 60000` → produces `ckpts/model.npz`. ~5 min on a T4, ~13 min on Apple MPS.
- [ ] **Export ONNX for the browser demo.** `python scripts/export_onnx.py --ckpt ./ckpts/model.npz --out demo/model.onnx`.
- [ ] **Push the repo to GitHub** under `mlnomadpy/painting-arithmetic`. Don't make it public yet — wait until everything below resolves.
- [ ] **Regenerate marketing assets** if any numbers changed:
  - `python marketing/make_social_card.py` (no model needed)
  - `python marketing/make_prediction_gif.py` (needs a trained ckpt)

## Week 1 — quiet setup, no public posts

- [ ] **Hugging Face Space.** Run `marketing/deploy/deploy_hf_space.sh mlnomadpy painting-arithmetic` after `pip install huggingface_hub && huggingface-cli login`. Verify the URL loads and predictions work.
- [ ] **Update the placeholder URLs.** Search-and-replace `<USER>` / `<ARXIV-ID>` / `<REPO-URL>` / `<KAGGLE-URL>` in every `marketing/threads/*.md` and `marketing/posts/*.md` once the URLs are final.
- [ ] **Add the social-card meta tags to the live demo.** In `demo/index.html` `<head>`:
  ```html
  <meta property="og:image" content="https://github.com/mlnomadpy/painting-arithmetic/raw/main/marketing/social_card.png">
  <meta property="og:title" content="Painting Arithmetic">
  <meta property="og:description" content="A 0.8M-parameter rational-kernel network that paints its answer.">
  <meta property="twitter:card" content="summary_large_image">
  ```
- [ ] **arXiv submission.** Compile `paper/main.tex` → upload to arXiv. Submit on a Monday so it lists on Tuesday.
- [ ] **Make the GitHub repo public.** Pin the repo on your GitHub profile.
- [ ] **Record a 30–45 s screen capture** of you drawing in the live demo, the painted answer appearing, and scrolling the interpretability dashboard. Export as MP4 and as a fallback GIF.

## Week 2 — Twitter / Reddit

- [ ] **Tuesday, 8 AM US Eastern: post the X thread.** Use `marketing/threads/twitter_hybrid.md`. Attach `prediction.gif` in tweet 1 and `social_card.png` in tweet 3. Pin the thread.
- [ ] **Wednesday, 9 AM US Eastern: r/MachineLearning submission.** Use `marketing/posts/reddit_ml.md`. Reply to early comments within the first hour — moderators boost active threads.
- [ ] **Same day: Kaggle notebook publish.** Upload `notebooks/painting_arithmetic.ipynb` with the description from `marketing/posts/kaggle_description.md`. Set tags: `interpretability`, `mnist`, `jax`, `flax`.
- [ ] **Friday: LinkedIn post.** Use `marketing/posts/linkedin.md` with `social_card.png` attached. LinkedIn rewards Friday posting if your network is professional.

## Week 3 — long-tail

- [ ] **Tuesday: Hacker News.** Submit the HF Space URL with the title from `marketing/posts/hackernews.md`. Post the self-explainer comment immediately after.
- [ ] **Thursday: short-form video.** Same screen capture re-cut to vertical for TikTok / YouTube Shorts / Bluesky.
- [ ] **Cross-link from the `nmn` library README** — the flagship demo for Yat layers in the wild.

## After launch — monitoring

- [ ] Star counts and HF Space visits are vanity metrics; the actually useful signal is **issues / PRs** on the repo and **citations** of the arXiv paper. Set up a Google Scholar alert on the paper title.
- [ ] Reply to every Twitter / Reddit / HN comment in the first 48 h. Even a one-line acknowledgement keeps the thread alive on the recommender.

## Assets inventory

| file | purpose |
|---|---|
| `marketing/social_card.png` | 1200×630 OpenGraph image; attach to LinkedIn, embed in repo, use as Space thumbnail |
| `marketing/prediction.gif` | 12-frame painted-answer animation; main visual for Twitter / Reddit |
| `marketing/threads/twitter_hybrid.md` | recommended X thread draft |
| `marketing/threads/twitter_visual.md` | alternative — leads with the GIF |
| `marketing/threads/twitter_research.md` | alternative — leads with the Goodfire claim |
| `marketing/posts/reddit_ml.md` | r/MachineLearning [R] post |
| `marketing/posts/hackernews.md` | HN title + self-explainer comment |
| `marketing/posts/linkedin.md` | LinkedIn copy |
| `marketing/posts/kaggle_description.md` | description for the Kaggle notebook |
| `marketing/deploy/deploy_hf_space.sh` | one-command static HF Space deploy |
| `marketing/deploy/hf_space_readme.md` | front-matter README for the Space |

## What I (the agent) couldn't automate, and why

- **Pushing to GitHub** requires your `gh` auth and creates public state under your identity. Per your durable instructions I only push when you explicitly ask.
- **HF Space deploy** requires your `huggingface-cli login` token.
- **Twitter / LinkedIn / Reddit / HN posts** are public statements under your identity. These are deliberately one-keystroke-away-from-you.
- **arXiv submission** requires your endorsement / account.

Everything else — the asset rendering, the copy, the layout, the runbook — is in this folder, regenerable, and version-controlled.
