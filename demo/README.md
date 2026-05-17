# Live browser demo

A zero-server, fully client-side demo: three drawing pads, ONNX-Runtime-Web loads the trained model, every prediction runs in the browser via WebAssembly.

## Files

- `index.html` — the demo page. Expects `./model.onnx` next to it.
- `model.onnx` — produced by `scripts/export_onnx.py` after you train.

## Build the ONNX file

```bash
# 1. Train (5 min on a T4, 13 min on Apple MPS).
painting-arithmetic-train --epochs 25 --train-size 60000 --ckpt-dir ./ckpts

# 2. Export the model to ONNX.
python scripts/export_onnx.py --ckpt ./ckpts/model.npz --out demo/model.onnx
```

## Run locally

```bash
cd demo
python -m http.server 8000
# open http://localhost:8000
```

## Deploy

The directory is fully static — drop it on any static host.

- **GitHub Pages.** Push the repo, enable Pages for the `main` branch with directory `/demo`. Done.
- **Hugging Face Spaces.** Create a "Static" Space, upload `index.html` and `model.onnx` at the root.
- **Cloudflare Pages / Netlify / Vercel.** Point the build to `demo/`, no build command, publish directory `demo`. Done.

## How the inference works

1. The page renders the three 280×280 drawing canvases.
2. On each stroke-end, `Pad.tensor()` crops to the bounding box of non-zero pixels, places a centred 20×20 thumbnail in a 28×28 frame, and applies MNIST mean/std normalization.
3. `onnxruntime-web` runs the model in WASM and returns a 28×84 image.
4. The page paints that image (scaled 6×) into the output canvas.
5. A small nearest-neighbour over the 91 canonical target glyphs gives a numeric OCR readout below the image.

Total page payload: ≈ 3 MB ONNX + ≈ 200 KB ORT runtime + ≈ 12 KB HTML/JS.

## Notes

- The 91 target images are rendered in JS (no extra asset shipping needed). The OCR routine is a tight pixel-MSE nearest neighbour — fine for ≤ 100 candidates.
- The aux symbol head and other training-only outputs are not included in the ONNX export — only `img` (28×84) is. That keeps the ONNX small.
- For Hugging Face Spaces with a Python backend (live training, plotting, etc.) see the optional `demo-server/` folder in the repo (not included by default).
