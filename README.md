# bonsai2-mlx

Stock-runtime MLX builds of
[**prism-ml/Ternary-Bonsai-2-27B**](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf) —
a ternarized Qwen3.8-27B (`qwen3_5` hybrid GDN + full attention, with vision) whose released
weights are **blockwise-Hadamard-rotated** and normally require prism's bundled runtime.

This converter **unfolds the rotation** (`W·H·diag(s)` per 1024-block, order proven by negative
control; `refold(unfold(W))` bitwise exact) so the checkpoints load in **unmodified mlx-vlm ≥ 0.7**.
Parity against prism's own runtime: max|Δlogit| 0.22 (pure bf16-vs-fp16 dtype; 100% argmax in
fp16), paired wikitext-2 ppl ratio **0.9992 [0.9990, 0.9994]**.

Published: [bf16 54.7 GB](https://huggingface.co/pipenetwork/Ternary-Bonsai-2-27B-MLX-bf16) (the
unquantized standard-basis build) ·
[8-bit 29.5](https://huggingface.co/pipenetwork/Ternary-Bonsai-2-27B-MLX-8bit) ·
[6-bit 22.8](https://huggingface.co/pipenetwork/Ternary-Bonsai-2-27B-MLX-6bit) ·
[4-bit 16.1](https://huggingface.co/pipenetwork/Ternary-Bonsai-2-27B-MLX-4bit). ppl: 8.9679 /
8.9636 / 8.9548 / 9.1497 vs prism-2bit 8.9607. For the true efficiency frontier (8.6 GB, exactly
the ternary weights) use [prism's 2-bit pack](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-mlx-2bit)
with their runtime — a ternary model makes 2-bit affine lossless, which is why this set has no
2/3-bit rows.

| path | what |
|---|---|
| `bonsai2_mlx/core.py` | unfold/refold, GGUF→mlx-vlm tensor map, GDN v-head regroup, config builder |
| `scripts/convert_gguf.py` | F16 GGUF + mmproj → bf16 (100 s); `--from-bf16 --bits N` for the quants |
| `tests/test_parity.py` | fold-contract, runtime-vs-runtime parity, negative control, prism's validation files |
| `scripts/ppl_*.py`, `smoke_generate.py`, `check_strict_load.py`, `upload.py`, `make_collection.py` | eval + publishing |
