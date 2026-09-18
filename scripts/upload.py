"""Publish a Ternary-Bonsai-2-27B MLX build, card rendered from the measurements.

    .venv/bin/python scripts/upload.py --dir <build dir> --repo pipenetwork/<name> [--yes] [--card-only]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = Path("/Users/david/llm/bonsai2-out")
UPSTREAM = "prism-ml/Ternary-Bonsai-2-27B-gguf"
PRISM_MLX = "prism-ml/Ternary-Bonsai-2-27B-mlx-2bit"
CODE_REPO = "https://github.com/PipeNetwork/bonsai2-mlx"
NAMES = {"bf16": "Ternary-Bonsai-2-27B-MLX-bf16", "8bit": "Ternary-Bonsai-2-27B-MLX-8bit",
         "6bit": "Ternary-Bonsai-2-27B-MLX-6bit", "4bit": "Ternary-Bonsai-2-27B-MLX-4bit"}
LADDER = ["bf16", "8bit", "6bit", "4bit"]

CARD = """---
license: apache-2.0
base_model: {upstream}
base_model_relation: quantized
tags:
- mlx
- apple-silicon
- qwen3_5
- ternary
- vision-language
pipeline_tag: image-text-to-text
library_name: mlx
---

# {repo_name}

**Stock-runtime MLX build** of
[**Ternary-Bonsai-2-27B**](https://huggingface.co/{upstream}) — prism-ml's ternarized
Qwen3.8-27B (64-layer `qwen3_5` hybrid: Gated-DeltaNet + every-4th full attention, with the
official vision tower) — with the **blockwise Hadamard rotation unfolded** back into the standard
weight basis, so it loads in **unmodified mlx-vlm (≥ 0.7)**: no custom runtime, no forked kernels.

{flavor}

```python
from mlx_vlm import load, generate
model, processor = load("pipenetwork/{repo_name}")
```

## How this relates to prism-ml's own MLX release

prism-ml ships an excellent [8.6 GB 2-bit pack]({prism_url}) whose weights are the exact ternary
values — **it is the efficiency frontier for this model**, and it requires their bundled runtime
(the stored weights are Hadamard-rotated; activations are transformed to match). This set serves
the complementary case: standard-basis weights for stock tooling, fine-tuning, and downstream
conversion. The unfold is exact — `refold(unfold(W))` is bitwise identical in fp32, and the fold
contract was verified against their runtime, not assumed (applying the sign vector in the wrong
order moves logits by 7.4; the test catches it).

## Fidelity

Against prism's 2-bit pack running under **their** runtime, same prompts, 81 positions: max
|Δlogit| 0.22 on a ±20 scale, cosine 0.99999, argmax 96.7–100% (the flips are ties; in fp16 —
their activation dtype — agreement is 100%). Paired over 145 shared wikitext-2 windows the
perplexity ratio is 0.9992 [0.9990, 0.9994]: a bf16-rounding-level difference.

**Perplexity** (wikitext-2 test, 296,815 tokens, identical windows through stock mlx-vlm):

| build | size | ppl |
|---|---:|---:|
| [prism 2-bit, their runtime]({prism_url}) | 8.6 GB | 8.9607 |
| [bf16 (this set's unquantized)](https://huggingface.co/pipenetwork/Ternary-Bonsai-2-27B-MLX-bf16) | 54.7 GB | 8.9679 |
| [8-bit](https://huggingface.co/pipenetwork/Ternary-Bonsai-2-27B-MLX-8bit) | 29.5 GB | 8.9636 |
| [6-bit](https://huggingface.co/pipenetwork/Ternary-Bonsai-2-27B-MLX-6bit) | 22.8 GB | 8.9548 |
| [4-bit](https://huggingface.co/pipenetwork/Ternary-Bonsai-2-27B-MLX-4bit) | 16.1 GB | 9.1497 |

8-bit and 6-bit are statistically indistinguishable from bf16; 4-bit costs +2.1% — the only
build with a measurable loss, and still the smallest stock-loadable one. Vision verified
end-to-end. Requires **mlx-vlm ≥ 0.7** (earlier versions double-shift `qwen3_5` norms).

## License

Apache-2.0, as upstream ({upstream}); their NOTICE.txt is included. Conversion code:
[{code_repo}]({code_repo}).
"""

FLAVOR = {
 "bf16": "This is the **unquantized** build — the full-precision standard-basis weights (the ternary values, exactly, in bf16), the right starting point for fine-tuning or further conversion.",
 "8bit": "8-bit affine (group 64): statistically indistinguishable from bf16 at 54% of the size.",
 "6bit": "6-bit affine (group 64): statistically indistinguishable from bf16 at 42% of the size.",
 "4bit": "4-bit affine (group 64): +2.1% perplexity, the smallest stock-loadable build. For the true efficiency frontier (8.6 GB, lossless ternary) use prism-ml's own 2-bit pack with their runtime.",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True); ap.add_argument("--repo", required=True)
    ap.add_argument("--yes", action="store_true"); ap.add_argument("--card-only", action="store_true")
    args = ap.parse_args()
    d = Path(args.dir); name = args.repo.split("/")[-1]
    key = next(k for k, v in NAMES.items() if v == name)
    gb = sum(p.stat().st_size for p in d.iterdir() if p.is_file()) / 1e9
    card = CARD.format(upstream=UPSTREAM, prism_url=f"https://huggingface.co/{PRISM_MLX}", code_repo=CODE_REPO,
                       repo_name=name, flavor=FLAVOR[key])
    (d / "README.md").write_text(card)
    print(f"repo {args.repo}\ndir {d}\nfiles {sum(1 for p in d.iterdir() if p.is_file())}, {gb:.1f} GB")
    if not args.yes:
        print("dry run — pass --yes to upload"); return 0
    from huggingface_hub import HfApi
    import time
    api = HfApi()
    if args.card_only:
        api.upload_file(path_or_fileobj=str(d / "README.md"), path_in_repo="README.md", repo_id=args.repo, repo_type="model")
        print(f"card refreshed https://huggingface.co/{args.repo}"); return 0
    api.create_repo(args.repo, exist_ok=True, repo_type="model")
    for _ in range(30):
        try: api.model_info(args.repo); break
        except Exception: time.sleep(2)
    api.upload_folder(folder_path=str(d), repo_id=args.repo, repo_type="model")
    print(f"uploaded https://huggingface.co/{args.repo}"); return 0


if __name__ == "__main__":
    raise SystemExit(main())
