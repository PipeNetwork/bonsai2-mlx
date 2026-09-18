"""Convert Ternary-Bonsai-2-27B folded F16 GGUF (+ mmproj) to a STOCK mlx-vlm
qwen3_5 checkpoint, unfolding the blockwise Hadamard rotation.

bf16:
    python scripts/convert_gguf.py \
        --gguf /Users/david/llm/Bonsai2-gguf-src/Ternary-Bonsai-2-27B-F16.gguf \
        --mmproj /Users/david/llm/Bonsai2-gguf-src/Ternary-Bonsai-2-27B-mmproj-BF16.gguf \
        --pack /Users/david/llm/Bonsai2-mlx2bit-src \
        --out /Users/david/llm/bonsai2-out/bf16

quantized (from the bf16 dir, via mlx-vlm's own quantization machinery):
    python scripts/convert_gguf.py --from-bf16 /Users/david/llm/bonsai2-out/bf16 \
        --bits 4 --group-size 64 --out /Users/david/llm/bonsai2-out/4bit
"""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mlx.core as mx

from bonsai2_mlx.core import (
    build_config,
    convert_text_tensor,
    convert_vision_tensors,
    load_hadamard_spec,
    shard_and_save,
)

TOKENIZER_FILES = [
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "generation_config.json",
    "preprocessor_config.json",
]


def copy_aux(pack, out_dir):
    for f in TOKENIZER_FILES:
        src = Path(pack) / f
        if src.exists():
            shutil.copyfile(src, Path(out_dir) / f)


def convert_bf16(args):
    from gguf import GGUFReader

    t0 = time.time()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    reader = GGUFReader(args.gguf)
    fields = {k: v.contents() for k, v in reader.fields.items()}
    if fields.get("general.architecture") != "qwen35":
        raise ValueError("Expected qwen35 GGUF")
    block, signs, folded, inverse = load_hadamard_spec(fields)
    print(f"[convert] fold spec: block={block} folded={len(folded)} inverse={len(inverse)}")

    skip_unfold = set(args.skip_unfold or [])
    if skip_unfold:
        print(f"[convert] NEGATIVE CONTROL: skipping unfold for {sorted(skip_unfold)}")

    weights = {}
    n = 0
    for t in reader.tensors:
        key, arr = convert_text_tensor(
            t.name, t, signs, folded, inverse, skip_unfold=skip_unfold
        )
        mx.eval(arr)
        weights[key] = arr
        n += 1
        if n % 100 == 0:
            print(f"[convert] {n} text tensors ({time.time()-t0:.0f}s)")
            mx.clear_cache()
    print(f"[convert] text done: {n} tensors ({time.time()-t0:.0f}s)")

    if args.mmproj:
        vreader = GGUFReader(args.mmproj)
        vw = convert_vision_tensors(vreader)
        mx.eval(list(vw.values()))
        weights.update(vw)
        print(f"[convert] vision done: {len(vw)} tensors")

    cfg = build_config(Path(args.pack) / "config.json")
    with open(out_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    copy_aux(args.pack, out_dir)

    total = sum(v.nbytes for v in weights.values())
    print(f"[convert] saving {len(weights)} tensors, {total/1e9:.2f} GB")
    shard_and_save(str(out_dir), weights)
    print(f"[convert] wrote {out_dir} in {time.time()-t0:.0f}s")


def convert_quantized(args):
    from mlx_vlm.utils import load_model, save_weights, skip_multimodal_module
    from mlx_vlm.quant_utils import quantize_model

    t0 = time.time()
    src = Path(args.from_bf16)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[quant] loading {src}")
    model = load_model(src, lazy=True, strict=True)
    config = json.loads((src / "config.json").read_text())
    config.setdefault("vision_config", {})

    model_quant_predicate = getattr(model, "quant_predicate", None)

    def base_quant_predicate(path, module):
        # identical to mlx_vlm.convert's default: vision/audio stay unquantized
        if skip_multimodal_module(path):
            return False
        if model_quant_predicate is not None:
            return model_quant_predicate(path, module)
        return True

    print(f"[quant] quantizing bits={args.bits} group_size={args.group_size}")
    model, config = quantize_model(
        model,
        config,
        args.group_size,
        args.bits,
        mode="affine",
        quant_predicate=base_quant_predicate,
    )

    save_weights(out_dir, model, donate_weights=True)
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    copy_aux(src, out_dir)
    print(f"[quant] wrote {out_dir} in {time.time()-t0:.0f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", help="folded F16 GGUF")
    ap.add_argument("--mmproj", help="mmproj BF16 GGUF (vision tower)")
    ap.add_argument("--pack", help="prism mlx-2bit pack dir (config/tokenizer source)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--from-bf16", help="quantize from this bf16 checkpoint instead")
    ap.add_argument("--bits", type=int)
    ap.add_argument("--group-size", type=int, default=64)
    ap.add_argument(
        "--skip-unfold",
        nargs="*",
        help="GGUF tensor names whose unfold to skip (negative control)",
    )
    args = ap.parse_args()

    if args.from_bf16:
        if not args.bits:
            ap.error("--from-bf16 requires --bits")
        convert_quantized(args)
    else:
        if not (args.gguf and args.pack):
            ap.error("bf16 conversion requires --gguf and --pack")
        convert_bf16(args)


if __name__ == "__main__":
    main()
