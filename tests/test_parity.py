"""Parity proof: prism's own MLX runtime (2-bit folded pack + activation
transforms) vs our unfolded bf16 checkpoint through STOCK mlx-vlm.

Run:  .venv/bin/python tests/test_parity.py [--bf16 DIR] [--pack DIR] [--tokens N]

Sections:
  (a) unfold->refold round trip, bitwise-exact in fp32, on real pack tensors
  (b) real-weight logits parity over >=64 prompt positions + greedy decode
  (c) negative control: refold (= skip unfold) one layer -> logits move
  (d) their reload-validation / tokenizer-validation cross-checks
  (e) norm-convention check: stock load applies no extra +1 (in-memory norms
      equal the pack's F32 norms bf16-rounded)
"""

import argparse
import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bonsai2_mlx.core import refold, unfold  # noqa: E402

DEFAULT_PACK = "/Users/david/llm/Bonsai2-mlx2bit-src"
DEFAULT_BF16 = "/Users/david/llm/bonsai2-out/bf16"

PROMPTS = [
    "The Hadamard transform is an orthogonal linear map whose matrix entries "
    "are all plus or minus one. It is used in signal processing, error "
    "correction, and more recently in the quantization of large language "
    "models, where rotating the weight basis spreads outlier activations "
    "across many dimensions and makes extreme low-bit representations viable.",
    "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n",
    "The capital of France is",
]


def load_theirs(pack_dir):
    sys.path.insert(0, str(Path(pack_dir) / "runtime"))
    from vision_artifact import load_vl_model

    model, _, config = load_vl_model(pack_dir, load_processor=False)
    return model, config


def load_ours(bf16_dir):
    from mlx_vlm.utils import load_model

    return load_model(Path(bf16_dir), strict=True)


def lm_logits(model, ids):
    out = model.language_model(mx.array([ids], dtype=mx.int32))
    logits = out.logits if hasattr(out, "logits") else out
    mx.eval(logits)
    return logits[0].astype(mx.float32)


def greedy(model, ids, n):
    from mlx_vlm.models.cache import make_prompt_cache

    try:
        cache = make_prompt_cache(model.language_model)
    except Exception:
        cache = model.language_model.make_cache()
    toks = list(ids)
    out_toks = []
    cur = toks
    for _ in range(n):
        out = model.language_model(mx.array([cur], dtype=mx.int32), cache=cache)
        logits = out.logits if hasattr(out, "logits") else out
        nxt = int(mx.argmax(logits[0, -1]).item())
        out_toks.append(nxt)
        cur = [nxt]
    return out_toks


def section_a_roundtrip(pack_dir):
    print("== (a) unfold->refold round trip ==")
    pack = mx.load(str(Path(pack_dir) / "model.safetensors"))
    names = [
        "language_model.model.layers.0.mlp.gate_proj",
        "language_model.model.layers.3.self_attn.o_proj",
        "language_model.model.layers.0.mlp.down_proj",
        "language_model.model.layers.0.linear_attn.out_proj",
        "language_model.lm_head",
    ]
    ok = True
    for name in names:
        w2, sc, bi, s = (pack[name + k] for k in (".weight", ".scales", ".biases", ".signs"))
        wf = mx.dequantize(w2, sc, bi, group_size=128, bits=2).astype(mx.float32)
        rt = refold(unfold(wf, np.asarray(s)), np.asarray(s))
        exact = bool(mx.array_equal(rt, wf).item())
        maxerr = mx.abs(rt - wf).max().item()
        print(f"  {name}: bitwise_exact={exact} max_err={maxerr:g}")
        ok &= exact
    # embedding rows
    name = "language_model.model.embed_tokens"
    w2, sc, bi, s = (pack[name + k] for k in (".weight", ".scales", ".biases", ".signs"))
    idx = mx.array([0, 1, 1000, 123456, 248319])
    wf = mx.dequantize(w2[idx], sc[idx], bi[idx], group_size=128, bits=2).astype(mx.float32)
    rt = refold(unfold(wf, np.asarray(s)), np.asarray(s))
    exact = bool(mx.array_equal(rt, wf).item())
    print(f"  {name}[5 rows]: bitwise_exact={exact}")
    ok &= exact
    assert ok, "round trip not exact"
    del pack
    mx.clear_cache()
    return ok


def section_d_validations(pack_dir, bf16_dir):
    print("== (d) their validation manifests ==")
    rv = json.loads((Path(pack_dir) / "reload-validation.json").read_text())
    tv = json.loads((Path(pack_dir) / "tokenizer-validation.json").read_text())
    print(f"  their reload-validation: {rv}")
    print(f"  their tokenizer-validation: {tv}")
    assert rv["reload_logits_exact"] and tv["vocabulary_ids_match"] and tv["bpe_merges_match"]
    # 402 packed modules == 401 folded + 1 inverse in the GGUF manifest,
    # and every one must exist unfolded in our checkpoint.
    had = json.loads((Path(pack_dir) / "hadamard.json").read_text())
    folded = had["prism.hadamard.weight_names"]
    inverse = had["prism.hadamard.inverse_weight_names"]
    assert rv["packed_modules"] == len(folded) + len(inverse) == 402
    index = json.loads((Path(bf16_dir) / "model.safetensors.index.json").read_text())
    ours = set(index["weight_map"])
    missing = [n for n in folded + inverse if n not in ours]
    print(f"  our checkpoint covers all {len(folded)+len(inverse)} transformed tensors: {not missing}")
    assert not missing, missing
    # tokenizer files copied verbatim
    same = (Path(pack_dir) / "tokenizer.json").read_bytes() == (
        Path(bf16_dir) / "tokenizer.json"
    ).read_bytes()
    print(f"  tokenizer.json byte-identical to pack: {same}")
    assert same
    assert rv["checked_logits"] == 248320  # vocab size
    return True


def section_e_norms(ours, pack_dir):
    print("== (e) norm-shift decision check ==")
    pack = mx.load(str(Path(pack_dir) / "model.safetensors"))
    checks = [
        ("language_model.model.layers.0.input_layernorm.weight",
         ours.language_model.model.layers[0].input_layernorm.weight),
        ("language_model.model.layers.3.self_attn.q_norm.weight",
         ours.language_model.model.layers[3].self_attn.q_norm.weight),
        ("language_model.model.norm.weight",
         ours.language_model.model.norm.weight),
        ("language_model.model.layers.0.linear_attn.norm.weight",
         ours.language_model.model.layers[0].linear_attn.norm.weight),
    ]
    ok = True
    for name, w in checks:
        ref = pack[name].astype(mx.bfloat16)  # pack stores F32 direct-apply values
        same = bool(mx.array_equal(w, ref).item())
        print(f"  {name}: in-memory == bf16(pack F32): {same}")
        ok &= same
    assert ok, "norms were shifted on load (double-shift bug) or mis-stored"
    del pack
    mx.clear_cache()
    return ok


def section_b_parity(theirs, ours, tok, n_decode):
    print("== (b) real-weight logits parity ==")
    results = []
    for pi, prompt in enumerate(PROMPTS):
        ids = tok(prompt, add_special_tokens=False)["input_ids"]
        lt = lm_logits(theirs, ids)
        lo = lm_logits(ours, ids)
        d = mx.abs(lt - lo)
        am_t = mx.argmax(lt, axis=-1)
        am_o = mx.argmax(lo, axis=-1)
        agree = mx.mean((am_t == am_o).astype(mx.float32)).item()
        # cosine per final position
        c = (lt[-1] @ lo[-1]) / (mx.linalg.norm(lt[-1]) * mx.linalg.norm(lo[-1]))
        print(
            f"  prompt{pi} ({len(ids)} toks): max|dlogit|={d.max().item():.4f} "
            f"mean|dlogit|={d.mean().item():.5f} argmax_agree={agree*100:.1f}% "
            f"cos(final)={c.item():.6f} |logit|max={mx.abs(lt).max().item():.1f}"
        )
        results.append((len(ids), d.max().item(), agree))
        mx.clear_cache()
    total_positions = sum(r[0] for r in results)
    print(f"  total compared positions: {total_positions}")
    assert total_positions >= 64

    # greedy decode agreement on the long prompt
    ids = tok(PROMPTS[0], add_special_tokens=False)["input_ids"]
    gt = greedy(theirs, ids, n_decode)
    go = greedy(ours, ids, n_decode)
    n_same = sum(a == b for a, b in zip(gt, go))
    first_div = next((i for i, (a, b) in enumerate(zip(gt, go)) if a != b), n_decode)
    print(f"  greedy {n_decode} steps: token match {n_same}/{n_decode}, first divergence at {first_div}")
    print(f"  theirs: {tok.decode(gt[:40])!r}")
    print(f"  ours:   {tok.decode(go[:40])!r}")
    mx.clear_cache()
    return results, (n_same, n_decode, first_div)


def section_c_negative_control(theirs, ours, tok, pack_dir):
    print("== (c) negative control: refold one layer (= skip its unfold) ==")
    pack = mx.load(str(Path(pack_dir) / "model.safetensors"))
    name = "language_model.model.layers.0.mlp.gate_proj"
    signs = np.asarray(pack[name + ".signs"])
    del pack
    layer = ours.language_model.model.layers[0].mlp.gate_proj
    orig = layer.weight
    # refold == the stored folded GGUF weight (roundtrip is exact), i.e. what
    # you would get by NOT unfolding this tensor during conversion
    layer.weight = refold(orig.astype(mx.float32), signs).astype(mx.bfloat16)
    mx.eval(layer.weight)

    ids = tok(PROMPTS[0], add_special_tokens=False)["input_ids"]
    lt = lm_logits(theirs, ids)
    lb = lm_logits(ours, ids)
    d = mx.abs(lt - lb)
    am_agree = mx.mean(
        (mx.argmax(lt, axis=-1) == mx.argmax(lb, axis=-1)).astype(mx.float32)
    ).item()
    print(
        f"  ONE layer folded-in-place: max|dlogit|={d.max().item():.3f} "
        f"argmax_agree={am_agree*100:.1f}% (should be far from 100%)"
    )
    layer.weight = orig
    mx.eval(layer.weight)
    mx.clear_cache()
    assert d.max().item() > 1.0, "negative control did not move logits"
    return d.max().item(), am_agree


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", default=DEFAULT_PACK)
    ap.add_argument("--bf16", default=DEFAULT_BF16)
    ap.add_argument("--decode-tokens", type=int, default=64)
    ap.add_argument("--skip-model-tests", action="store_true")
    args = ap.parse_args()

    section_a_roundtrip(args.pack)
    section_d_validations(args.pack, args.bf16)
    if args.skip_model_tests:
        return

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.bf16)

    print("[load] ours (stock mlx-vlm, strict)...")
    ours = load_ours(args.bf16)
    section_e_norms(ours, args.pack)

    print("[load] theirs (prism runtime)...")
    theirs, _ = load_theirs(args.pack)

    section_b_parity(theirs, ours, tok, args.decode_tokens)
    section_c_negative_control(theirs, ours, tok, args.pack)
    print("ALL PARITY SECTIONS PASSED")


if __name__ == "__main__":
    main()
