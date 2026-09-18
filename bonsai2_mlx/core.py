"""Core conversion logic: Ternary-Bonsai-2-27B folded GGUF -> stock mlx-vlm qwen3_5.

Verified facts (against prism's own MLX pack + runtime, 2026-09-18):

- The GGUF F16 file stores the FOLDED ternary weights (dequantized). The pack's
  2-bit affine dequant is bit-identical to the GGUF F16 values (checked).
- Fold contract (prism.hadamard.* KVs, version 1): block 1024,
  transform=normalized-sylvester-walsh-hadamard, axis=input-last-dimension,
  sign_mode=explicit. Their runtime computes, for a folded linear,
      y = quantized_matmul(fwht(x), W_fold)   with fwht(x) = ((x * s) @ H) / sqrt(b)
  and for the (inverse-listed) embedding,
      e = (dequant_row @ H) / sqrt(b) * s
  H is the Sylvester Hadamard (symmetric, orthogonal with the 1/sqrt(b) scale).
  Therefore the unfold for BOTH the 401 forward-folded weights and the
  inverse-folded token embedding is the same row-space operation:
      W_orig[j, :] = (W_fold[j, :] @ H) / sqrt(b) * s        (per 1024-block)
  (For a linear, y @ W_fold^T = x diag(s) H W_fold^T = x W_orig^T with
   W_orig = W_fold H diag(s); for the embedding it is literally their inverse
   transform applied to the stored rows.)
- Signs: three shared +-1 vectors keyed by input width {5120, 6144, 17408},
  concatenated in prism.hadamard.sign_values (bit-equal to the per-module
  .signs tensors in the MLX pack; checked).
- GDN v-head regrouping: GGUF stores value-head-indexed rows in llama.cpp
  (rep-major) order; mlx wants (k-group-major). vperm from their runtime.py
  applies to attn_qkv v-rows, attn_gate rows, ssm_alpha/beta rows, ssm_a,
  ssm_dt.bias and the conv1d v-channels. ssm_out columns are already grouped
  (prism.hadamard.gdn_v_grouped=true; checked A_log/conv1d/in_proj_a).
- Norms in the GGUF are in the direct-apply convention (~1-centred, +1 baked,
  bit-equal to the pack's F32 norms; checked). We store them as-is together
  with a sanitized conv1d (C, K, 1), so stock mlx-vlm's qwen3_5 sanitize()
  (guarded since ~0.6.5: shifts language_model.* norms only when it sees an
  unsanitized conv1d or MTP weights) applies NO further +1 on load.
- Vision (mmproj BF16): mapping verified bit-exact against the pack's fp16
  vision tower for every tensor kind, including the split temporal patch_embd
  (stack the two GGUF tensors on axis 2, then transpose to mlx channel-last).
"""

import json
import math

import mlx.core as mx
import numpy as np

BLOCK = 1024
NV, NK = 48, 16  # linear_num_value_heads, linear_num_key_heads
HK = 128  # linear_key_head_dim
HD = 128  # linear_value_head_dim
QK_ROWS = 2 * NK * HK  # 4096: q+k rows of in_proj_qkv / conv channels

LM_PREFIX = "language_model."

# GGUF blk-stem -> mlx-vlm layer-relative name (their runtime.py mapping,
# prefixed into the mlx-vlm namespace).
TEXT_STEM_MAP = {
    "attn_norm.weight": "input_layernorm.weight",
    "post_attention_norm.weight": "post_attention_layernorm.weight",
    "ffn_gate.weight": "mlp.gate_proj.weight",
    "ffn_up.weight": "mlp.up_proj.weight",
    "ffn_down.weight": "mlp.down_proj.weight",
    "attn_q.weight": "self_attn.q_proj.weight",
    "attn_k.weight": "self_attn.k_proj.weight",
    "attn_v.weight": "self_attn.v_proj.weight",
    "attn_output.weight": "self_attn.o_proj.weight",
    "attn_q_norm.weight": "self_attn.q_norm.weight",
    "attn_k_norm.weight": "self_attn.k_norm.weight",
    "attn_qkv.weight": "linear_attn.in_proj_qkv.weight",
    "attn_gate.weight": "linear_attn.in_proj_z.weight",
    "ssm_alpha.weight": "linear_attn.in_proj_a.weight",
    "ssm_beta.weight": "linear_attn.in_proj_b.weight",
    "ssm_out.weight": "linear_attn.out_proj.weight",
    "ssm_norm.weight": "linear_attn.norm.weight",
    "ssm_a": "linear_attn.A_log",
    "ssm_dt.bias": "linear_attn.dt_bias",
    "ssm_conv1d.weight": "linear_attn.conv1d.weight",
}

TEXT_GLOBAL_MAP = {
    "output.weight": LM_PREFIX + "lm_head.weight",
    "output_norm.weight": LM_PREFIX + "model.norm.weight",
    "token_embd.weight": LM_PREFIX + "model.embed_tokens.weight",
}

# mmproj GGUF name -> mlx-vlm vision name (block stems)
VISION_STEM_MAP = {
    "attn_qkv.weight": "attn.qkv.weight",
    "attn_qkv.bias": "attn.qkv.bias",
    "attn_out.weight": "attn.proj.weight",
    "attn_out.bias": "attn.proj.bias",
    "ffn_up.weight": "mlp.linear_fc1.weight",
    "ffn_up.bias": "mlp.linear_fc1.bias",
    "ffn_down.weight": "mlp.linear_fc2.weight",
    "ffn_down.bias": "mlp.linear_fc2.bias",
    "ln1.weight": "norm1.weight",
    "ln1.bias": "norm1.bias",
    "ln2.weight": "norm2.weight",
    "ln2.bias": "norm2.bias",
}

VISION_GLOBAL_MAP = {
    "v.position_embd.weight": "vision_tower.pos_embed.weight",
    "v.patch_embd.bias": "vision_tower.patch_embed.proj.bias",
    "v.post_ln.weight": "vision_tower.merger.norm.weight",
    "v.post_ln.bias": "vision_tower.merger.norm.bias",
    "mm.0.weight": "vision_tower.merger.linear_fc1.weight",
    "mm.0.bias": "vision_tower.merger.linear_fc1.bias",
    "mm.2.weight": "vision_tower.merger.linear_fc2.weight",
    "mm.2.bias": "vision_tower.merger.linear_fc2.bias",
}


def gguf_np(tensor):
    """GGUF tensor -> numpy array in logical (row-major) shape.

    The gguf reader already reverses GGML's ne order for F16/F32. BF16 comes
    back as uint8 with a doubled last dim; widen to float32.
    """
    d = tensor.data
    if d.dtype in (np.float16, np.float32):
        return np.asarray(d)
    if tensor.tensor_type.name == "BF16":
        u16 = np.ascontiguousarray(d).view(np.uint16)
        return (u16.astype(np.uint32) << 16).view(np.float32)
    raise ValueError(f"Unsupported GGUF tensor type {tensor.tensor_type.name}")


def load_hadamard_spec(fields):
    """Parse prism.hadamard.* KVs from a GGUF fields dict (contents() form)."""
    if fields.get("prism.hadamard.version") != 1:
        raise ValueError("Unsupported prism.hadamard.version")
    if fields.get("prism.hadamard.transform") != "normalized-sylvester-walsh-hadamard":
        raise ValueError("Unexpected transform")
    if fields.get("prism.hadamard.axis") != "input-last-dimension":
        raise ValueError("Unexpected transform axis")
    if fields.get("prism.hadamard.sign_mode") != "explicit":
        raise ValueError("Explicit signs required")
    block = int(fields["prism.hadamard.block_size"])
    widths = [int(w) for w in fields["prism.hadamard.sign_widths"]]
    values = np.asarray(fields["prism.hadamard.sign_values"], dtype=np.float32)
    signs, off = {}, 0
    for w in widths:
        v = values[off : off + w]
        if len(v) != w or not np.isin(v, (-1.0, 1.0)).all():
            raise ValueError("Invalid sign vector")
        signs[w] = v
        off += w
    if off != len(values):
        raise ValueError("Trailing sign values")
    folded = set(fields["prism.hadamard.weight_names"])
    inverse = set(fields.get("prism.hadamard.inverse_weight_names", []))
    if folded & inverse:
        raise ValueError("Fold manifests overlap")
    if not fields.get("prism.hadamard.gdn_v_grouped", False):
        raise ValueError("Ungrouped folded GDN output is not implemented")
    return block, signs, folded, inverse


def unfold(w, signs, block=BLOCK):
    """Undo the Hadamard fold on the input (last) dimension.

    w: (rows, cols) mx or np array (any float dtype); signs: (cols,) +-1.
    Returns float32 mx array: per 1024-block  (w @ H) / sqrt(block), then * signs.
    """
    w = mx.array(w).astype(mx.float32)
    rows, cols = w.shape
    if cols % block:
        raise ValueError(f"cols {cols} not divisible by block {block}")
    s = mx.array(np.asarray(signs, dtype=np.float32)).reshape(1, cols)
    out = mx.hadamard_transform(
        w.reshape(rows * (cols // block), block), scale=1.0 / math.sqrt(block)
    ).reshape(rows, cols)
    return out * s


def refold(w, signs, block=BLOCK):
    """Redo the fold (inverse of unfold): per-block ((w * signs) @ H) / sqrt(block)."""
    w = mx.array(w).astype(mx.float32)
    rows, cols = w.shape
    s = mx.array(np.asarray(signs, dtype=np.float32)).reshape(1, cols)
    w = w * s
    return mx.hadamard_transform(
        w.reshape(rows * (cols // block), block), scale=1.0 / math.sqrt(block)
    ).reshape(rows, cols)


def vperm(unit, nv=NV, nk=NK):
    """Row index permutation from llama.cpp (rep-major) to mlx (group-major)."""
    return (
        np.arange(nv * unit).reshape(nv // nk, nk, unit).transpose(1, 0, 2).reshape(-1)
    )


def reorder_gdn(a, stem):
    """Apply the GDN v-head regrouping their runtime applies (rows / channels)."""
    if stem == "attn_qkv.weight":
        return np.concatenate([a[:QK_ROWS], a[QK_ROWS:][vperm(HD)]], axis=0)
    if stem == "attn_gate.weight":
        return a[vperm(HD)]
    if stem in ("ssm_alpha.weight", "ssm_beta.weight", "ssm_a", "ssm_dt.bias"):
        return a[vperm(1)]
    if stem == "ssm_conv1d.weight":
        return np.concatenate([a[:QK_ROWS], a[QK_ROWS:][vperm(HD)]], axis=0)
    return a


def convert_text_tensor(name, tensor, signs, folded, inverse, skip_unfold=()):
    """One GGUF text tensor -> (mlx_key, mx.array ready to store).

    skip_unfold: iterable of GGUF names whose unfold is deliberately skipped
    (negative-control testing only).
    """
    stem = name
    if name in TEXT_GLOBAL_MAP:
        key = TEXT_GLOBAL_MAP[name]
    elif name.startswith("blk."):
        _, layer, stem = name.split(".", 2)
        if stem not in TEXT_STEM_MAP:
            raise ValueError(f"Unmapped tensor {name}")
        key = f"{LM_PREFIX}model.layers.{layer}." + TEXT_STEM_MAP[stem]
    else:
        raise ValueError(f"Unmapped tensor {name}")

    a = gguf_np(tensor)
    a = reorder_gdn(a, stem)

    if stem == "ssm_a":
        if not (a < 0).all():
            raise ValueError("Invalid stored SSM A")
        # keep float32, matching mlx-vlm's cast_predicate (A_log is never cast)
        return key, mx.array(np.log(-a).astype(np.float32))
    if stem == "ssm_conv1d.weight":
        # store sanitized mlx layout (C, K, 1) so stock sanitize() adds no norm shift
        return key, mx.array(a.astype(np.float32)[..., None]).astype(mx.bfloat16)

    is_folded = name in folded or name in inverse
    if name in skip_unfold:
        is_folded = False
    if is_folded:
        w = unfold(a, signs[a.shape[-1]])
        return key, w.astype(mx.bfloat16)
    return key, mx.array(a.astype(np.float32)).astype(mx.bfloat16)


def convert_vision_tensors(reader):
    """mmproj GGUF reader -> dict of mlx-vlm vision_tower.* bf16 tensors."""
    tensors = {t.name: t for t in reader.tensors}
    out = {}
    patch0 = patch1 = None
    for name, t in tensors.items():
        if name == "v.patch_embd.weight":
            patch0 = gguf_np(t)
            continue
        if name == "v.patch_embd.weight.1":
            patch1 = gguf_np(t)
            continue
        if name in VISION_GLOBAL_MAP:
            key = VISION_GLOBAL_MAP[name]
        elif name.startswith("v.blk."):
            _, _, layer, stem = name.split(".", 3)
            if stem not in VISION_STEM_MAP:
                raise ValueError(f"Unmapped vision tensor {name}")
            key = f"vision_tower.blocks.{layer}." + VISION_STEM_MAP[stem]
        else:
            raise ValueError(f"Unmapped vision tensor {name}")
        out[key] = mx.array(gguf_np(t).astype(np.float32)).astype(mx.bfloat16)
    if patch0 is None or patch1 is None:
        raise ValueError("Missing patch_embd temporal slices")
    # (O, C, H, W) x2 -> (O, C, T, H, W) -> mlx channel-last (O, T, H, W, C)
    hf = np.stack([patch0, patch1], axis=2)
    out["vision_tower.patch_embed.proj.weight"] = mx.array(
        hf.transpose(0, 2, 3, 4, 1).astype(np.float32)
    ).astype(mx.bfloat16)
    return out


def build_config(pack_config_path):
    """Stock mlx-vlm qwen3_5 config from prism's pack config (HF-ish layout)."""
    pc = json.loads(open(pack_config_path).read())
    text = dict(pc["text_config"])
    vision = dict(pc["vision_config"])
    cfg = {
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "model_type": "qwen3_5",
        "text_config": text,
        "vision_config": vision,
        "image_token_id": pc["image_token_id"],
        "video_token_id": pc["video_token_id"],
        "vision_start_token_id": pc["vision_start_token_id"],
        "vision_end_token_id": pc["vision_end_token_id"],
        "tie_word_embeddings": pc.get("tie_word_embeddings", False),
        "vocab_size": text["vocab_size"],
        "torch_dtype": "bfloat16",
    }
    return cfg


def shard_and_save(out_dir, weights, max_shard_bytes=int(4.8e9)):
    """Write weights as mlx-style shards + index (mirrors mlx-vlm save_weights)."""
    import os

    os.makedirs(out_dir, exist_ok=True)
    items = list(weights.items())
    shards, cur, cur_bytes = [], {}, 0
    for k, v in items:
        nbytes = v.nbytes
        if cur and cur_bytes + nbytes > max_shard_bytes:
            shards.append(cur)
            cur, cur_bytes = {}, 0
        cur[k] = v
        cur_bytes += nbytes
    if cur:
        shards.append(cur)
    n = len(shards)
    fmt = "model-{:05d}-of-{:05d}.safetensors" if n > 1 else "model.safetensors"
    index = {
        "metadata": {"total_size": sum(v.nbytes for _, v in items)},
        "weight_map": {},
    }
    for i, shard in enumerate(shards):
        fname = fmt.format(i + 1, n)
        mx.save_safetensors(
            os.path.join(out_dir, fname), shard, metadata={"format": "mlx"}
        )
        for k in shard:
            index["weight_map"][k] = fname
        del shard
        mx.clear_cache()
    with open(os.path.join(out_dir, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f, indent=2)
