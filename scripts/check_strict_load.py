"""Strict-load a checkpoint through STOCK mlx_vlm and prove every parameter is
real (no silent random-init: mlx_vlm.load_model defaults to strict=True, but we
also cross-check the checkpoint keys against the model's parameter tree).

    .venv/bin/python scripts/check_strict_load.py <MODEL_DIR>
"""

import json
import sys
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten


def main():
    model_dir = Path(sys.argv[1])
    from mlx_vlm.utils import load_model

    model = load_model(model_dir, strict=True)
    params = dict(tree_flatten(model.parameters()))
    n_params = sum(v.size for v in params.values())
    n_bytes = sum(v.nbytes for v in params.values())
    print(f"[strict] loaded {model_dir}")
    print(f"[strict] {len(params)} arrays, {n_params/1e9:.3f}B params, {n_bytes/1e9:.2f} GB in memory")

    # cross-check: every checkpoint tensor landed somewhere in the module tree
    index = model_dir / "model.safetensors.index.json"
    if index.exists():
        weight_map = json.loads(index.read_text())["weight_map"]
        ckpt_keys = set(weight_map)
    else:
        ckpt_keys = set(mx.load(str(model_dir / "model.safetensors")))
    param_keys = set(params)
    missing_in_model = sorted(ckpt_keys - param_keys)
    missing_in_ckpt = sorted(param_keys - ckpt_keys)
    print(f"[strict] checkpoint tensors not in model tree: {len(missing_in_model)}")
    for k in missing_in_model[:10]:
        print("   ", k)
    print(f"[strict] model params not in checkpoint: {len(missing_in_ckpt)}")
    for k in missing_in_ckpt[:10]:
        print("   ", k)
    if missing_in_ckpt:
        raise SystemExit("FAIL: model has parameters the checkpoint did not provide")
    dtypes = {}
    for k, v in params.items():
        dtypes[str(v.dtype)] = dtypes.get(str(v.dtype), 0) + 1
    print(f"[strict] dtype histogram: {dtypes}")
    print("[strict] OK")


if __name__ == "__main__":
    main()
