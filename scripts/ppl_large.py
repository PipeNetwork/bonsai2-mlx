"""Perplexity over the shared held-out corpus — identical windows for every build.

Loads through STOCK mlx_vlm (strict). Also accepts --prism-pack to score
prism's own 2-bit pack through THEIR runtime on the same windows.

    .venv/bin/python scripts/ppl_large.py <MODEL_DIR> [CORPUS.npy] [SEQ_LEN] [RESULTS.json]
    .venv/bin/python scripts/ppl_large.py --prism-pack <PACK_DIR> [CORPUS.npy] [SEQ_LEN] [RESULTS.json]
"""
import json, math, os, sys, time
import numpy as np
import mlx.core as mx

HERE = os.path.dirname(os.path.abspath(__file__))

args = [a for a in sys.argv[1:]]
PRISM = "--prism-pack" in args
if PRISM:
    args.remove("--prism-pack")
MODEL = args[0]
CORPUS = args[1] if len(args) > 1 else os.path.join(HERE, "..", "ppl_corpus.npy")
SEQ = int(args[2]) if len(args) > 2 else 2048
OUT = args[3] if len(args) > 3 else os.path.join(HERE, "..", "ppl_results.json")

try:
    mx.set_wired_limit(int(440e9))
except Exception as e:
    print("[warn]", e, flush=True)


def bootstrap_ci(win_nll, win_tok, n=2000, seed=0):
    rng = np.random.default_rng(seed); k = len(win_nll)
    idx = rng.integers(0, k, size=(n, k))
    ppl = np.exp(win_nll[idx].sum(1) / win_tok[idx].sum(1))
    return float(np.percentile(ppl, 2.5)), float(np.percentile(ppl, 97.5))


def load_lm(path):
    if PRISM:
        sys.path.insert(0, os.path.join(path, "runtime"))
        from vision_artifact import load_vl_model
        model, _, _ = load_vl_model(path, load_processor=False)
        return model.language_model
    from pathlib import Path
    from mlx_vlm.utils import load_model
    return load_model(Path(path), strict=True).language_model


def main():
    name = os.path.basename(MODEL.rstrip("/"))
    if PRISM:
        name = name + "-prism-runtime"
    ids_all = np.load(CORPUS); n_win = len(ids_all) // SEQ
    print(f"[ppl] {name}: {n_win} windows x {SEQ} tokens = {n_win*SEQ:,}", flush=True)
    lm = load_lm(MODEL)
    win_nll, win_tok, t0 = [], [], time.time()
    for w in range(n_win):
        ids = ids_all[w*SEQ:(w+1)*SEQ].tolist()
        out = lm(mx.array([ids]))
        lg = (out.logits if hasattr(out, "logits") else out)[0].astype(mx.float32)
        lg = lg - mx.logsumexp(lg, axis=-1, keepdims=True)
        nll = -lg[mx.arange(len(ids)-1), mx.array(ids[1:])]
        win_nll.append(float(nll.sum().item())); win_tok.append(len(ids)-1)
        mx.clear_cache()
        if (w+1) % 10 == 0 or w == n_win-1:
            done = sum(win_tok); el = time.time()-t0
            print(f"[ppl] {w+1}/{n_win}  ppl {math.exp(sum(win_nll)/done):.4f}  ({done/el:.0f} tok/s, {el:.0f}s, peak {mx.get_peak_memory()/1e9:.0f} GB)", flush=True)
    win_nll, win_tok = np.array(win_nll), np.array(win_tok)
    ppl = float(np.exp(win_nll.sum()/win_tok.sum())); lo, hi = bootstrap_ci(win_nll, win_tok)
    print(f"[ppl] {name}: perplexity {ppl:.4f}  95% CI [{lo:.4f}, {hi:.4f}]  over {int(win_tok.sum()):,} tokens", flush=True)
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    res[name] = {"perplexity": round(ppl, 4), "ci95": [round(lo, 4), round(hi, 4)], "tokens": int(win_tok.sum()),
                 "windows": int(n_win), "seq_len": SEQ, "window_nll": win_nll.tolist(), "window_tok": win_tok.tolist()}
    json.dump(res, open(OUT, "w"), indent=2); print(f"[ppl] saved -> {OUT}")

if __name__ == "__main__":
    main()
