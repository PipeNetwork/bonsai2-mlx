"""Text smoke test through STOCK mlx_vlm.load + generate.

    .venv/bin/python scripts/smoke_generate.py <MODEL_DIR> [--max-tokens N] [--prompt STR]
"""

import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument(
        "--prompt", default="Explain in two sentences why the sky is blue."
    )
    args = ap.parse_args()

    from mlx_vlm import generate, load
    from mlx_vlm.prompt_utils import apply_chat_template

    model, processor = load(args.model_dir, strict=True)
    config = model.config
    formatted = apply_chat_template(processor, config, args.prompt, num_images=0)
    out = generate(
        model,
        processor,
        formatted,
        max_tokens=args.max_tokens,
        temperature=0.0,
        verbose=True,
    )
    text = out.text if hasattr(out, "text") else out
    print("\n--- output ---\n", text)


if __name__ == "__main__":
    main()
