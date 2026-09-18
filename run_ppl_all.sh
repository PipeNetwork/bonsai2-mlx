#!/bin/bash
V=/Users/david/llm/bonsai2-mlx/.venv/bin/python
S=/Users/david/llm/bonsai2-mlx/scripts/ppl_large.py
C=/Users/david/llm/bonsai2-mlx/ppl_corpus.npy
R=/Users/david/llm/bonsai2-mlx/ppl_results.json
$V $S /Users/david/llm/bonsai2-out/bf16 $C 2048 $R
$V $S --prism-pack /Users/david/llm/Bonsai2-mlx2bit-src $C 2048 $R
$V $S /Users/david/llm/bonsai2-out/8bit $C 2048 $R
$V $S /Users/david/llm/bonsai2-out/6bit $C 2048 $R
$V $S /Users/david/llm/bonsai2-out/4bit $C 2048 $R
echo PPL ALL DONE
