# BPE — Optimal Tokenization for Protein & Genome Language Models

## Hypothesis

Current protein models (ESM-2, ESM-3, ProtGPT2, etc.) use **single-amino-acid tokenization**. That collapses the token distribution to roughly **amino-acid frequency** — not the heavy-tailed Zipf structure that makes LLMs scale.

Natural language scales because token statistics follow Zipf's law (rank-frequency exponent α ≈ 1). Single-AA tokenization cannot produce that tail: there are only ~20 types with relatively flat frequencies.

**Claim:** Domain-adaptive BPE (and spectrally guided merge schedules) can reshape protein/genome token distributions toward language-like Zipf tails **without losing biological motif structure** — unlocking LLM-style scaling when paired with the right architecture.

**Negative control:** Applying GPT-2's English BPE directly to protein strings produces arbitrary byte merges that **do not** align with biological patterns or Zipf optimality.

## What's in this repo

| Component | Purpose |
|-----------|---------|
| `bpe/zipf.py` | Fit Zipf exponents, entropy, Gini — compare tokenizers |
| `bpe/spectral.py` | Rank bigram merges by PMI + spectral gap (co-occurrence operator) |
| `bpe/tokenizers.py` | Single-AA, GPT-2-on-protein, domain BPE |
| `experiments/run_tokenization_trap.py` | Main experiment pipeline |
| `experiments/run_genome_bpe.py` | Same analysis on nucleotide windows |
| `experiments/train_tiny_lm.py` | Tiny GPT to compare bits/token at fixed params |

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Main experiment (synthetic corpus, no download)
python experiments/run_tokenization_trap.py --plot

# With tiny LM comparison (~2 min CPU)
python experiments/run_tokenization_trap.py --plot --train-lm --lm-steps 200

# Real Swiss-Prot sample (network)
python experiments/run_tokenization_trap.py --corpus uniprot --max-sequences 5000 --plot

# Genome BPE extension
python experiments/run_genome_bpe.py
```

Results land in `results/tokenization_trap/`:
- `zipf_comparison.csv` — α, entropy, Gini per tokenizer
- `spectral_merges.csv` — top PMI/spectral bigram merges
- `summary.json` — best tokenizer by distance to α=1
- `zipf_rank_frequency.png` — rank-frequency plots

## Expected findings

1. **Single-AA:** low type count, α far from 1, entropy bounded by log₂(20)
2. **GPT-2 BPE on protein:** irregular merges, poor motif alignment, suboptimal α
3. **Domain BPE:** higher vocab, heavier tail, α closer to 1.0, lower bits/token in tiny LM

## Open problems (from exploration)

These are **not** solved here — tracked for follow-up:

1. **Optimal residuals & attention** — replace vanilla attention with operators that respect co-occurrence / spectral structure at each layer
2. **ESM-2/3 retrofit** — re-tokenize + continued pretrain at scale
3. **Unified genome+protein model** — shared BPE over mixed corpus with phase-aware architecture

## Relation to scaling laws

LLM scaling assumes:
- Heavy-tailed token distribution (Zipf)
- Low effective rank in residual stream after token embedding
- Sufficient context length to exploit multi-token motifs

Single-AA protein LMs violate the first condition by construction. This repo measures that gap and tests whether BPE closes it.

## Citation

If this direction pans out, cite as work-in-progress from the BPE repository.
