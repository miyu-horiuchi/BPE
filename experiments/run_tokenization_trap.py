#!/usr/bin/env python3
"""Main experiment: the tokenization trap for protein / genome LMs.

Compares:
  1. single-amino-acid tokenization (baseline — frequency-dominated)
  2. GPT-2 BPE applied to protein strings (negative control — wrong distribution)
  3. domain-adaptive BPE trained on protein corpus (hypothesis — Zipf-aligned)

Outputs Zipf metrics, spectral merge rankings, and optional tiny-LM loss curves.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from bpe.corpus import corpus_to_corpus_file, load_corpus
from bpe.spectral import rank_bigram_merges, target_vocab_from_spectrum
from bpe.tokenizers import (
    AminoAcidTokenizer,
    DomainBPETrainer,
    GPT2ProteinTokenizer,
    train_all_domain_bpes,
)
from bpe.zipf import compare_zipf_profiles, distance_to_language_zipf


def run(args: argparse.Namespace) -> None:
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    print("Loading corpus...")
    sequences = load_corpus(
        Path(args.data_dir),
        source=args.corpus,
        max_sequences=args.max_sequences,
    )
    print(f"  {len(sequences)} sequences, median length {sorted(map(len, sequences))[len(sequences)//2]}")

    corpus_path = out / "corpus.txt"
    corpus_to_corpus_file(sequences, corpus_path)

    print("Spectral merge analysis...")
    merges = rank_bigram_merges(sequences, top_k=args.top_merges)
    merge_rows = [
        {
            "bigram": m.bigram,
            "score": m.score,
            "spectral_gap_delta": m.spectral_gap_delta,
            "pmi": m.pmi,
        }
        for m in merges
    ]
    pd.DataFrame(merge_rows).to_csv(out / "spectral_merges.csv", index=False)
    suggested_vocab = target_vocab_from_spectrum(sequences)
    print(f"  suggested starting vocab (spectral): {suggested_vocab}")
    print(f"  top merges: {[m.bigram for m in merges[:8]]}")

    vocab_sizes = [int(v) for v in args.vocab_sizes.split(",")]
    if suggested_vocab not in vocab_sizes:
        vocab_sizes = sorted(set(vocab_sizes + [suggested_vocab]))

    print("Training domain BPE tokenizers...")
    tok_dir = out / "tokenizers"
    domain_bpes = train_all_domain_bpes(corpus_path, tok_dir, vocab_sizes)

    tokenizers = [AminoAcidTokenizer(), GPT2ProteinTokenizer()]
    tokenizers.extend(domain_bpes.values())

    print("Computing Zipf profiles...")
    metrics = compare_zipf_profiles(sequences, tokenizers)
    rows = [m.to_dict() for m in metrics]
    for m in metrics:
        dist = distance_to_language_zipf(m)
        print(
            f"  {m.name:20s}  alpha={m.zipf_exponent:.3f}  "
            f"entropy={m.entropy_bits:.2f}b  gini={m.gini:.3f}  "
            f"|alpha-1|={dist:.3f}"
        )
    df = pd.DataFrame(rows)
    df["distance_to_language_zipf"] = df["zipf_exponent"].apply(
        lambda a: abs(a - 1.0) if pd.notna(a) else float("nan")
    )
    df.to_csv(out / "zipf_comparison.csv", index=False)

    summary = {
        "n_sequences": len(sequences),
        "suggested_vocab_spectral": suggested_vocab,
        "best_tokenizer_by_zipf": df.loc[df["distance_to_language_zipf"].idxmin(), "name"],
        "single_aa_alpha": float(df.loc[df["name"] == "single_aa", "zipf_exponent"].iloc[0]),
        "gpt2_alpha": float(df.loc[df["name"] == "gpt2_on_protein", "zipf_exponent"].iloc[0]),
        "top_spectral_merges": [m.bigram for m in merges[:16]],
    }
    best_domain = df[df["name"].str.startswith("domain_bpe")].sort_values("distance_to_language_zipf")
    if len(best_domain):
        summary["best_domain_bpe"] = best_domain.iloc[0]["name"]
        summary["best_domain_alpha"] = float(best_domain.iloc[0]["zipf_exponent"])
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    if args.plot:
        from experiments.plot_results import plot_zipf_comparison

        plot_zipf_comparison(sequences, tokenizers, out / "zipf_rank_frequency.png")
        print(f"  wrote {out / 'zipf_rank_frequency.png'}")

    if args.train_lm:
        from experiments.train_tiny_lm import run_scaling_comparison

        print("Running tiny LM scaling comparison (CPU-friendly)...")
        lm_results = run_scaling_comparison(
            sequences,
            tokenizers=[AminoAcidTokenizer(), domain_bpes[vocab_sizes[-1]]],
            out_dir=out / "tiny_lm",
            steps=args.lm_steps,
        )
        (out / "tiny_lm_results.json").write_text(json.dumps(lm_results, indent=2))

    print(f"\nDone. Results in {out}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--output", default="results/tokenization_trap")
    p.add_argument("--corpus", choices=["synthetic", "bundled", "uniprot"], default="synthetic")
    p.add_argument("--max-sequences", type=int, default=2000)
    p.add_argument("--vocab-sizes", default="128,256,512,1024")
    p.add_argument("--top-merges", type=int, default=64)
    p.add_argument("--plot", action="store_true")
    p.add_argument("--train-lm", action="store_true")
    p.add_argument("--lm-steps", type=int, default=300)
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
