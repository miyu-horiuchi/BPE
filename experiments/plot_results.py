"""Plot Zipf rank-frequency curves for tokenizer comparison."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np

from bpe.zipf import TokenCounter, token_frequencies


def plot_zipf_comparison(
    sequences: Iterable[str],
    tokenizers: list[TokenCounter],
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for tok in tokenizers:
        freqs = token_frequencies(sequences, tok)
        ranks = np.arange(1, len(freqs) + 1)
        axes[0].loglog(ranks, freqs, label=tok.name, alpha=0.85)
        # reference Zipf with alpha=1
    ref_r = np.logspace(0, np.log10(max(len(tokenizers[0].tokenize(next(iter(sequences)))), 2)), 50)
    axes[0].loglog(ref_r, ref_r ** -1.0 / ref_r.sum() * 0.1, "k--", alpha=0.4, label="alpha=1 reference")

    alphas = []
    names = []
    for tok in tokenizers:
        from bpe.zipf import compute_distribution_metrics

        m = compute_distribution_metrics(sequences, tok)
        alphas.append(m.p_median)
        names.append(tok.name)

    axes[1].barh(names, alphas, color=["#4C72B0", "#DD8452", "#55A868", "#C44E52"][: len(names)])
    axes[1].axvline(1.0, color="black", linestyle="--", alpha=0.5, label="language Zipf (alpha=1)")
    axes[1].set_xlabel("Zipf exponent (fitted)")
    axes[1].set_title("Distance from LLM-scaling token statistics")
    axes[1].legend()

    axes[0].set_xlabel("Token rank")
    axes[0].set_ylabel("Relative frequency")
    axes[0].set_title("Rank-frequency tails by tokenizer")
    axes[0].legend(fontsize=8)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
