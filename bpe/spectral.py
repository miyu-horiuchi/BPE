"""Spectral analysis for domain-adaptive BPE merge prioritization.

Hypothesis: optimal merges are those that concentrate variance in the leading
eigenmodes of the sequence co-occurrence operator — mirroring how LLM scaling
laws assume heavy-tailed token statistics with low effective rank in residual
structure.

This module scores candidate bigram merges using a PMI-weighted co-occurrence
matrix and prioritizes merges that maximally increase spectral gap / reduce
effective rank of the residual after removing the merged pair's contribution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass
class SpectralMergeScore:
    bigram: str
    score: float
    spectral_gap_delta: float
    pmi: float


def _bigram_counts(sequences: Iterable[str]) -> dict[str, int]:
    from collections import Counter

    counts: Counter[str] = Counter()
    unigram: Counter[str] = Counter()
    for seq in sequences:
        seq = seq.upper()
        for c in seq:
            unigram[c] += 1
        for i in range(len(seq) - 1):
            counts[seq[i : i + 2]] += 1
    counts["_unigram_"] = dict(unigram)  # type: ignore[assignment]
    return dict(counts)


def pmi_matrix(sequences: Iterable[str], alphabet: str = "ACDEFGHIKLMNPQRSTVWY") -> tuple[np.ndarray, list[str]]:
    counts = _bigram_counts(sequences)
    unigram = counts.pop("_unigram_", {})
    total_bigrams = sum(v for k, v in counts.items() if len(k) == 2)
    chars = list(alphabet)
    idx = {c: i for i, c in enumerate(chars)}
    n = len(chars)
    co = np.zeros((n, n), dtype=float)
    for bg, c in counts.items():
        if len(bg) != 2:
            continue
        i, j = idx.get(bg[0]), idx.get(bg[1])
        if i is None or j is None:
            continue
        co[i, j] = c / max(total_bigrams, 1)
    # symmetrize for spectral analysis
    sym = 0.5 * (co + co.T)
    return sym, chars


def effective_rank(matrix: np.ndarray, eps: float = 1e-12) -> float:
    s = np.linalg.svd(matrix, compute_uv=False)
    s = s[s > eps]
    if len(s) == 0:
        return 0.0
    p = s / s.sum()
    return float(np.exp(-np.sum(p * np.log(p + eps))))


def spectral_gap(matrix: np.ndarray) -> float:
    s = np.linalg.svd(matrix, compute_uv=False)
    if len(s) < 2:
        return 0.0
    return float(s[0] - s[1])


def rank_bigram_merges(
    sequences: Iterable[str],
    top_k: int = 64,
) -> list[SpectralMergeScore]:
    """Rank bigram merges by PMI plus predicted spectral-gap improvement."""
    counts = _bigram_counts(sequences)
    unigram = counts.pop("_unigram_", {})
    total = sum(counts.values())
    base, chars = pmi_matrix(sequences)
    base_gap = spectral_gap(base)
    base_rank = effective_rank(base)

    idx = {c: i for i, c in enumerate(chars)}
    scored: list[SpectralMergeScore] = []
    for bg, c in counts.items():
        if len(bg) != 2 or c < 2:
            continue
        i, j = idx.get(bg[0]), idx.get(bg[1])
        if i is None or j is None:
            continue
        p_xy = c / max(total, 1)
        p_x = unigram.get(bg[0], 0) / max(sum(unigram.values()), 1)
        p_y = unigram.get(bg[1], 0) / max(sum(unigram.values()), 1)
        pmi = float(np.log((p_xy + 1e-12) / (p_x * p_y + 1e-12)))

        # approximate residual after down-weighting this bigram's co-occurrence
        residual = base.copy()
        residual[i, j] *= 0.1
        residual[j, i] *= 0.1
        gap_delta = spectral_gap(residual) - base_gap
        rank_delta = base_rank - effective_rank(residual)
        score = pmi + 2.0 * gap_delta + 0.5 * rank_delta
        scored.append(
            SpectralMergeScore(
                bigram=bg,
                score=float(score),
                spectral_gap_delta=float(gap_delta),
                pmi=pmi,
            )
        )
    scored.sort(key=lambda x: x.score, reverse=True)
    return scored[:top_k]


def target_vocab_from_spectrum(
    sequences: Iterable[str],
    alpha_min: float = 0.85,
    alpha_max: float = 1.15,
    vocab_candidates: Iterable[int] = (64, 128, 256, 512, 1024, 2048),
) -> int:
    """Heuristic: pick smallest vocab whose spectral effective rank matches Zipf target band.

    Used as a starting point before full BPE training + Zipf validation loop.
    """
    mat, _ = pmi_matrix(sequences)
    er = effective_rank(mat)
    # map effective rank (~1-20 for AA co-occurrence) to vocab size
    scale = er / 5.0
    for v in sorted(vocab_candidates):
        if v >= 32 * scale:
            return v
    return max(vocab_candidates)
