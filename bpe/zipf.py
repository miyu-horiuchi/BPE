"""Zipf distribution analysis — the scaling-law prerequisite for LMs."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable, Protocol

import numpy as np
from scipy import stats


class TokenCounter(Protocol):
    name: str

    def tokenize(self, sequence: str) -> list[str]: ...


@dataclass
class ZipfMetrics:
    name: str
    zipf_exponent: float  # slope of log(rank) vs log(freq); ~1.0 for natural language
    r_squared: float
    entropy_bits: float
    n_tokens: int
    n_types: int
    type_token_ratio: float
    gini: float  # inequality of token frequencies; higher = more Zipf-like tail
    top10_mass: float  # fraction of tokens in top-10 types

    def to_dict(self) -> dict:
        return asdict(self)


def token_frequencies(sequences: Iterable[str], tokenizer: TokenCounter) -> np.ndarray:
    from collections import Counter

    counts: Counter[str] = Counter()
    for seq in sequences:
        counts.update(tokenizer.tokenize(seq))
    freqs = np.array([c for _, c in counts.most_common()], dtype=float)
    if freqs.sum() == 0:
        return freqs
    return freqs / freqs.sum()


def fit_zipf_exponent(frequencies: np.ndarray) -> tuple[float, float]:
    """MLE-style Zipf fit on ranked frequencies (freq ~ rank^{-alpha})."""
    if len(frequencies) < 3:
        return float("nan"), float("nan")
    ranks = np.arange(1, len(frequencies) + 1, dtype=float)
    mask = frequencies > 0
    log_r = np.log(ranks[mask])
    log_f = np.log(frequencies[mask])
    slope, intercept, r_value, _, _ = stats.linregress(log_r, log_f)
    # freq = C * rank^{-alpha}  =>  log(freq) = log(C) - alpha * log(rank)
    alpha = -slope
    return float(alpha), float(r_value ** 2)


def entropy_bits(frequencies: np.ndarray) -> float:
    p = frequencies[frequencies > 0]
    return float(-np.sum(p * np.log2(p)))


def gini_coefficient(frequencies: np.ndarray) -> float:
    x = np.sort(frequencies)
    n = len(x)
    if n == 0:
        return float("nan")
    cum = np.cumsum(x)
    return float((n + 1 - 2 * np.sum(cum) / cum[-1]) / n)


def compute_zipf_metrics(
    sequences: Iterable[str],
    tokenizer: TokenCounter,
) -> ZipfMetrics:
    freqs = token_frequencies(sequences, tokenizer)
    alpha, r2 = fit_zipf_exponent(freqs)
    n_tokens = sum(len(tokenizer.tokenize(s)) for s in sequences)
    n_types = len(freqs)
    ttr = n_types / max(n_tokens, 1)
    top10 = float(freqs[:10].sum()) if len(freqs) >= 10 else float(freqs.sum())
    return ZipfMetrics(
        name=tokenizer.name,
        zipf_exponent=alpha,
        r_squared=r2,
        entropy_bits=entropy_bits(freqs),
        n_tokens=n_tokens,
        n_types=n_types,
        type_token_ratio=ttr,
        gini=gini_coefficient(freqs),
        top10_mass=top10,
    )


def compare_zipf_profiles(
    sequences: Iterable[str],
    tokenizers: list[TokenCounter],
) -> list[ZipfMetrics]:
    return [compute_zipf_metrics(sequences, tok) for tok in tokenizers]


def distance_to_language_zipf(metrics: ZipfMetrics, target_alpha: float = 1.0) -> float:
    """How far a tokenizer's rank-frequency tail is from natural-language Zipf."""
    return abs(metrics.zipf_exponent - target_alpha)
