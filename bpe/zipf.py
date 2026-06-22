"""Zipf / composition distribution metrics for tokenizer scaling analysis."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable, Protocol

import numpy as np
from scipy import stats


class TokenCounter(Protocol):
    name: str

    def tokenize(self, sequence: str) -> list[str]: ...

    @property
    def vocab_size(self) -> int: ...


@dataclass
class DistributionMetrics:
    """Metrics matching the scaling-law tokenizer tables."""

    tokenizer: str
    vocab: int
    p_median: float
    p_zipf: float | None
    p_comp: float
    entropy_pct: float
    n_tokens: int = 0
    n_types: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def bold_median(self) -> bool:
        if self.tokenizer.lower().startswith("single"):
            return False
        return self.p_median >= 1.0


def _ranked_freqs(counts: dict[str, int]) -> np.ndarray:
    freqs = np.array([c for _, c in sorted(counts.items(), key=lambda x: -x[1])], dtype=float)
    if freqs.sum() == 0:
        return freqs
    return freqs / freqs.sum()


def fit_zipf_exponent(frequencies: np.ndarray) -> tuple[float, float]:
    """Power-law fit on ranked frequencies: freq(r) ~ r^{-alpha}."""
    if len(frequencies) < 3:
        return float("nan"), float("nan")
    ranks = np.arange(1, len(frequencies) + 1, dtype=float)
    mask = frequencies > 0
    log_r = np.log(ranks[mask])
    log_f = np.log(frequencies[mask])
    slope, _, r_value, _, _ = stats.linregress(log_r, log_f)
    return float(-slope), float(r_value**2)


def entropy_bits(frequencies: np.ndarray) -> float:
    p = frequencies[frequencies > 0]
    return float(-np.sum(p * np.log2(p)))


def corpus_token_stream(sequences: Iterable[str], tokenizer: TokenCounter) -> list[str]:
    tokens: list[str] = []
    for seq in sequences:
        tokens.extend(tokenizer.tokenize(seq))
    return tokens


def bootstrap_median_zipf(
    tokens: list[str],
    n_boot: int = 40,
    seed: int = 0,
    min_r2: float = 0.35,
) -> float:
    """p_median: bootstrap median of Zipf exponents on resampled token streams."""
    n = len(tokens)
    if n < 32:
        return float("nan")
    rng = np.random.default_rng(seed)
    # integer-code tokens once; bootstrap = multinomial resample of type counts
    _, inv = np.unique(np.asarray(tokens, dtype=object), return_inverse=True)
    base_counts = np.bincount(inv)
    n_types = len(base_counts)
    if n_types < 3:
        return float("nan")
    p = base_counts / base_counts.sum()
    exponents: list[float] = []
    for _ in range(n_boot):
        sampled = rng.multinomial(n, p).astype(float)
        sampled = sampled[sampled > 0]
        if len(sampled) < 3:
            continue
        sampled.sort()
        freqs = sampled[::-1] / sampled.sum()
        alpha, r2 = fit_zipf_exponent(freqs)
        if np.isfinite(alpha) and r2 >= min_r2:
            exponents.append(alpha)
    if not exponents:
        return float("nan")
    return float(np.median(exponents))


def composition_spectrum_exponent(
    sequences: Iterable[str],
    tokenizer: TokenCounter,
    max_types: int = 512,
) -> float:
    """p_comp: Zipf exponent on the token co-occurrence operator spectrum.

    Flat spectrum (single-nt / uniform AA) -> exponent ~ 0.
    Motif-structured BPE -> heavier tail, exponent ~ 1.
    """
    from collections import Counter

    seq_list = list(sequences)
    if not seq_list:
        return float("nan")

    type_counts: Counter[str] = Counter()
    for seq in seq_list:
        type_counts.update(tokenizer.tokenize(seq))
    top_types = [t for t, _ in type_counts.most_common(max_types)]
    if len(top_types) < 2:
        return 0.0
    idx = {t: i for i, t in enumerate(top_types)}
    n = len(top_types)
    co = np.zeros((n, n), dtype=float)
    for seq in seq_list:
        toks = [t for t in tokenizer.tokenize(seq) if t in idx]
        for a, b in zip(toks, toks[1:]):
            co[idx[a], idx[b]] += 1.0
    if co.sum() == 0:
        return 0.0
    co /= co.sum()
    sym = 0.5 * (co + co.T)
    # Remove unigram independence baseline -> flat when composition is trivial
    uni = np.array([type_counts[t] for t in top_types], dtype=float)
    uni /= uni.sum()
    indep = np.outer(uni, uni)
    residual = sym - indep
    residual = np.clip(residual, 0, None)
    if residual.sum() < 1e-9:
        return 0.0
    singular = np.linalg.svd(residual, compute_uv=False)
    singular = singular[singular > 1e-12]
    if len(singular) < 3:
        return 0.0
    singular /= singular.sum()
    alpha, r2 = fit_zipf_exponent(singular)
    if not np.isfinite(alpha) or r2 < 0.3:
        return 0.0
    return float(alpha)


def compute_distribution_metrics(
    sequences: Iterable[str],
    tokenizer: TokenCounter,
    *,
    display_name: str | None = None,
    vocab_override: int | None = None,
) -> DistributionMetrics:
    seq_list = list(sequences)
    tokens = corpus_token_stream(seq_list, tokenizer)
    from collections import Counter

    type_counts = Counter(tokens)
    freqs = _ranked_freqs(dict(type_counts))

    p_zipf, _ = fit_zipf_exponent(freqs) if len(freqs) >= 3 else (float("nan"), 0.0)

    # p_median: bootstrap; for tiny vocabs fall back to global fit
    if len(type_counts) <= 20:
        p_median = float(p_zipf) if np.isfinite(p_zipf) else float("nan")
    else:
        p_median = bootstrap_median_zipf(tokens)
        if not np.isfinite(p_median):
            p_median = float(p_zipf)

    p_comp = composition_spectrum_exponent(seq_list, tokenizer)

    vocab = vocab_override if vocab_override is not None else tokenizer.vocab_size
    h = entropy_bits(freqs)
    h_max = np.log2(max(vocab, 2))
    entropy_pct = float(100.0 * h / h_max) if h_max > 0 else float("nan")

    # No meaningful global Zipf fit for 4-letter genome baseline
    p_zipf_out: float | None
    if vocab <= 4:
        p_zipf_out = None
    else:
        p_zipf_out = float(p_zipf) if np.isfinite(p_zipf) else None

    return DistributionMetrics(
        tokenizer=display_name or tokenizer.name,
        vocab=vocab,
        p_median=p_median,
        p_zipf=p_zipf_out,
        p_comp=p_comp,
        entropy_pct=entropy_pct,
        n_tokens=len(tokens),
        n_types=len(type_counts),
    )


def compare_distributions(
    sequences: Iterable[str],
    tokenizers: list[tuple[TokenCounter, str, int | None]],
) -> list[DistributionMetrics]:
    """Each entry: (tokenizer, display_name, vocab_override)."""
    return [
        compute_distribution_metrics(
            sequences, tok, display_name=name, vocab_override=vocab
        )
        for tok, name, vocab in tokenizers
    ]

# --- legacy aliases used elsewhere in the repo ---
ZipfMetrics = DistributionMetrics


def compare_zipf_profiles(sequences, tokenizers):
    return [
        compute_distribution_metrics(sequences, tok, display_name=tok.name)
        for tok in tokenizers
    ]


def token_frequencies(sequences, tokenizer):
    from collections import Counter

    counts: Counter[str] = Counter()
    for seq in sequences:
        counts.update(tokenizer.tokenize(seq))
    freqs = np.array([c for _, c in counts.most_common()], dtype=float)
    if freqs.sum() == 0:
        return freqs
    return freqs / freqs.sum()


def distance_to_language_zipf(metrics: DistributionMetrics, target: float = 1.0) -> float:
    return abs(metrics.p_median - target)
