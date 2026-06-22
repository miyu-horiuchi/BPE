#!/usr/bin/env python3
"""Diagnose why GPT-2-on-protein p_median jumped (0.88 -> 2.01).

Compares the OLD metric (single global log-log fit) with the NEW metric
(bootstrap-median with r2 filter) on the exact same token distribution, and
dumps the distribution shape so we can see *why* they disagree.
"""
from __future__ import annotations

from collections import Counter

import numpy as np

from bpe.corpus import load_corpus
from bpe.tokenizers import AminoAcidTokenizer, GPT2ProteinTokenizer, DomainBPETrainer
from bpe.zipf import (
    corpus_token_stream,
    fit_zipf_exponent,
    bootstrap_median_zipf,
    _ranked_freqs,
)
from pathlib import Path


def diagnose(name, tokens):
    counts = Counter(tokens)
    n_types = len(counts)
    n_tokens = len(tokens)
    freqs = _ranked_freqs(dict(counts))

    # OLD metric: single global fit over ALL ranks
    alpha_global, r2_global = fit_zipf_exponent(freqs)

    # NEW metric: bootstrap median with r2>=0.35 filter, plus how many pass
    rng = np.random.default_rng(0)
    _, inv = np.unique(np.asarray(tokens, dtype=object), return_inverse=True)
    base = np.bincount(inv)
    p = base / base.sum()
    alphas, r2s, passed = [], [], 0
    for _ in range(40):
        s = rng.multinomial(n_tokens, p).astype(float)
        s = s[s > 0]
        if len(s) < 3:
            continue
        s.sort()
        f = s[::-1] / s.sum()
        a, r2 = fit_zipf_exponent(f)
        if np.isfinite(a):
            r2s.append(r2)
            if r2 >= 0.35:
                alphas.append(a)
                passed += 1
    p_median = float(np.median(alphas)) if alphas else float("nan")

    # fit restricted to the head (top decile) vs tail to show curvature
    head = freqs[: max(3, n_types // 10)]
    tail = freqs[n_types // 2 :]
    a_head, _ = fit_zipf_exponent(head)
    a_tail, _ = fit_zipf_exponent(tail if len(tail) >= 3 else freqs)

    print(f"\n=== {name} ===")
    print(f"  n_tokens={n_tokens:,}  n_types={n_types:,}")
    print(f"  top-5 token shares: {[round(float(x),4) for x in freqs[:5]]}")
    print(f"  share of top type: {freqs[0]:.4f}   top-10 cumulative: {freqs[:10].sum():.4f}")
    print(f"  singletons (freq==1): {sum(1 for c in counts.values() if c==1)} "
          f"({100*sum(1 for c in counts.values() if c==1)/n_types:.1f}% of types)")
    print(f"  OLD metric  global fit  alpha={alpha_global:.3f}  r2={r2_global:.3f}")
    print(f"  NEW metric  p_median   ={p_median:.3f}  (boot samples passing r2>=0.35: {passed}/40, "
          f"mean r2={np.mean(r2s):.3f})")
    print(f"  head-only alpha={a_head:.3f}   tail-only alpha={a_tail:.3f}  "
          f"(gap => log-log curvature, not a clean power law)")


def main():
    seqs = load_corpus(Path("data"), source="synthetic", max_sequences=5000)
    print(f"corpus: {len(seqs)} synthetic protein sequences")

    toks = {
        "single_aa": AminoAcidTokenizer(),
        "gpt2_on_protein": GPT2ProteinTokenizer(),
    }
    # one domain BPE for contrast
    from bpe.corpus import corpus_to_corpus_file
    cp = Path("results/_diag_corpus.txt")
    cp.parent.mkdir(parents=True, exist_ok=True)
    corpus_to_corpus_file(seqs, cp)
    toks["domain_bpe_1000"] = DomainBPETrainer(vocab_size=1000).train(cp, "domain_bpe_1000")

    for name, tk in toks.items():
        diagnose(name, corpus_token_stream(seqs, tk))


if __name__ == "__main__":
    main()
