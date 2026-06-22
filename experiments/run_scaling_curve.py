#!/usr/bin/env python3
"""Scaling *curve*: bits-per-residue vs. model size for each tokenizer.

The single-point scaling proof shows domain-BPE wins at one budget. The real
claim of the tokenization trap is about *scaling*: a language-like token
distribution should give a better loss-vs-size curve (lower offset and/or
steeper slope). This script sweeps several TinyGPT sizes per tokenizer and fits
a power law bpr = a * params^(-b) over the non-embedding parameter count, then
plots everything on log-log axes.

CPU-friendly: small models, short training, residue-capped corpus. Use
--corpus uniprot to run on the cached Swiss-Prot FASTA.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from bpe.corpus import corpus_to_corpus_file, load_corpus
from bpe.tokenizers import (
    AminoAcidTokenizer,
    GPT2ProteinTokenizer,
    train_all_domain_bpes,
)
from experiments.train_tiny_lm import LMConfig
from experiments.run_scaling_proof import train_and_eval


# (label, d_model, n_heads, n_layers) — increasing capacity, CPU-tractable.
MODEL_SIZES = [
    ("xs", 48, 2, 1),
    ("s", 80, 4, 2),
    ("m", 128, 4, 3),
    ("l", 192, 6, 4),
]


def fit_power_law(params: list[int], bpr: list[float]) -> tuple[float, float, float]:
    """Fit log(bpr) = log(a) - b*log(params). Returns (a, b, r2)."""
    x = np.log(np.asarray(params, dtype=float))
    y = np.log(np.asarray(bpr, dtype=float))
    if len(x) < 2:
        return float("nan"), float("nan"), float("nan")
    b_neg, log_a = np.polyfit(x, y, 1)
    a = math.exp(log_a)
    b = -b_neg
    yhat = log_a + b_neg * x
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2)) or 1e-12
    return a, b, 1.0 - ss_res / ss_tot


def write_outputs(curves: dict[str, list[dict]], fits: dict[str, dict], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "scaling_curve.json").write_text(
        json.dumps({"curves": curves, "fits": fits}, indent=2)
    )

    lines = [
        "# Scaling curve — bits/residue vs. model size\n",
        "Each tokenizer trained at several TinyGPT sizes; fit bpr = a * N^(-b) "
        "over non-embedding params N. Lower bpr and larger b are better.\n",
        "| Tokenizer | size | non-emb params | bits/residue |",
        "|-----------|------|---------------:|-------------:|",
    ]
    for name, rows in curves.items():
        for r in rows:
            lines.append(
                f"| {name} | {r['size']} | {r['non_embedding_params']:,} | "
                f"{r['bits_per_residue']:.3f} |"
            )
    lines += [
        "\n## Power-law fits (bpr = a · N^(-b))\n",
        "| Tokenizer | a | b (slope) | fit r² | bpr @ largest |",
        "|-----------|--:|----------:|-------:|--------------:|",
    ]
    for name, f in fits.items():
        lines.append(
            f"| {name} | {f['a']:.3f} | {f['b']:.4f} | {f['r2']:.3f} | "
            f"{f['bpr_largest']:.3f} |"
        )
    (out / "scaling_curve.md").write_text("\n".join(lines) + "\n")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 6))
        palette = {
            "single_aa": "#C44E52",
            "gpt2_on_protein": "#DD8452",
        }
        default_colors = ["#4C72B0", "#55A868", "#8172B3", "#937860"]
        ci = 0
        for name, rows in curves.items():
            N = [r["non_embedding_params"] for r in rows]
            y = [r["bits_per_residue"] for r in rows]
            if name in palette:
                color = palette[name]
            else:
                color = default_colors[ci % len(default_colors)]
                ci += 1
            ax.plot(N, y, "o-", color=color, label=f"{name} (b={fits[name]['b']:.2f})",
                    alpha=0.9, markersize=6)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("non-embedding parameters (log)")
        ax.set_ylabel("bits per residue (log, lower = better)")
        ax.set_title("Scaling curve by tokenizer")
        ax.grid(True, which="both", ls=":", alpha=0.4)
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(out / "scaling_curve.png", dpi=150)
        plt.close(fig)
    except Exception as exc:  # plotting best-effort
        print(f"  (plot skipped: {exc})")


def run(args: argparse.Namespace) -> None:
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    raw = load_corpus(Path(args.data_dir), source=args.corpus, max_sequences=args.max_seqs)
    seqs = [s[: args.residue_cap] for s in raw if len(s) >= 20]
    split = int(len(seqs) * 0.9)
    train_seqs, eval_seqs = seqs[:split], seqs[split:]
    print(f"Corpus: {len(seqs)} seqs (cap {args.residue_cap}) -> "
          f"{len(train_seqs)} train / {len(eval_seqs)} eval")

    corpus_path = out / "corpus.txt"
    corpus_to_corpus_file(train_seqs, corpus_path)

    bpe_vocabs = [int(v) for v in args.bpe_vocabs.split(",")]
    domain = train_all_domain_bpes(corpus_path, out / "tokenizers", bpe_vocabs)

    tokenizers = [(AminoAcidTokenizer(), "single_aa")]
    for vs in bpe_vocabs:
        tokenizers.append((domain[vs], f"domain_bpe_{vs}"))
    if not args.no_gpt2:
        try:
            tokenizers.append((GPT2ProteinTokenizer(), "gpt2_on_protein"))
        except Exception as exc:
            print(f"  (skipping GPT-2: {exc})")

    curves: dict[str, list[dict]] = {}
    fits: dict[str, dict] = {}
    for tok, name in tokenizers:
        print(f"\nTokenizer {name} (vocab={tok.vocab_size})")
        rows = []
        for label, d_model, n_heads, n_layers in MODEL_SIZES:
            cfg = LMConfig(d_model=d_model, n_heads=n_heads, n_layers=n_layers,
                           max_len=args.max_len, dropout=0.1)
            res = train_and_eval(
                train_seqs, eval_seqs, tok, name, cfg,
                steps=args.steps, batch_size=args.batch_size, lr=args.lr, seed=args.seed,
            )
            if "error" in res:
                print(f"  size {label}: ERROR {res['error']}")
                continue
            res["size"] = label
            rows.append(res)
            print(f"  size {label}: non-emb params={res['non_embedding_params']:,}  "
                  f"bpr={res['bits_per_residue']:.3f}")
        if not rows:
            continue
        curves[name] = rows
        a, b, r2 = fit_power_law(
            [r["non_embedding_params"] for r in rows],
            [r["bits_per_residue"] for r in rows],
        )
        fits[name] = {
            "a": round(a, 4), "b": round(b, 4), "r2": round(r2, 4),
            "bpr_largest": rows[-1]["bits_per_residue"],
        }
        print(f"  fit: bpr = {a:.3f} * N^(-{b:.4f})  (r2={r2:.3f})")

    write_outputs(curves, fits, out)
    print(f"\nWrote {out/'scaling_curve.md'}, scaling_curve.json, scaling_curve.png")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--output", default="results/scaling_curve")
    p.add_argument("--corpus", choices=["synthetic", "bundled", "uniprot"], default="uniprot")
    p.add_argument("--max-seqs", type=int, default=1500)
    p.add_argument("--residue-cap", type=int, default=160)
    p.add_argument("--bpe-vocabs", default="256,1000")
    p.add_argument("--no-gpt2", action="store_true")
    p.add_argument("--max-len", type=int, default=256)
    p.add_argument("--steps", type=int, default=150)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=0)
    run(p.parse_args())


if __name__ == "__main__":
    main()
