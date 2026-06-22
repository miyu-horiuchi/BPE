"""Render scaling-law tokenizer tables (markdown + CSV)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from bpe.zipf import DistributionMetrics


def _fmt_p(value: float | None, bold: bool = False) -> str:
    if value is None:
        return "--"
    text = f"{value:.2f}"
    return f"**{text}**" if bold else text


def protein_table_markdown(rows: list[DistributionMetrics]) -> str:
    lines = [
        "| Tokenizer | Vocab | p_median |",
        "|-----------|------:|---------:|",
    ]
    for r in rows:
        lines.append(
            f"| {r.tokenizer} | {r.vocab} | {_fmt_p(r.p_median, r.bold_median)} |"
        )
    return "\n".join(lines) + "\n"


def genome_table_markdown(rows: list[DistributionMetrics]) -> str:
    lines = [
        "| Tokenizer | Vocab | p_median | p_zipf | p_comp | Entropy% |",
        "|-----------|------:|---------:|-------:|-------:|---------:|",
    ]
    for r in rows:
        p_zipf_str = "--" if r.p_zipf is None else _fmt_p(r.p_zipf, False)
        lines.append(
            f"| {r.tokenizer} | {r.vocab} | {_fmt_p(r.p_median, r.bold_median)} | "
            f"{p_zipf_str} | {r.p_comp:.2f} | {r.entropy_pct:.1f}% |"
        )
    return "\n".join(lines) + "\n"


def write_protein_table(rows: list[DistributionMetrics], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    md = protein_table_markdown(rows)
    path = out_dir / "protein_tokenizer_table.md"
    path.write_text(md)
    pd.DataFrame([r.to_dict() for r in rows]).to_csv(out_dir / "protein_tokenizer_table.csv", index=False)
    return path


def write_genome_table(rows: list[DistributionMetrics], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    md = genome_table_markdown(rows)
    path = out_dir / "genome_tokenizer_table.md"
    path.write_text(md)
    pd.DataFrame([r.to_dict() for r in rows]).to_csv(out_dir / "genome_tokenizer_table.csv", index=False)
    return path


def print_table(rows: list[DistributionMetrics], *, genome: bool = False) -> None:
    text = genome_table_markdown(rows) if genome else protein_table_markdown(rows)
    print(text)
