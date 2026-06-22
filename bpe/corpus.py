"""Load protein / genome sequence corpora for tokenizer experiments."""

from __future__ import annotations

import gzip
import re
import urllib.request
from pathlib import Path

AA = set("ACDEFGHIKLMNPQRSTVWY")
DNA = set("ACGTN")

SWISSPROT_SAMPLE_URL = (
    "https://rest.uniprot.org/uniprotkb/stream?"
    "format=fasta&query=reviewed:true+AND+length:[100+TO+500]"
    "&size=5000"
)


def _clean_sequence(seq: str, alphabet: set[str]) -> str:
    seq = seq.upper().strip()
    return "".join(c for c in seq if c in alphabet)


def read_fasta(path: Path, alphabet: set[str] = AA) -> list[str]:
    sequences: list[str] = []
    chunks: list[str] = []
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            if chunks:
                seq = _clean_sequence("".join(chunks), alphabet)
                if len(seq) >= 20:
                    sequences.append(seq)
            chunks = []
        else:
            chunks.append(line.strip())
    if chunks:
        seq = _clean_sequence("".join(chunks), alphabet)
        if len(seq) >= 20:
            sequences.append(seq)
    return sequences


def read_text_corpus(path: Path, alphabet: set[str] = AA) -> list[str]:
    sequences: list[str] = []
    for line in path.read_text().splitlines():
        seq = _clean_sequence(line, alphabet)
        if len(seq) >= 20:
            sequences.append(seq)
    return sequences


def download_swissprot_sample(out_path: Path, n: int = 5000) -> list[str]:
    """Fetch a Swiss-Prot FASTA slice (requires network)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    url = SWISSPROT_SAMPLE_URL.replace("size=5000", f"size={n}")
    with urllib.request.urlopen(url, timeout=120) as resp:
        raw = resp.read()
    out_path.write_bytes(raw)
    return read_fasta(out_path)


def synthetic_motif_corpus(n: int = 2000, seed: int = 0) -> list[str]:
    """Generate sequences with embedded motifs (helix repeats, zinc fingers, signal peptides).

    Useful offline when UniProt is unavailable. Motifs create non-uniform bigram structure
    that single-AA tokenization cannot expose as token-level Zipf structure.
    """
    import random

    rng = random.Random(seed)
    motifs = [
        "GPGPGPGP",  # collagen-like
        "EEEEEE",  # acidic patch
        "KKKKKK",  # basic patch
        "CCHHCCHH",  # zinc-finger-ish
        "MKTLLLTLVV",  # signal peptide start
        "DEAD",  # DEAD box
        "GGGGG",  # glycine-rich loop
        "PDGEA",  # common beta-turn fragment
    ]
    aa = list(AA)
    sequences: list[str] = []
    for _ in range(n):
        length = rng.randint(80, 400)
        parts: list[str] = []
        while sum(len(p) for p in parts) < length:
            if rng.random() < 0.35:
                parts.append(rng.choice(motifs))
            else:
                run = rng.randint(3, 12)
                parts.append("".join(rng.choices(aa, k=run)))
        seq = "".join(parts)[:length]
        sequences.append(seq)
    return sequences


def load_corpus(
    data_dir: Path,
    *,
    source: str = "synthetic",
    max_sequences: int | None = None,
) -> list[str]:
    data_dir.mkdir(parents=True, exist_ok=True)
    if source == "synthetic":
        seqs = synthetic_motif_corpus(n=max_sequences or 2000)
    elif source == "bundled":
        bundled = data_dir / "sample_proteins.fasta"
        seqs = read_fasta(bundled)
    elif source == "uniprot":
        fasta = data_dir / "swissprot_sample.fasta"
        if fasta.exists():
            seqs = read_fasta(fasta)
        else:
            seqs = download_swissprot_sample(fasta, n=max_sequences or 5000)
    else:
        raise ValueError(f"unknown corpus source: {source}")
    if max_sequences:
        seqs = seqs[:max_sequences]
    return seqs


def corpus_to_corpus_file(sequences: list[str], path: Path) -> None:
    """Write one sequence per line for HuggingFace tokenizers BPE training."""
    path.write_text("\n".join(sequences) + "\n")


# Standard genetic code, all synonymous codons per amino acid. Real genomes use
# codon degeneracy (esp. the wobble 3rd position), which makes the nucleotide
# distribution near-uniform; using a single fixed codon per AA collapses entropy.
SYNONYMOUS_CODONS = {
    "A": ["GCT", "GCC", "GCA", "GCG"],
    "R": ["CGT", "CGC", "CGA", "CGG", "AGA", "AGG"],
    "N": ["AAT", "AAC"],
    "D": ["GAT", "GAC"],
    "C": ["TGT", "TGC"],
    "Q": ["CAA", "CAG"],
    "E": ["GAA", "GAG"],
    "G": ["GGT", "GGC", "GGA", "GGG"],
    "H": ["CAT", "CAC"],
    "I": ["ATT", "ATC", "ATA"],
    "L": ["TTA", "TTG", "CTT", "CTC", "CTA", "CTG"],
    "K": ["AAA", "AAG"],
    "M": ["ATG"],
    "F": ["TTT", "TTC"],
    "P": ["CCT", "CCC", "CCA", "CCG"],
    "S": ["TCT", "TCC", "TCA", "TCG", "AGT", "AGC"],
    "T": ["ACT", "ACC", "ACA", "ACG"],
    "W": ["TGG"],
    "Y": ["TAT", "TAC"],
    "V": ["GTT", "GTC", "GTA", "GTG"],
}


def genome_snippets_from_proteins(
    sequences: list[str], window: int = 512, stride: int = 256
) -> list[str]:
    """Genome windows by reverse-translating proteins with random synonymous codons.

    Codon degeneracy (random synonymous choice) reproduces the near-uniform
    nucleotide distribution of real genomes; a fixed codon per AA would collapse
    genome entropy and flatten the BPE token statistics.
    """
    import random

    rng = random.Random(0)
    genome = "".join(
        "".join(rng.choice(SYNONYMOUS_CODONS.get(a, ["GCT"])) for a in s)
        for s in sequences
    )
    out: list[str] = []
    for i in range(0, max(len(genome) - window, 0), stride):
        out.append(genome[i : i + window])
    if not out:
        out = [genome[:window]] if genome else []
    rng.shuffle(out)
    return out
