"""Tokenizers: single-AA, GPT-2 BPE on proteins, domain-adaptive BPE."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Protocol

from tokenizers import Tokenizer, models, pre_tokenizers, trainers

AA_VOCAB = list("ACDEFGHIKLMNPQRSTVWY")
SPECIAL = ["<pad>", "<bos>", "<eos>", "<unk>"]


class SequenceTokenizer(Protocol):
    name: str

    def encode(self, sequence: str) -> list[int]: ...

    def decode(self, ids: list[int]) -> str: ...

    def tokenize(self, sequence: str) -> list[str]: ...


@dataclass
class AminoAcidTokenizer:
    """One token per amino acid — the baseline that collapses to AA frequency."""

    name: str = "single_aa"

    def __post_init__(self) -> None:
        self._stoi = {aa: i + len(SPECIAL) for i, aa in enumerate(AA_VOCAB)}
        self._itos = {i: aa for aa, i in self._stoi.items()}
        for i, tok in enumerate(SPECIAL):
            self._itos[i] = tok

    @property
    def vocab_size(self) -> int:
        return len(SPECIAL) + len(AA_VOCAB)

    def tokenize(self, sequence: str) -> list[str]:
        return [c if c in self._stoi else "<unk>" for c in sequence.upper()]

    def encode(self, sequence: str) -> list[int]:
        return [self._stoi.get(c, 0) for c in sequence.upper() if c.isalpha()]

    def decode(self, ids: list[int]) -> str:
        return "".join(
            self._itos.get(i, "")
            for i in ids
            if self._itos.get(i, "") not in SPECIAL
        )


@dataclass
class HuggingFaceBPETokenizer:
    """Wrapper around a trained or loaded HF tokenizers BPE model."""

    tokenizer: Tokenizer
    name: str

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size(with_added_tokens=True)

    def tokenize(self, sequence: str) -> list[str]:
        return self.tokenizer.encode(sequence).tokens

    def encode(self, sequence: str) -> list[int]:
        return self.tokenizer.encode(sequence).ids

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save(str(directory / "tokenizer.json"))
        (directory / "meta.json").write_text(json.dumps({"name": self.name}))


class DomainBPETrainer:
    """Train byte-level BPE on protein sequences (no spaces between residues)."""

    def __init__(
        self,
        vocab_size: int = 512,
        min_frequency: int = 2,
        special_tokens: list[str] | None = None,
    ) -> None:
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.special_tokens = special_tokens or SPECIAL

    def train(self, corpus_path: Path, name: str = "domain_bpe") -> HuggingFaceBPETokenizer:
        tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
        # ByteLevel on ASCII AA strings = char-level BPE with working merge statistics.
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            min_frequency=max(1, self.min_frequency),
            special_tokens=self.special_tokens,
            show_progress=False,
        )
        tokenizer.train([str(corpus_path)], trainer)
        return HuggingFaceBPETokenizer(tokenizer=tokenizer, name=name)

    @staticmethod
    def load(directory: Path) -> HuggingFaceBPETokenizer:
        meta = json.loads((directory / "meta.json").read_text())
        tok = Tokenizer.from_file(str(directory / "tokenizer.json"))
        return HuggingFaceBPETokenizer(tokenizer=tok, name=meta["name"])


@dataclass
class GPT2ProteinTokenizer:
    """Apply GPT-2 BPE directly to concatenated amino-acid strings.

    This is the negative control: merges follow English byte statistics and
    do not align with biological motif boundaries.
    """

    name: str = "gpt2_on_protein"

    def __post_init__(self) -> None:
        from transformers import GPT2TokenizerFast

        self._tok = GPT2TokenizerFast.from_pretrained("gpt2")

    @property
    def vocab_size(self) -> int:
        return len(self._tok)

    def tokenize(self, sequence: str) -> list[str]:
        # GPT-2 expects spaces for word-level BPE; without them, merges are arbitrary.
        return self._tok.tokenize(sequence)

    def encode(self, sequence: str) -> list[int]:
        return self._tok.encode(sequence, add_special_tokens=False)

    def decode(self, ids: list[int]) -> str:
        return self._tok.decode(ids)


def load_tokenizer(kind: str, artifacts_dir: Path) -> SequenceTokenizer:
    if kind == "single_aa":
        return AminoAcidTokenizer()
    if kind == "gpt2":
        return GPT2ProteinTokenizer()
    path = artifacts_dir / kind
    if path.exists():
        return DomainBPETrainer.load(path)
    raise FileNotFoundError(f"tokenizer {kind!r} not found under {artifacts_dir}")


def train_all_domain_bpes(
    corpus_path: Path,
    out_dir: Path,
    vocab_sizes: Iterable[int] = (128, 256, 512, 1024),
) -> dict[int, HuggingFaceBPETokenizer]:
    out_dir.mkdir(parents=True, exist_ok=True)
    trained: dict[int, HuggingFaceBPETokenizer] = {}
    for vs in vocab_sizes:
        name = f"domain_bpe_{vs}"
        trainer = DomainBPETrainer(vocab_size=vs)
        tok = trainer.train(corpus_path, name=name)
        tok.save(out_dir / name)
        trained[vs] = tok
    return trained
