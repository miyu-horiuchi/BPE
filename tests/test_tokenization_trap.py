"""Tests for BPE tokenization experiment."""

from bpe.corpus import synthetic_motif_corpus
from bpe.spectral import rank_bigram_merges, target_vocab_from_spectrum
from bpe.tokenizers import AminoAcidTokenizer, DomainBPETrainer
from bpe.zipf import compute_zipf_metrics, compare_zipf_profiles


def test_synthetic_corpus_has_motifs():
    seqs = synthetic_motif_corpus(n=50, seed=0)
    assert len(seqs) == 50
    joined = "".join(seqs)
    assert "GPGPGPGP" in joined or "DEAD" in joined


def test_single_aa_vocab():
    tok = AminoAcidTokenizer()
    ids = tok.encode("MKTL")
    assert len(ids) == 4
    assert tok.decode(ids) == "MKTL"


def test_domain_bpe_trains(tmp_path):
    seqs = synthetic_motif_corpus(n=500, seed=1)
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("\n".join(seqs))
    tok = DomainBPETrainer(vocab_size=128, min_frequency=2).train(corpus, name="test_bpe")
    assert tok.vocab_size > 24
    # Repeated motifs should merge on at least some sequences
    merged = [s for s in seqs if len(tok.tokenize(s)) < len(s) * 0.95]
    assert len(merged) > 0


def test_zipf_single_aa_vs_bpe(tmp_path):
    seqs = synthetic_motif_corpus(n=500, seed=2)
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("\n".join(seqs))
    aa = AminoAcidTokenizer()
    bpe = DomainBPETrainer(vocab_size=256, min_frequency=2).train(corpus, name="domain_bpe_256")
    metrics = compare_zipf_profiles(seqs, [aa, bpe])
    assert metrics[0].n_types <= 21
    assert bpe.vocab_size > aa.vocab_size


def test_spectral_merges_ranked():
    seqs = synthetic_motif_corpus(n=100, seed=3)
    merges = rank_bigram_merges(seqs, top_k=10)
    assert len(merges) <= 10
    assert merges[0].score >= merges[-1].score


def test_suggested_vocab_in_range():
    seqs = synthetic_motif_corpus(n=100, seed=4)
    v = target_vocab_from_spectrum(seqs)
    assert v in {64, 128, 256, 512, 1024, 2048}
