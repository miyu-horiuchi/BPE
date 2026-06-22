"""Tests for BPE tokenization experiment."""

from bpe.corpus import synthetic_motif_corpus
from bpe.report import genome_table_markdown, protein_table_markdown
from bpe.spectral import rank_bigram_merges, target_vocab_from_spectrum
from bpe.tokenizers import AminoAcidTokenizer, DomainBPETrainer
from bpe.zipf import compute_distribution_metrics


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
    merged = [s for s in seqs if len(tok.tokenize(s)) < len(s) * 0.95]
    assert len(merged) > 0


def test_distribution_metrics_shape(tmp_path):
    seqs = synthetic_motif_corpus(n=300, seed=2)
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("\n".join(seqs))
    aa = AminoAcidTokenizer()
    bpe = DomainBPETrainer(vocab_size=256, min_frequency=2).train(corpus, name="bpe_256")
    m_aa = compute_distribution_metrics(seqs, aa, display_name="Single AA", vocab_override=20)
    m_bpe = compute_distribution_metrics(seqs, bpe, display_name="BPE 256", vocab_override=256)
    assert 0 < m_aa.p_median < 2
    assert m_bpe.p_median > m_aa.p_median
    assert 0 <= m_aa.entropy_pct <= 100


def test_spectral_merges_ranked():
    seqs = synthetic_motif_corpus(n=100, seed=3)
    merges = rank_bigram_merges(seqs, top_k=10)
    assert len(merges) <= 10
    assert merges[0].score >= merges[-1].score


def test_markdown_table_format():
    from bpe.zipf import DistributionMetrics

    rows = [
        DistributionMetrics("Single AA", 20, 0.60, 0.60, 0.0, 98.0),
        DistributionMetrics("BPE 500", 500, 1.12, 1.12, 1.12, 88.0),
    ]
    md = protein_table_markdown(rows)
    assert "| Tokenizer | Vocab | p_median |" in md
    assert "**1.12**" in md
    assert "0.60" in md  # below 1.0, not bold
