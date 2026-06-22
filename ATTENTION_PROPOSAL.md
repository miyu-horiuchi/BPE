# The second half of the trap: attention that matches the data's spectrum

## What the scaling proof just told us (and why it changes the architecture question)

`results/scaling/scaling_proof.png` — fixed TinyGPT, same params/steps, only the tokenizer changes:

| Tokenizer | Vocab | p_median | tok/res | **bits/residue** |
|-----------|------:|---------:|--------:|-----------------:|
| single_aa | 24 | 0.42 | 0.99 | 3.869 |
| domain_bpe_256 | 256 | 0.84 | 0.49 | **3.535 (best)** |
| domain_bpe_1000 | 1000 | 1.23 | 0.41 | 3.537 |
| gpt2_on_protein | 50257 | 1.68 | 0.58 | 4.572 (worst) |

Two facts matter for the architecture question:

1. **Domain BPE beats single-AA at fixed model size** (3.54 vs 3.87 bits/residue, ~9%). The tokenization trap is real and the payoff shows up in actual LM training, not just in the Zipf table.
2. **`p_median` is necessary but not sufficient.** GPT-2 has the *highest* Zipf exponent (1.68) and the *worst* bits/residue. A high Zipf exponent built from the *wrong* merges (English byte statistics) actively hurts. The exponent only helps when the tokens correspond to real biological structure (domain BPE).

Fact (2) is the bridge to attention. A Zipfian token inventory gives the model the *right units*. But once you have the right units, the model still has to learn the *relationships between them* — and that is the attention operator's job. If the operator's inductive bias doesn't match the structure of those relationships, you waste the tokenizer's head start. So the architecture question is not "what's a cooler attention" — it's **"what operator has the same relationship to the data's coupling spectrum that domain-BPE has to the data's token spectrum?"**

That reframing is what makes one option clearly the most theoretically grounded.

## The recommendation: Spectral Composition Attention (SCA)

**One line:** keep softmax attention's content routing, but bias it with the empirical residue-coupling operator (the same spectral object you used to derive the optimal tokenizer), and add an explicit multi-scale residual so motif→domain composition is represented, not just inferred.

This is the only one of the four options that is *derived from the data distribution the same way the tokenizer was*. The others (plain SSM/Mamba, plain hierarchical pooling) are reasonable but are generic sequence-mixing swaps — they don't use the spectral structure that is the whole basis of your tokenization-trap result. SCA is the direct continuation of "derive the optimal operator from the spectral effects of the data."

### Why vanilla attention is the bottleneck (theory)

There is a well-established equivalence: a single softmax-attention layer trained with a masked/auto-regressive objective on biological sequences learns, in its attention logits, a **pairwise coupling matrix** — empirically these recover residue–residue contacts and coevolution (this is why ESM/MSA-Transformer attention maps predict contact maps; it's a learned Potts/DCA model). So attention already captures **pairwise interactions** — exactly the "patterns and interactions" you said it's stuck at.

What it does *not* have a bias toward:

- **The right metric.** Softmax attention scores are `q·k` in a learned but *isotropic* space. The data's couplings live on a low-rank, highly anisotropic manifold (a handful of dominant eigenmodes of the PMI/covariance operator carry most of the structure — same reason a few hundred BPE merges capture most of the token structure). Generic attention has to *learn* that anisotropy from scratch with no inductive bias, which is sample-inefficient.
- **Composition / higher-order structure.** Proteins are hierarchical: residue → motif → secondary structure → domain. Pairwise attention represents order-2 couplings; composition (a motif is more than the sum of its residue pairs) has to be emulated through depth and is not represented explicitly.

### The three components of SCA

1. **Spectral bias (the core idea).** Estimate the empirical token-coupling operator `C` from the corpus — the same PMI / co-occurrence matrix your `bpe/spectral.py` already builds for merge ranking. Take its top-`k` eigenvectors `U_k` (k ≈ 16–64; this is the "intrinsic dimension" of the coupling, analogous to the ~256-merge sweet spot in the table above). Add a low-rank, data-derived bias to the attention logits:

   `logit_ij = (q_i·k_j)/√d  +  λ · φ(x_i)ᵀ M φ(x_j)`,  where `φ(x) = U_kᵀ e(x)` projects a token onto the dominant coupling modes and `M` (k×k) is learned.

   This is *exactly* the tokenizer move applied to attention: project onto the dominant spectral modes of the data instead of working in the raw isotropic space.

2. **Multi-scale residual stream (the "residuals" half you flagged).** Run two parallel residual streams at different granularities — a fine stream over BPE tokens and a coarse stream over pooled motif-blocks (pool by the BPE merge boundaries you already compute) — and let the coarse stream write back into the fine one each block. This gives explicit residue→motif→domain composition instead of forcing depth to fake it. Cost is sub-linear in the coarse stream, so it's CPU-friendly.

3. **(Optional) anisotropic value mixing.** Replace the post-attention residual add with a gated mix weighted by spectral energy, so high-information (rare, high-Zipf-tail) tokens get a larger residual write. This is the natural way to make the residual path "Zipf-aware."

### Why this over the alternatives

- **vs plain SSM/Mamba:** SSMs swap the mixing mechanism but keep a generic, content-free recurrence; they don't exploit the coupling spectrum. They'd likely match attention, not beat it on the metric that matters here. Good as a *baseline*, not the hypothesis.
- **vs plain hierarchical pooling:** Component 2 already captures the useful part of "hierarchical." Pure pooling without the spectral bias throws away the exact structure your tokenizer result says is load-bearing.
- **SCA is falsifiable in your existing harness** (below), and it's the same methodology end-to-end: *derive the operator from the data's spectrum.*

## The experiment (drops into the harness we just built)

The proof harness is already tokenizer-agnostic and reports bits/residue at fixed params. The architecture A/B is the mirror image: **fix the tokenizer (domain_bpe_256, the winner), swap the attention block**, compare bits/residue at fixed params + steps.

1. Add `attn={"vanilla","sca","ssm"}` to `TinyGPT` (swap only the mixing block; keep d_model/params matched — SCA's extra params are just `M` (k×k) and the projection, kept tiny).
2. Build `C` once from the training corpus via the existing `bpe/spectral.py` co-occurrence counts; cache `U_k`.
3. Run all three with identical config; report bits/residue + loss curves exactly like `scaling_proof.png`.

**Falsifiable predictions (so we know if the theory is wrong):**
- SCA < vanilla on bits/residue at fixed params (primary).
- The gain *grows* with sequence length / depth (composition compounds).
- Ablating the spectral bias (λ=0) collapses SCA back to vanilla — i.e. the gain comes from the spectrum, not the extra params.
- SSM ≈ vanilla (controls for "any non-attention mixer helps").

All four are CPU-runnable at the current TinyGPT scale.

## Open clarifications before I build it

1. **Scope of the A/B:** just `vanilla vs sca` (fastest, cleanest test of the hypothesis), or include the `ssm` control too?
2. **Coupling operator source:** build `C` from token co-occurrence (uses `bpe/spectral.py` as-is) vs from raw residue PMI (closer to DCA/contacts, slightly more code). I'd start with token co-occurrence since it reuses existing code.
3. **Synthetic limit:** the synthetic corpus has *injected* motifs, so it may flatter any motif-aware operator. If SCA wins there, the honest next step is one real Swiss-Prot confirmation run. OK to plan that as a follow-up?
