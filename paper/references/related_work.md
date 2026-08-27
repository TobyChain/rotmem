# RotMem — Related-Work Notes

Last updated: 2026-08-27 (Round 5, after 4 reflection rounds). Each entry
records the citation, the mechanism it introduces, and the *delta* from
RotMem.

---

## TL;DR positioning

RotMem is a **5–20MB deterministic math buffer** for single-session
long-horizon LLM agents. Its three mechanisms are:

1. **Lazy orthogonal rotation** — a deterministic orthogonal basis V_t
   computed from recent-item covariance, used at *query-time* to project
   both the query and the stored keys. Stored keys are never mutated.
2. **Exponential strength decay** — a scalar time-constant τ that
   smooths the retrieval ranking toward recent items.
3. **Weighted-mean consolidation** — when buffer exceeds capacity, the
   weakest pair of similar items (cosine > τ_sim) is replaced by their
   strength-weighted average.

All three mechanisms are **deterministic rules grounded in
information-theoretic priors**. There is **no learned controller** and
**no second LM**.

The single sentence positioning: **RotMem = Oblivion-minus-the-controller**
(a learned-decay controller replaced by a deterministic exponential),
= **SmartSearch-minus-the-CrossEncoder** (deterministic rank fusion
without a learned ranker), and = **LoMA-minus-the-bit-perfectness**
(cosine-preserving rather than bit-perfect-lossless).

---

## Closest naming neighbour

### MEMRES — arXiv:2604.16941 (2026) — Duy Minh et al.

- **What it is.** An agentic Python-dependency resolver. Gemma-2 9B
  + four cooperating components: Self-Evolving Memory (tip pool +
  shortcuts), Error Pattern KB (200+ import→package mappings), Semantic
  Import Analyser, Python-2 heuristic detector.
- **Result.** On HG2.9K, 2503/2890 (86.6%) snippets resolved.
- **Why this is *not* RotMem.** Their MEMRES is a domain-specific
  coding tool. Ours is a generic, math-only buffer architecture that
  drops into any LLM agent. We have no error KB, no semantic analyser,
  no heuristic detector, and no domain knowledge.

---

## Architectural origin

### Residual Memory Networks (RMN) — emergentmind topic page; Pinna et al. (13 Aug 2025)

- **What it is.** Reservoir-computing paradigm where
  `m(t) = V_m · m(t−1) + V_x · x(t)`, with `V_m` a cyclic orthogonal
  matrix (spectral radius `ρ = 1`). Past inputs are stored as a
  lossless rotation in a fixed-size state.
- **Why we lift it.** The RMN property — that an orthogonal update
  preserves past information in a low-rank residual — is exactly what
  LLM-agent memory needs to avoid the *memory cliff*.
- **What we change.** We replace the cyclic orthogonal reservoir with a
  *learnable* orthogonal basis V_t that is refreshed on demand, and
  we add an explicit `strength` signal that reservoir models lack.

### Echo State Transformer (EST) — arXiv:2507.02917 (2025) — Bendi-Ouis & Hinaut

- **What it is.** Hybrid architecture combining Transformer attention
  with reservoir computing dynamics, achieving SOTA on classification
  and detection tasks while reducing compute cost.
- **Why this is the closest architectural cousin.** EST combines
  ESN-style state with attention. RotMem combines an ESN-style
  orthogonal basis with cosine retrieval.
- **Delta.** EST uses the ESN state as a *processing substrate*
  (replaces attention); RotMem uses the orthogonal basis only as a
  *projection* at query-time, leaving the backbone LLM untouched.
  EST is a *model*; RotMem is a *memory buffer*.

---

## Closest behavioural neighbours

### Oblivion — arXiv:2604.00131 (2026)

- **What it is.** A memory control framework that casts forgetting
  as decay-driven reductions in accessibility, not explicit deletion.
  Decouples read and write paths: read decides when to consult memory
  based on agent uncertainty and buffer sufficiency; write decides
  what to reinforce by tracing response contributions.
- **Why this is the closest behavioural twin.** Both RotMem and
  Oblivion share the read/write decoupling and decay-based strength
  reweighting. Both deliver the "use it or lose it" property.
- **Delta (RotMem's contribution).** Oblivion still uses an **LLM-based
  controller** for both read and write decisions. RotMem replaces
  the controller with deterministic thresholds:
  - Oblivion's "read when uncertain" → RotMem's "retrieve by
    strength-weighted cosine" (top-k always; ranking handles
    uncertainty implicitly)
  - Oblivion's "write to strengthen" → RotMem's "strength = 1 on
    write, exponential decay thereafter"
- **Why this matters.** We show the *learned controller is
  unnecessary* in this setting, dropping 1.5GB FP16 / 400MB INT4 of
  GPU memory and ~30ms/turn of inference latency per session.

### SmartSearch — arXiv:2603.15599 (2026)

- **What it is.** Retrieves from raw, unstructured conversation
  history using a fully deterministic pipeline: NER-weighted
  substring matching + CrossEncoder+ColBERT rank fusion. The
  CrossEncoder is the only learned component, runs on CPU in ~650ms.
- **Why this is a thesis cousin.** SmartSearch's abstract opens
  with: *"Recent conversational memory systems invest heavily in
  LLM-based structuring at ingestion time and learned retrieval
  policies at query time. We show that neither is necessary."* —
  RotMem shares this thesis.
- **Oracle insight (SmartSearch).** Retrieval recall reaches 98.6%
  but, *without intelligent ranking*, only 22.5% of gold evidence
  survives truncation to the token budget. **This is the
  ranking-over-compression argument** we adopt.
- **Delta.** SmartSearch still uses a learned CrossEncoder ranker;
  RotMem uses strength-weighted cosine (also deterministic, but
  without learned weights).

### MemSIF — arXiv:2608.01742 (2026)

- **What it is.** Structured Interaction-to-Fact memory framework for
  long-term memory agents. Identifies two recurring misalignment
  patterns in long-term interaction:
  - **Temporal-Structural Misalignment (TSM)** — temporal proximity
    does not reliably align with topical relatedness.
  - **Delayed Utility Manifestation (DUM)** — write-time salience
    does not reliably predict future query utility.
- **Why this is relevant.** MemSIF formalises exactly the failure
  mode that RotMem mitigates.
- **Delta.** RotMem directly mitigates DUM via:
  - *Lazy rotation*: stored keys never change after insertion, so
    the value of an item does not depend on write-time context.
  - *Strength decay*: an item's retrieval ranking depends on
    query-time recency-weighted strength, not write-time salience.
  TSM is not directly addressed by RotMem; we cite it as future
  work (topic-aware re-ranking would address it).

### Selective Forgetting — arXiv:2604.20300 (2026); FSFM arXiv:2405.18663 (2024)

- **What they are.** Use-it-or-Lose-it frames selective forgetting
  as a design *choice* delivering efficiency, quality, and security
  benefits. FSFM uses contrastive dispersion for forgotten classes
  vs compactness for preserved ones.
- **Delta.** RotMem's decay+merge implements selective forgetting
  with a *simpler* mechanism: exponential decay + weighted-mean
  consolidation (no contrastive loss). The security benefit (active
  forgetting of malicious inputs) emerges automatically: an
  adversarially injected memory loses retrieval priority within
  `τ · log(1/0.5)` turns, even if it has high write-time strength.

---

## Memory-consolidation theory

### Scaffold-flow memory — arXiv:2508.11646 (2025)

- **What it is.** Computational model of memory consolidation that
  decomposes memory into *flow* (fast state-dependent responses)
  and *scaffold* (slowly-changing constraints).
- **Why this is the theoretical anchor.** RotMem's strength field
  *is* the scaffold (slowly-changing, governs retrieval ranking);
  RotMem's lazy V_t basis *is* the flow (responds to recent
  items).
- **Delta.** Scaffold-flow is a general principle; we instantiate it
  with concrete deterministic rules and report empirical effects.

---

## Compression-theoretical foundations

### LoMA — arXiv:2401.09486 (2024)

- **What it is.** Lossless Compressed Memory Attention: compresses
  the KV cache after every `tc` generated tokens with a compression
  ratio that preserves bit-perfect reconstruction.
- **Delta.** LoMA claims *bit-perfect losslessness*; RotMem is
  *cosine-preserving* (near-lossless for retrieval, but not
  bit-perfect). Our theoretical guarantee is weaker but the
  implementation cost is orders of magnitude lower (no model
  re-training, no per-token compression).

### Johnson-Lindenstrauss transforms — arXiv:2009.08320 (2020)

- **What they are.** Random orthogonal projections approximately
  preserve all pairwise Euclidean distances in a point set.
- **Why we cite them.** RotMem's V_t is a *deterministic*
  orthogonal projection (computed from recent-item covariance), not
  a random one. Strictly more information-preserving than random JL.
- **Delta.** Our V_t adapts to the data distribution via the
  eigendecomposition of recent items' covariance, whereas random JL
  projects uniformly. We do not rely on the JL concentration
  bound — our basis is *optimal* for the recent subspace.

---

## Orthogonal-RNN background

### Orthogonal Gated Recurrent Unit — arXiv:2504.05646 (2025)

- **What it is.** Recurrent mechanism using the low-rank structure
  of K-V matrices via Neumann-Cayley transformation. Achieves
  orthogonal recurrence with bounded gradient norms.
- **Why we cite it.** Establishes the *orthogonal RNN* prior that
  underwrites RotMem's V_t design — orthogonal updates are the
  canonical way to preserve past information in fixed-size state.

---

## Orthogonal-to-RotMem papers

### MemOPD — arXiv:2608.07068 (2026) — Liu et al.

- Internal scoring *infra* for memory-compression agents. RotMem
  is orthogonal: state-design vs scoring-infra.

### Mem0 — arXiv:2504.19413 (2024)

- Episodic store with extractive facts. Suffers the memory cliff.

### HyMEM — arXiv:2603.10291 (2026)

- Graph-structured GUI-agent memory. Not residual, not plug-and-play.

### MeMento — arXiv:2608.01456 (2026)

- Multimodal compression for embodied agents. Not text/code.

### StreamMeCo — arXiv:2511.21726 (2025)

- Goal-agnostic compression. RotMem is residual + goal-agnostic.

### Zombie Agents — arXiv:2602.15654 (2026)

- Persistent-attack surface of self-evolving memory. **RotMem
  narrows attack surface by construction**: deterministic policy
  has no learned weights to be poisoned.

### MemMA — arXiv:2603.18718 (2026)

- Multi-agent memory coordination. Single-agent deterministic
  buffer.

### xRAG — arXiv:2406.02266 (2024)

- Extreme context compression for RAG via feature re-interpretation.
  Compresses *retrieved documents*; RotMem compresses the *memory
  state itself*.

### Infini-attention — arXiv:2512.23862 (2025)

- Compressed memory for SLMs using past-segment summaries. Reports
  *"retrieval accuracy drops with repeated memory compressions
  over long sequences"*. This is the failure my ablation A1
  (drop rotation) directly tests — and my lazy rotation avoids it
  by *not mutating stored keys*.

### IGPO — arXiv:2510.14967 (2025)

- Information-gain intrinsic reward for multi-turn agents. Cited
  as the per-step credit signal RotMem *does not* need.

### π-Distill — arXiv:2602.04942 (2026)

- Privileged-hindsight distillation. Cited as the theoretical
  anchor of the *learned-controller approach we deliberately
  rejected* (deterministic rules proved sufficient).

### A-MemGuard — arXiv:2510.02373 (2025)

- Proactive defence for memory poisoning. Adopted as the design
  template for our adversarial-injection test (extension).

---

## Internal-prior citations (this lab)

- `deepseek-harness/packages/compaction/{compaction,command-compact,compaction-basic,compaction-tool-result-pruner}` — the existing in-house compactors that RotMem is benchmarked against.

---

## Summary delta-table

| Paper | Mechanism | Δ from RotMem |
|---|---|---|
| MEMRES (2604.16941) | tip-pool memory for dependency resolution | Domain-specific; no math buffer |
| RMN (emergentmind) | orthogonal reservoir m(t)=Vm·m(t-1)+Vx·x(t) | We lift it; reservoir ↔ memory state |
| EST (2507.02917) | ESN + attention hybrid | Substrate vs projection; model vs buffer |
| **Oblivion (2604.00131)** | decay-driven read/write decoupling, **learned** controller | **RotMem = Oblivion minus the controller** |
| **SmartSearch (2603.15599)** | deterministic NER+ranker | **RotMem = SmartSearch minus the CrossEncoder** |
| MemSIF (2608.01742) | identifies TSM, DUM | RotMem directly mitigates DUM |
| Use-it-or-Lose-it (2604.20300) | selective forgetting as design choice | Decay+merge as security primitive |
| FSFM (2405.18663) | contrastive selective forgetting | Simpler: weighted-mean consolidation |
| Scaffold-flow (2508.11646) | flow/scaffold decomposition | Concrete instantiation of scaffold=strength, flow=V_t |
| LoMA (2401.09486) | lossless KV-cache compression | Weaker guarantee (cosine-preserving) at lower cost |
| Johnson-Lindenstrauss (2009.08320) | random orthogonal distance preservation | Our V_t is *deterministic*, data-adaptive |
| MemOPD (2608.07068) | on-policy scoring infra | Orthogonal: infra vs state-design |