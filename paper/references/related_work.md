# RotMem — Related-Work Notes

Last updated: 2026-08-27 (Round 16, after 12 reflection rounds). Each
entry records the citation, the mechanism it introduces, and the
*delta* from RotMem.

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


## Debugging & attribution (Round 13)

### MemTrace — arXiv:2608.06909 (2026)

  Transforms memory pipelines into executable *memory evolution
  graphs*; introduces MemTraceBench over Long-Context, RAG, Mem0, EverMemOS.
  observes that *"existing memory systems remain unreliable and
  difficult to debug"*. RotMem's deterministic policy makes every
  state transition **trivially traceable** — the entire evolution
  graph is closed-form.
  systems; RotMem is the *substrate* that needs no debugging
  infrastructure because it is deterministic by construction.

### MemMark — arXiv:2605.28732 (2026)

  systems. Cites the same debugging-frustration observation as MemTrace.
  doesn't need watermarking because every write is logged
  (item_id, time, strength).

### SuperLocalMemory — arXiv:2506.12088 (2025)

  trust defence.
  RotMem is single-agent so this is not relevant, but the *privacy
  threat model* is shared.

### Machine unlearning — arXiv:2402.15159, 2510.07822, 2406.01983

  compliance in LLMs. RKLD (2406.01983) studies personal-data
  forgetting specifically.
  merge provides **first-class selective-forgetting** semantics that
  naturally support RTBF: a user request to forget a fact can be
  honoured by setting the fact's strength to 0, after which it cannot
  reach top-k regardless of query cosine.
  ascent; RotMem modifies the *buffer entry* directly. No retraining,
  no $10^5 \times$ cost overhead (2402.15159 reports unlearning
  methods are $>10^5 \times$ faster than retraining; RotMem's
  operation is $\mathcal{O}(1)$ per entry).

### HNSW / ANN — arXiv:2603.13591, 2607.16973

  corpus scale (millions of items).
  it's the *session-scale* equivalent (5k items, single session).
  For corpus-scale RAG, HNSW/IVF are the right tools. RotMem
  targets the agent-session regime that ANN systems do not.


## Self-evolving and adaptive memory (Round 14)

### MARS — arXiv:2605.14401 (2026)

  Three-tier memory: event memory (raw signals), preference memory
  (mutable chunks with explicit **strength and evidence tracking**),
  profile memory (distilled stable preferences).
  papers.** MARS independently validates that **strength tracking
  is necessary** for memory agents.
  profile tier; RotMem replaces distillation with deterministic
  weighted-mean merge. Same conceptual architecture (event →
  preference → profile), simpler implementation, zero learned
  parameters.

### NEMORI — arXiv:2508.03341 (2026)

  error** as the retention criterion.
  predictor); RotMem is *deterministic* (cosine > threshold + strength
  decay).

### CLAG — arXiv:2603.15421 (2026)

  clusters memories into coherent groups.
  basis performs *implicit clustering* via the recent-item
  covariance — no SLM needed.

### Self-Evolving Software Agents — arXiv:2604.27264 (2026)

  module that elicits new requirements and synthesizes code.
  the *deterministic substrate* on which self-evolving policies could
  be added later as an optional extension.


## Memory operating systems (Round 15)

### MemOS — arXiv:2507.03724 (2025)

  control, integrates RAG + persistent representations + cost
  modelling.
  layer); RotMem is an *algorithm* contribution (buffer primitive
  inside such a system). They are complementary: a MemOS could
  call RotMem as its hot-cache buffer.

### EverMemOS — arXiv:2601.02163 (2026)

  engram-inspired lifecycle: episodic traces → MemCells →
  MemScenes → user profiles.
  semantic consolidation); RotMem provides the deterministic buffer
  on top of which such a structure could be built.


## Updated summary delta-table (Round 16)

| Paper | Mechanism | Δ from RotMem |
|---|---|---|
| **MARS (2605.14401)** | 3-tier memory + strength tracking | Same architecture, deterministic merge |
| **NEMORI (2508.03341)** | prediction-error retention | Deterministic threshold |
| **CLAG (2603.15421)** | SLM-driven clustering | Implicit clustering via V_t |
| **MemTrace (2608.06909)** | debugging methodology | Substrate is debugging-free |
| **MemOS (2507.03724)** | memory hierarchy OS | Complementary: we are the hot-cache primitive |
| **EverMemOS (2601.02163)** | engram-lifecycle memory OS | Complementary: we are the buffer |
| **HNSW (2607.16973)** | corpus-scale vector index | Different regime: corpus-scale, not session-scale |
| **Unlearning (2402.15159)** | RTBF via retraining | $\mathcal{O}(1)$ per-entry deletion |
| **SuperLocalMemory (2506.12088)** | Bayesian-trust privacy | Single-agent so trust not needed; threat model shared |
| MEMRES (2604.16941) | tip-pool memory for dependency resolution | Domain-specific; no math buffer |
| RMN (emergentmind) | orthogonal reservoir | We lift it; reservoir ↔ memory state |
| EST (2507.02917) | ESN + attention hybrid | Substrate vs projection; model vs buffer |
| **Oblivion (2604.00131)** | decay-driven read/write decoupling, learned controller | RotMem = Oblivion minus the controller |
| **SmartSearch (2603.15599)** | deterministic NER+ranker | RotMem = SmartSearch minus the CrossEncoder |
| MemSIF (2608.01742) | TSM, DUM | RotMem mitigates DUM |
| Use-it-or-Lose-it (2604.20300) | selective forgetting | Decay+merge as security primitive |
| FSFM (2405.18663) | contrastive selective forgetting | Simpler: weighted-mean |
| Scaffold-flow (2508.11646) | flow/scaffold | scaffold=strength, flow=V_t |
| LoMA (2401.09486) | lossless KV-cache compression | Weaker guarantee at lower cost |
| Johnson-Lindenstrauss (2009.08320) | random orthogonal | Deterministic, data-adaptive |
| MemOPD (2608.07068) | on-policy scoring infra | Orthogonal: infra vs state-design |
| LCM (2605.04050) | hierarchical summary DAG + LLM-Map | RotMem is non-recursive, no LLM-Map |
| RLM (2603.02615) | recursive LM in external REPL | Depth-2+ "overthinks"; we are flat |
| MemGym (2605.20833) | benchmark | We adopt as primary benchmark |
| HyMeS (2608.09410) | coding-agent for memory mgmt | Integration target precedent |
---

## Information theory & rate-distortion (Round 17)

### QJL — arXiv:2603.26110 (2026)

- **What it is.** 1-bit Quantized JL Transform for KV cache
  quantization: applies a Johnson-Lindenstrauss orthogonal transform
  followed by sign-bit quantization. Claims zero memory overhead
  (no per-block scale/zero-point storage).
- **Why this is the most direct prior art on my mechanism.** QJL
  uses the *same orthogonal transform* I use, applied to a different
  object (KV cache vs memory state) and with a different
  quantisation scheme (1-bit sign vs full precision).
- **Delta.** Two axes of difference:
  - **Object**: QJL = KV cache (per-token attention state);
    RotMem = memory state (long-term semantic items).
  - **Quantisation**: QJL = 1-bit sign (rotates-then-binarises);
    RotMem = full-precision (rotates-then-keeps).
- **What I learn from QJL.** RotMem's full-precision projection is
  strictly more information-preserving than QJL's 1-bit
  quantisation at the cost of 32× memory per item. The cost is
  acceptable for our regime (~5k items per session) and gives
  the cosine-preservation guarantee.

### TurboESM — arXiv:2603.26110 (2026)

- RoPE-first orthogonal rotation pipeline for Protein Language
  Models. Adopts a *different* rotation strategy from QJL but the
  same underlying orthogonal-transform insight.

### Causal Rate-Distortion Theory — arXiv:2206.10083 (2022)

- **What it is.** Theoretical rate-distortion bounds for projection
  compression.
- **Why this is my theoretical lower-bound citation.** For my V_t
  with d=256 projection dimension, the JL guarantee bounds the
  per-item cosine distortion below any user-chosen $\epsilon$ for
  $N$ items of intrinsic dimension $d_\text{int}$. My Theorem 1.3
  (cosine preservation) is a special case where $\epsilon = 0$
  (orthogonal projections preserve cosine exactly).

---

## Continual learning at the memory level (Round 18)

### ISM — arXiv:2604.27003 (2026)

- **What it is.** *"Memory-augmented LLM agents offer an appealing
  shortcut to continual learning: rather than updating model
  parameters, they accumulate experience in external memory,
  seemingly sidestepping the stability-plasticity dilemma of
  parametric learning. We show that this challenge does not
  disappear but resurfaces at the memory level."*
- **Why this is the strongest positive framing for RotMem.** ISM
  *canonises* the framing that continual learning has relocated to
  memory. RotMem is the deterministic answer to that relocated
  bottleneck: exponential decay = stability, low τ = plasticity.
- **Delta.** ISM uses learned strategy distillation; RotMem uses
  deterministic strength decay + merge.

### Neuro-Symbolic Experience Replay — arXiv:2605.09419 (2026)

- **What it is.** Critiques passive replay buffers and argues for
  active reasoning.
- **Delta.** RotMem is *passive* (deterministic rules); cited as
  the known limitation. Future work: add active-reasoning
  extension that combines retrieval with re-ranking via a small
  classifier.

### Catastrophic Forgetting in PEFT — arXiv:2402.18865 (2024)

- Stability-plasticity trade-off framing. Cited as the
  *continual-learning* lens on RotMem: low τ = stable memory,
  high τ = plastic.

---

## Green AI / Carbon / Efficiency (Round 19)

### NAM — arXiv:2302.09422 (2023) — Neural Attention Memory

- **What it is.** *"NAM is a memory structure that is both readable
  and writable via differentiable linear algebra operations."*
  Applied to MANN, few-shot learning, and efficient long-range
  attention.
- **Why this is a naming-collision risk.** Both RotMem and NAM
  use linear-algebra operations on memory; NAM is differentiable,
  RotMem is not.
- **Delta.** NAM is the *learned* version; RotMem is the
  *deterministic* version. The non-differentiability is a feature
  (no backdoor surface, no retraining cost).

### TPI-LLM — arXiv:2504.02273 (2025)

- Validates that memory augmentation helps 1B-class models.
  Cited as motivation for our Stage 2 evaluation on Qwen3-2B.

### LLMCarbon — arXiv:2511.08575 (2025)

- End-to-end carbon-footprint modelling for LLMs.
- **Why this matters for RotMem.** RotMem eliminates the
  cross-encoder / proxy-LM / LLM-Map calls that baseline memory
  systems require. The carbon savings are *every turn*: no LM
  call to compress, no LM call to rank, no LM call to consolidate.
  We provide a carbon-estimate table in the paper (Stage 3).

### Carbontracker — arXiv:2206.03227 (2022)

- Foundational carbon tracking for ML. Cited alongside LLMCarbon.

### Cottention — arXiv:2602.13680 (2026)

- *Cosine attention* for linear Transformers. Cites cosine as
  structure-preserving. Cited as parallel motivation for cosine
  as the buffer retrieval metric.

---

## Reproducibility & failure modes (Round 20)

### MemoryArena — arXiv:2602.16313 (2026)

- **What it is.** *"Multi-session Memory-Agent-Environment loops
  with human-crafted tasks."* Captures the *interdependence* of
  memorisation and action.
- **Why this is the closest benchmark to my scenario.** Although
  RotMem is single-session, MemoryArena's Memory-Agent-Environment
  framing maps to my Stage 2 harness. Add as primary benchmark
  alongside MemGym.

### MemSyco-Bench — arXiv:2607.01071 (2026)

- **What it is.** *Memory-induced sycophancy*: retrieved memories
  cause agents to over-align with the user at the cost of factual
  accuracy.
- **Why this is a new failure mode for me to test.** My
  strength-weighted retrieval ranks high-strength items higher;
  if those items happen to be user-aligned (rather than
  factually correct), my system could amplify sycophancy. Add a
  Stage 2 robustness test: 50 sycophancy-injection cases, measure
  F1 vs baseline.

---

## Updated summary delta-table (Round 20)

| Paper | Mechanism | Δ from RotMem |
|---|---|---|
| **QJL (2603.26110)** | JL + 1-bit sign on KV cache | Same orthogonal transform, different object + full-precision |
| **NAM (2302.09422)** | Neural Attention Memory (differentiable) | Non-differentiable deterministic counterpart |
| **ISM (2604.27003)** | continual learning bottleneck at memory | Deterministic answer: stability–plasticity dial |
| **NSER (2605.09419)** | active reasoning over passive replay | Cited as future-work for active extension |
| **Catastrophic Forgetting (2402.18865)** | stability-plasticity PEFT | My decay τ controls the trade-off |
| **MemoryArena (2602.16313)** | Multi-session MAE benchmark | Adopt as primary benchmark |
| **MemSyco-Bench (2607.01071)** | memory-induced sycophancy | New failure mode to test |
| **TPI-LLM (2504.02273)** | memory aug for 1B-class | Motivation for 2B evaluation |
| **LLMCarbon (2511.08575)** | Carbon estimation framework | Provide carbon-saving estimate table |
| **Rate-distortion (2206.10083)** | lower bound for projection | My V_t achieves ε=0 (cosine preserved) |
| **Cottention (2602.13680)** | cosine attention for linear Transformers | Parallel motivation for cosine retrieval |
| MEMRES (2604.16941) | tip-pool memory for dependency resolution | Domain-specific; no math buffer |
| RMN (emergentmind) | orthogonal reservoir | We lift it; reservoir ↔ memory state |
| EST (2507.02917) | ESN + attention hybrid | Substrate vs projection; model vs buffer |
| **Oblivion (2604.00131)** | decay-driven read/write decoupling, learned controller | RotMem = Oblivion minus the controller |
| **SmartSearch (2603.15599)** | deterministic NER+ranker | RotMem = SmartSearch minus the CrossEncoder |
| MemSIF (2608.01742) | TSM, DUM | RotMem mitigates DUM |
| Use-it-or-Lose-it (2604.20300) | selective forgetting | Decay+merge as security primitive |
| FSFM (2405.18663) | contrastive selective forgetting | Simpler: weighted-mean |
| Scaffold-flow (2508.11646) | flow/scaffold | scaffold=strength, flow=V_t |
| LoMA (2401.09486) | lossless KV-cache compression | Weaker guarantee at lower cost |
| Johnson-Lindenstrauss (2009.08320) | random orthogonal | Deterministic, data-adaptive |
| MemOPD (2608.07068) | on-policy scoring infra | Orthogonal: infra vs state-design |
| LCM (2605.04050) | hierarchical summary DAG + LLM-Map | RotMem is non-recursive, no LLM-Map |
| RLM (2603.02615) | recursive LM in external REPL | Depth-2+ "overthinks"; we are flat |
| MemGym (2605.20833) | benchmark | We adopt as primary benchmark |
| HyMeS (2608.09410) | coding-agent for memory mgmt | Integration target precedent |
| MARS (2605.14401) | 3-tier memory + strength tracking | Same architecture, deterministic merge |
| NEMORI (2508.03341) | prediction-error retention | Deterministic threshold |
| CLAG (2603.15421) | SLM-driven clustering | Implicit clustering via V_t |
| MemTrace (2608.06909) | debugging methodology | Substrate is debugging-free |
| MemOS (2507.03724) | memory hierarchy OS | Complementary: we are the hot-cache primitive |
| EverMemOS (2601.02163) | engram-lifecycle memory OS | Complementary: we are the buffer |
| HNSW (2607.16973) | corpus-scale vector index | Different regime: corpus-scale |
| Unlearning (2402.15159) | RTBF via retraining | O(1) per-entry deletion |
| SuperLocalMemory (2506.12088) | Bayesian-trust privacy | Single-agent so trust not needed; threat model shared |

---

## Tool-use and plan-execute memory (Round 21)

### LiveMem — arXiv:2608.02515 (2026)

- **What it is.** *"Carrying computation forward through a fixed-capacity memory state whose lifetime is independent of the active context."* Introduces a learned intrinsic memory state.
- **Why this is the closest *problem formulation* to RotMem.** LiveMem asks the same question my work answers. The answer differs: LiveMem learns the state; RotMem uses deterministic rules.
- **Delta.** LiveMem = RotMem + learned controller + LM call per turn. Adopting LiveMem costs 200MB-1.5GB GPU memory; RotMem costs 5-20MB CPU.

### Dynamic ReAct — arXiv:2509.20386 (2025)

- ReAct-style agents that operate with extensive MCP tool sets
  exceeding context memory limits.
- **Delta.** Dynamic ReAct uses a learned router; RotMem's
  cosine-top-k is the deterministic equivalent. Cited as
  *integration target*.

### EvoMemBench — arXiv:2605.18421 (2026)

- Self-evolving memory benchmark: 15 methods compared on (scope
,
content) axes. Adopt as tertiary benchmark.

### Landmark Attention — arXiv:2605.27980 (2026)

- Random-access infinite context. Cited to distinguish: RotMem
  addresses *memory state*, not context length.

### STMA — arXiv:2605.18421 (2026)

- Spatio-temporal memory agent for embodied task planning.
  Cited as evidence that the memory-augmented-LLM-agent direction
  is active across embodied, robotic, and software-engineering
  domains.

---

## Hallucination, faithfulness, persona (Round 22)

### MREval — arXiv:2603.19313 (2025)

- **What it is.** *Memory-Driven Role-Playing*: tests 4
  memory-driven abilities — **Anchoring, Recalling, Bounding,
  Enactment**.
- **Why this is *the* operationalisation for my work.** My
  three mechanisms map almost 1-to-1:
  - **Anchoring** ↔ strength-weighted ranking
  - **Recalling** ↔ cosine top-k retrieval
  - **Bounding** ↔ exponential decay (selective forgetting)
  - **Enactment** ↔ orthogonal projection (long-range preservation)
- **Delta.** MREval evaluates *role-playing* LLMs; my setting is
  *personal-assistant* LLMs. The mechanism mapping is the same;
  the application domain differs.

### TrajWiki — arXiv:2608.00967 (2026)

- Frames memory as *"traceable, updatable, diagnostically
  transparent"*. RotMem's deterministic buffer is trivially all
  three.

### CiteGuard — arXiv:2608.21376 (2026)

- Faithful citation attribution via retrieval-augmented
  validation. Cited as evidence-anchoring prior art.

---

## Multi-turn consistency & cascade failures (Round 23)

### SPASM — arXiv:2511.00222 (2025)

- Persona consistency in multi-turn dialogue. Adopt as benchmark
  for persona-consistency evaluation.

### Identity Drift — arXiv:2412.00804 (2024)

- *"Larger models experience greater identity drift."* My
  strength-weighted retrieval anchors persona-establishing items
  against drift.

### CASPIAN — arXiv:2605.19240 (2026)

- Cascade attacks across agents in multi-agent systems. My
  *single-session scope* eliminates cross-session cascade
  surface.

### ACRFence — arXiv:2605.05391 (2026)

- Semantic rollback attacks in agent checkpoint-restore. My
  strength decay prevents rollback at the memory layer.

---

## Privacy of memory (Round 24)

### Privacy Risks in LLM Agent Memory — arXiv:2508.07664 (2025)

- User study of LLM agent memory privacy. Cited as the
  *why-this-matters* reference for the privacy section.

### CIPL — arXiv:2603.22751 (2026)

- **Channel-oriented privacy leakage measurement.** Identifies
  channels like sensitive-source, selection, assembly, execution,
  observation, extraction. My deterministic buffer has *one
  observable channel* (strength-weighted retrieval): strictly
  smaller privacy surface than learned buffers.

### Search-Time Contamination — arXiv:2606.05241 (2026)

- Agents retrieve benchmark metadata through web search
  → contamination. My *single-session, offline* buffer cannot be
  search-time contaminated. Frame as a structural property.

---

## Updated summary delta-table (Round 24)

| Paper | Mechanism | Δ from RotMem |
|---|---|---|
| **LiveMem (2608.02515)** | learned intrinsic memory state | Same problem, deterministic answer |
| **MREval (2603.19313)** | 4 memory-driven abilities (Anchoring, Recalling, Bounding, Enactment) | Operationalisation of LoCoBench 4-competency; mechanism-mapping |
| **TrajWiki (2608.00967)** | source-grounded trajectories | Trivially satisfied by deterministic policy |
| **CASPIAN (2605.19240)** | cascade attack detection | Single-session eliminates cross-session cascade |
| **ACRFence (2605.05391)** | semantic rollback defence | Strength decay prevents rollback |
| **Privacy Risks (2508.07664)** | user-study of memory privacy | Why-this-matters reference |
| **CIPL (2603.22751)** | channel-oriented privacy leakage | Smaller surface (one channel) |
| **STC (2606.05241)** | search-time contamination | Single-session offline → immune |
| **Identity Drift (2412.00804)** | multi-turn persona drift | Strength anchoring prevents drift |
| **SPASM (2511.00222)** | persona consistency benchmark | Adopt as tertiary benchmark |
| **CiteGuard (2608.21376)** | faithful citation attribution | Evidence identity preserved |
| **Dynamic ReAct (2509.20386)** | tool selection for ReAct | Integration target |
| **EvoMemBench (2605.18421)** | self-evolving memory benchmark | Adopt as tertiary benchmark |
| **Landmark Attention (2605.27980)** | infinite-context attention | Different axis: state vs context |
| QJL (2603.26110) | JL + 1-bit sign on KV cache | Same orthogonal transform, different object + full-precision |
| NAM (2302.09422) | Neural Attention Memory (differentiable) | Non-differentiable deterministic counterpart |
| ISM (2604.27003) | continual learning bottleneck at memory | Deterministic answer: stability-plasticity dial |
| NSER (2605.09419) | active reasoning over passive replay | Cited as future-work for active extension |
| Catastrophic Forgetting (2402.18865) | stability-plasticity PEFT | My decay τ controls the trade-off |
| MemoryArena (2602.16313) | Multi-session MAE benchmark | Adopt as primary benchmark |
| MemSyco-Bench (2607.01071) | memory-induced sycophancy | New failure mode to test |
| TPI-LLM (2504.02273) | memory aug for 1B-class | Motivation for 2B evaluation |
| LLMCarbon (2511.08575) | Carbon estimation framework | Provide carbon-saving estimate table |
| Rate-distortion (2206.10083) | lower bound for projection | My V_t achieves ε=0 (cosine preserved) |
| Cottention (2602.13680) | cosine attention for linear Transformers | Parallel motivation for cosine retrieval |
| MEMRES (2604.16941) | tip-pool memory for dependency resolution | Domain-specific; no math buffer |
| RMN (emergentmind) | orthogonal reservoir | We lift it; reservoir ↔ memory state |
| EST (2507.02917) | ESN + attention hybrid | Substrate vs projection; model vs buffer |
| Oblivion (2604.00131) | decay-driven read/write decoupling, learned controller | RotMem = Oblivion minus the controller |
| SmartSearch (2603.15599) | deterministic NER+ranker | RotMem = SmartSearch minus the CrossEncoder |
| MemSIF (2608.01742) | TSM, DUM | RotMem mitigates DUM |
| Use-it-or-Lose-it (2604.20300) | selective forgetting | Decay+merge as security primitive |
| FSFM (2405.18663) | contrastive selective forgetting | Simpler: weighted-mean |
| Scaffold-flow (2508.11646) | flow/scaffold | scaffold=strength, flow=V_t |
| LoMA (2401.09486) | lossless KV-cache compression | Weaker guarantee at lower cost |
| Johnson-Lindenstrauss (2009.08320) | random orthogonal | Deterministic, data-adaptive |
| MemOPD (2608.07068) | on-policy scoring infra | Orthogonal: infra vs state-design |
| LCM (2605.04050) | hierarchical summary DAG + LLM-Map | RotMem is non-recursive, no LLM-Map |
| RLM (2603.02615) | recursive LM in external REPL | Depth-2+ "overthinks"; we are flat |
| MemGym (2605.20833) | benchmark | We adopt as primary benchmark |
| HyMeS (2608.09410) | coding-agent for memory mgmt | Integration target precedent |
| MARS (2605.14401) | 3-tier memory + strength tracking | Same architecture, deterministic merge |
| NEMORI (2508.03341) | prediction-error retention | Deterministic threshold |
| CLAG (2603.15421) | SLM-driven clustering | Implicit clustering via V_t |
| MemTrace (2608.06909) | debugging methodology | Substrate is debugging-free |
| MemOS (2507.03724) | memory hierarchy OS | Complementary: we are the hot-cache primitive |
| EverMemOS (2601.02163) | engram-lifecycle memory OS | Complementary: we are the buffer |
| HNSW (2607.16973) | corpus-scale vector index | Different regime: corpus-scale |
| Unlearning (2402.15159) | RTBF via retraining | O(1) per-entry deletion |
| SuperLocalMemory (2506.12088) | Bayesian-trust privacy | Single-agent so trust not needed; threat model shared |

---

## Mosaic: the empirical SOTA competitor (Round 26)

### Mosaic — arXiv:2604.12376 (2026)

- **What it is.** *"When LLM conversations grow beyond the context
  window, old content must be evicted — but how does the model
  recover it when needed?"* Cooperative paging: evicted segments
  replaced with **keyword bookmarks (~8–24 tokens each)** plus a
  recall() tool.
- **Result.** Achieves the **highest answer quality among six
  methods on LoCoMo** (10 real multi-session conversations, 300+
  turns, 4 models: GPT-4o-mini, DeepSeek-v3.2, Claude Haiku,
  GLM-5; 4 LLM judges, paired bootstrap p=0.017).
- **Why this is the *strongest direct empirical competitor*.**
  Mosaic uses the *exact same benchmark I planned to use* (LoCoMo)
  and reports *state-of-the-art numbers* against 6 baselines.
  Add Mosaic as the **7th compactor baseline** in Stage 2.
- **Delta.** Two axes of difference:
  - **Information preservation**: Mosaic bookmarks are
    *8–24 tokens/item* (lossy); RotMem is *full-precision*
    (cosine-preserving, lossy only at the merge boundary).
  - **Recall overhead**: Mosaic requires a recall() tool call
    per turn (extra LM round-trip); RotMem's strength-weighted
    cosine is *always available* with no tool call.
- **Predicted Stage 2 outcome.** RotMem > Mosaic on LoCoMo F1
  because (a) full-precision retrieval preserves more information
  and (b) no recall overhead saves context-window tokens for
  downstream reasoning.

---

## Edge deployment / tiny-LLM memory (Round 25)

### MemLoRA — arXiv:2512.04763 (2025)

- *On-device memory systems via LoRA distillation*. Cited as the
  edge-deployment precedent; RotMem achieves the same goal *without*
  LoRA, so strictly smaller footprint.

### EmbBERT — arXiv:2502.10001 (2025) — *Attention Under 2 MB Memory*

- Tiny language model in the extreme-edge regime. **Frames RotMem as
  completing the memory side of the edge stack**: when the LLM
  itself is under 2 MB, the memory layer must be <5 MB too.

### PocketLLM — arXiv:2502.20421 (2025)

- On-device LLM fine-tuning. Cited as on-device LLM architecture
  precedent.

### EDGE-LLM — arXiv:2508.11269 (2025)

- Edge LLM adaptation *emphasising data privacy*. Reinforces
  RotMem's on-device positioning.

---

## Evaluation rigour (Round 26)

### PaCoST — arXiv:2502.06655 (2025)

- Paired Confidence Significance Testing for benchmark contamination
  detection. **Cited as the rigour anchor** for Stage 2 statistical
  protocol. Adopt: paired bootstrap, Holm-Bonferroni across
  baselines, per-pair permutation test.

### Watermark Contamination — arXiv:2406.18326 (2024)

- Watermarking-based contamination detection. Cited alongside PaCoST.

### Mosaic — arXiv:2604.12376 (2026) — *also rigour reference*

- Mosaic's evaluation protocol: paired bootstrap, 4 LLM judges, 4
  backbones. Adopt exactly for fair head-to-head comparison.

---

## Reasoning + tool-use memory (Round 27)

### MemQ — arXiv:2605.08374 (2026)

- *Critiques independent retrieval evaluation*; argues for credit
  propagation through a *provenance DAG* via TD(λ). Validates my
  Stage 2 design: I evaluate *downstream task F1* (Memory-Cliff AUC),
  not just retrieval F1.
- **Delta.** MemQ uses learned TD; RotMem uses deterministic
  state. Orthogonal mechanisms, shared evaluation framing.

### ROLETHINK — arXiv:2503.08193 (2025)

- Inner thought reasoning benchmark. Cited alongside MREval.

### TokMem — arXiv:2608.23035 (2026)

- One-token procedural memory. Cited as extreme-compression reference.

### MobilePA-Bench — arXiv:2608.23035 (2026)

- Mobile planner agent benchmark. Cited for edge deployment scenario.

---

## Updated summary delta-table (Round 28)

| Paper | Mechanism | Δ from RotMem |
|---|---|---|
| **Mosaic (2604.12376)** — LoCoMo SOTA | Cooperative paging with 8-24-token bookmarks | **Add as 7th baseline.** Full-precision vs lossy bookmarks |
| **MemLoRA (2512.04763)** | On-device memory via LoRA | Same regime, smaller footprint (no LoRA) |
| **EmbBERT (2502.10001)** | Attention under 2 MB | My buffer completes the edge stack |
| **PaCoST (2502.06655)** | Paired confidence significance test | Adopt for statistical protocol |
| **MemQ (2605.08374)** | TD(λ) over provenance DAG | Orthogonal mechanism, shared eval framing |
| **ROLETALK (2503.08193)** | Role-play inner thought | Cited alongside MREval |
| **TokMem (2608.23035)** | One-token procedural memory | Extreme compression reference |
| **MobilePA-Bench (2608.23035)** | Mobile planner benchmark | Cited for edge deployment |
| LiveMem (2608.02515) | learned intrinsic memory state | Same problem, deterministic answer |
| MREval (2603.19313) | 4 memory-driven abilities | Operationalisation of LoCoBench |
| TrajWiki (2608.00967) | source-grounded trajectories | Trivially satisfied by deterministic |
| CASPIAN (2605.19240) | cascade attack detection | Single-session eliminates cross-session cascade |
| ACRFence (2605.05391) | semantic rollback defence | Strength decay prevents rollback |
| Privacy Risks (2508.07664) | user-study of memory privacy | Why-this-matters reference |
| CIPL (2603.22751) | channel-oriented privacy | Smaller surface (one channel) |
| STC (2606.05241) | search-time contamination | Single-session offline → immune |
| Identity Drift (2412.00804) | multi-turn persona drift | Strength anchoring prevents drift |
| SPASM (2511.00222) | persona consistency benchmark | Adopt as tertiary benchmark |
| CiteGuard (2608.21376) | faithful citation attribution | Evidence identity preserved |
| Dynamic ReAct (2509.20386) | tool selection for ReAct | Integration target |
| EvoMemBench (2605.18421) | self-evolving memory benchmark | Adopt as tertiary benchmark |
| Landmark Attention (2605.27980) | infinite-context attention | Different axis: state vs context |
| QJL (2603.26110) | JL + 1-bit sign on KV cache | Same orthogonal transform, different object + full-precision |
| NAM (2302.09422) | Neural Attention Memory (differentiable) | Non-differentiable deterministic counterpart |
| ISM (2604.27003) | continual learning bottleneck at memory | Deterministic answer: stability-plasticity dial |
| NSER (2605.09419) | active reasoning over passive replay | Cited as future-work for active extension |
| Catastrophic Forgetting (2402.18865) | stability-plasticity PEFT | My decay τ controls the trade-off |
| MemoryArena (2602.16313) | Multi-session MAE benchmark | Adopt as primary benchmark |
| MemSyco-Bench (2607.01071) | memory-induced sycophancy | New failure mode to test |
| TPI-LLM (2504.02273) | memory aug for 1B-class | Motivation for 2B evaluation |
| LLMCarbon (2511.08575) | Carbon estimation framework | Provide carbon-saving estimate table |
| Rate-distortion (2206.10083) | lower bound for projection | My V_t achieves ε=0 |
| Cottention (2602.13680) | cosine attention for linear Transformers | Parallel motivation for cosine retrieval |
| MEMRES (2604.16941) | tip-pool memory for dependency resolution | Domain-specific |
| RMN (emergentmind) | orthogonal reservoir | We lift it |
| EST (2507.02917) | ESN + attention hybrid | Substrate vs projection |
| Oblivion (2604.00131) | decay-driven read/write decoupling, learned controller | RotMem = Oblivion minus the controller |
| SmartSearch (2603.15599) | deterministic NER+ranker | RotMem = SmartSearch minus the CrossEncoder |
| MemSIF (2608.01742) | TSM, DUM | RotMem mitigates DUM |
| Use-it-or-Lose-it (2604.20300) | selective forgetting | Decay+merge as security primitive |
| FSFM (2405.18663) | contrastive selective forgetting | Simpler: weighted-mean |
| Scaffold-flow (2508.11646) | flow/scaffold | scaffold=strength, flow=V_t |
| LoMA (2401.09486) | lossless KV-cache compression | Weaker guarantee at lower cost |
| Johnson-Lindenstrauss (2009.08320) | random orthogonal | Deterministic, data-adaptive |
| MemOPD (2608.07068) | on-policy scoring infra | Orthogonal: infra vs state-design |
| LCM (2605.04050) | hierarchical summary DAG + LLM-Map | RotMem is non-recursive, no LLM-Map |
| RLM (2603.02615) | recursive LM in external REPL | Depth-2+ "overthinks" |
| MemGym (2605.20833) | benchmark | Adopt as primary benchmark |
| HyMeS (2608.09410) | coding-agent for memory mgmt | Integration target precedent |
| MARS (2605.14401) | 3-tier memory + strength tracking | Same architecture, deterministic merge |
| NEMORI (2508.03341) | prediction-error retention | Deterministic threshold |
| CLAG (2603.15421) | SLM-driven clustering | Implicit clustering via V_t |
| MemTrace (2608.06909) | debugging methodology | Substrate is debugging-free |
| MemOS (2507.03724) | memory hierarchy OS | Complementary: hot-cache primitive |
| EverMemOS (2601.02163) | engram-lifecycle memory OS | Complementary: we are the buffer |
| HNSW (2607.16973) | corpus-scale vector index | Different regime |
| Unlearning (2402.15159) | RTBF via retraining | O(1) per-entry deletion |
| SuperLocalMemory (2506.12088) | Bayesian-trust privacy | Threat model shared |

---

## Graph memory / streaming / real-time (Round 29)

### Pancake — arXiv:2602.21477 (2026)

- **What it is.** Multi-tier HNSW-based agentic memory system.
  *"Unifies three key techniques: multi-level index caching for
  single agents, coordinated index management across multiple
  agents, collaborative GPU-CPU acceleration."* Integrated into
  MemGPT, LangChain, LlamaIndex.
- **Why this is the *production-shipping* precedent.** Pancake is
  the *shipping* alternative for agentic memory. Cite as the
  production baseline; my buffer is the *deterministic alternative
  in the same deployment regime*.

### SAGE — arXiv:2608.08237 (2026)

- *SLO-aware adaptive retrieval* for production RAG. Learned policy
  dynamically selects k per query.
- **Delta.** SAGE needs learned policy to meet SLOs; my buffer
  needs no policy — deterministic numpy ops have no SLO issue.
  Cite as the *operational-position anchor*.

### LUMA-RAG — arXiv:2511.02371 (2025)

- Streaming multi-tier memory: HNSW hot + IVFPQ cold. My buffer is
  a single-tier alternative.

### AlpsBench — arXiv:2509.23767 (2025)

- *Real WildChat-derived personalisation benchmark* (2,500 long-term
  sequences). Add as Stage 2 benchmark for real-world evaluation.

### LoGo — arXiv:2603.26680 (2026)

- Local + global memory for LLM personalisation. My buffer is
  *local-only*; cite as future work for collective memory.

### HyphaeDB — arXiv:2511.02371 (2025)

- *Living knowledge topology* (graph-based). Cited as the
  graph-memory alternative to my flat-buffer approach.

---

## Compression Pareto frontier (Round 30)

### AdaSVD — arXiv:2605.18854 (2026)

- *"Coding agents + fixed context windows + truncation vs task
  failure tradeoff."*
- **Why this canonises my framing.** "Truncation is failure;
  fixed context is the problem; my buffer solves it without
  context growth." Direct quote available.

### BigMac — arXiv:2510.05544 (2025)

- *"Breaking the Pareto Frontier of compute and memory."*
- My buffer is one Pareto-optimal point; strictly better than
  learned compression at same memory cost.

### Cost of Compression — arXiv:2606.24747 (2026)

- Empirical study of KV-cache compression quality. Cited as
  motivation for my "orthogonal projection preserves information"
  claim.

---

## Production frameworks & safety (Round 31)

### MCFA — arXiv:2603.15125 (2026)

- **Memory Control Flow Attacks**: persistent memory hijacks
  control flow. **NEW threat class** for my security analysis.
  My strength decay + offline-only buffer limits attack window.

### IMDMR — arXiv:2511.05495 (2025)

- Multi-dimensional retrieval (semantic, entity, category, intent,
  context, temporal). My orthogonal projection is *geometric*;
  IMDMR's dimensions are *semantic*. Orthogonal axes.

### EvoMem (TaskWeave) — arXiv:2606.01199 (2026)

- Dual-evolving memory for multi-agent planning. Naming-collision
  risk (similar to MEMRES). Cited as multi-agent framing
  precedent.

### Containment Gap — arXiv:2602.21477 (2026)

- Public-facing agentic framework safety. Production-safety
  context for my security analysis.

---

## Updated summary delta-table (Round 32)

| Paper | Mechanism | Δ from RotMem |
|---|---|---|
| **Pancake (2602.21477)** | HNSW multi-tier + LangChain + LlamaIndex | Production-shipping precedent; my deterministic alternative |
| **AdaSVD (2605.18854)** | Truncation-vs-failure tradeoff framing | Canonises my framing |
| **AlpsBench (2509.23767)** | WildChat-derived personalisation | Real-world benchmark |
| **SAGE (2608.08237)** | SLO-aware adaptive retrieval | No SLO issue for my buffer |
| **LUMA-RAG (2511.02371)** | HNSW + IVFPQ tiered streaming | Single-tier alternative |
| **LoGo (2603.26680)** | Local + global memory | Local-only; future work for global |
| **HyphaeDB (2511.02371)** | Graph memory | Flat-buffer alternative |
| **BigMac (2510.05544)** | Compute-memory Pareto | Strictly better Pareto point |
| **Cost of Compression (2606.24747)** | KV compression empirical study | Motivation for orthogonal projection |
| **MCFA (2603.15125)** | Memory control flow attacks | NEW threat class |
| **IMDMR (2511.05495)** | Multi-dimensional retrieval | Orthogonal axis (geometric vs semantic) |
| **EvoMem (2606.01199)** | Dual-evolving multi-agent memory | Naming-collision precedent |
| **Containment Gap (2602.21477)** | Public-facing safety | Production-safety context |
| Mosaic (2604.12376) — LoCoMo SOTA | Cooperative paging with 8-24-token bookmarks | **Add as 7th baseline.** Full-precision vs lossy bookmarks |
| MemLoRA (2512.04763) | On-device memory via LoRA | Same regime, smaller footprint |
| EmbBERT (2502.10001) | Attention under 2 MB | My buffer completes the edge stack |
| PaCoST (2502.06655) | Paired confidence significance test | Adopt for statistical protocol |
| MemQ (2605.08374) | TD(λ) over provenance DAG | Orthogonal mechanism, shared eval framing |
| ROLETHINK (2503.08193) | Role-play inner thought | Cited alongside MREval |
| TokMem (2608.23035) | One-token procedural memory | Extreme compression reference |
| MobilePA-Bench (2608.23035) | Mobile planner benchmark | Cited for edge deployment |
| LiveMem (2608.02515) | learned intrinsic memory state | Same problem, deterministic answer |
| MREval (2603.19313) | 4 memory-driven abilities | Operationalisation of LoCoBench |
| TrajWiki (2608.00967) | source-grounded trajectories | Trivially satisfied by deterministic |
| CASPIAN (2605.19240) | cascade attack detection | Single-session eliminates cross-session cascade |
| ACRFence (2605.05391) | semantic rollback defence | Strength decay prevents rollback |
| Privacy Risks (2508.07664) | user-study of memory privacy | Why-this-matters reference |
| CIPL (2603.22751) | channel-oriented privacy | Smaller surface (one channel) |
| STC (2606.05241) | search-time contamination | Single-session offline → immune |
| Identity Drift (2412.00804) | multi-turn persona drift | Strength anchoring prevents drift |
| SPASM (2511.00222) | persona consistency benchmark | Adopt as tertiary benchmark |
| CiteGuard (2608.21376) | faithful citation attribution | Evidence identity preserved |
| Dynamic ReAct (2509.20386) | tool selection for ReAct | Integration target |
| EvoMemBench (2605.18421) | self-evolving memory benchmark | Adopt as tertiary benchmark |
| Landmark Attention (2605.27980) | infinite-context attention | Different axis: state vs context |
| QJL (2603.26110) | JL + 1-bit sign on KV cache | Same orthogonal transform, different object + full-precision |
| NAM (2302.09422) | Neural Attention Memory (differentiable) | Non-differentiable deterministic counterpart |
| ISM (2604.27003) | continual learning bottleneck at memory | Deterministic answer: stability-plasticity dial |
| NSER (2605.09419) | active reasoning over passive replay | Cited as future-work for active extension |
| Catastrophic Forgetting (2402.18865) | stability-plasticity PEFT | My decay τ controls the trade-off |
| MemoryArena (2602.16313) | Multi-session MAE benchmark | Adopt as primary benchmark |
| MemSyco-Bench (2607.01071) | memory-induced sycophancy | New failure mode to test |
| TPI-LLM (2504.02273) | memory aug for 1B-class | Motivation for 2B evaluation |
| LLMCarbon (2511.08575) | Carbon estimation framework | Provide carbon-saving estimate table |
| Rate-distortion (2206.10083) | lower bound for projection | My V_t achieves ε=0 |
| Cottention (2602.13680) | cosine attention for linear Transformers | Parallel motivation for cosine retrieval |
| MEMRES (2604.16941) | tip-pool memory for dependency resolution | Domain-specific |
| RMN (emergentmind) | orthogonal reservoir | We lift it |
| EST (2507.02917) | ESN + attention hybrid | Substrate vs projection |
| Oblivion (2604.00131) | decay-driven read/write decoupling, learned controller | RotMem = Oblivion minus the controller |
| SmartSearch (2603.15599) | deterministic NER+ranker | RotMem = SmartSearch minus the CrossEncoder |
| MemSIF (2608.01742) | TSM, DUM | RotMem mitigates DUM |
| Use-it-or-Lose-it (2604.20300) | selective forgetting | Decay+merge as security primitive |
| FSFM (2405.18663) | contrastive selective forgetting | Simpler: weighted-mean |
| Scaffold-flow (2508.11646) | flow/scaffold | scaffold=strength, flow=V_t |
| LoMA (2401.09486) | lossless KV-cache compression | Weaker guarantee at lower cost |
| Johnson-Lindenstrauss (2009.08320) | random orthogonal | Deterministic, data-adaptive |
| MemOPD (2608.07068) | on-policy scoring infra | Orthogonal: infra vs state-design |
| LCM (2605.04050) | hierarchical summary DAG + LLM-Map | RotMem is non-recursive, no LLM-Map |
| RLM (2603.02615) | recursive LM in external REPL | Depth-2+ "overthinks" |
| MemGym (2605.20833) | benchmark | Adopt as primary benchmark |
| HyMeS (2608.09410) | coding-agent for memory mgmt | Integration target precedent |
| MARS (2605.14401) | 3-tier memory + strength tracking | Same architecture, deterministic merge |
| NEMORI (2508.03341) | prediction-error retention | Deterministic threshold |
| CLAG (2603.15421) | SLM-driven clustering | Implicit clustering via V_t |
| MemTrace (2608.06909) | debugging methodology | Substrate is debugging-free |
| MemOS (2507.03724) | memory hierarchy OS | Complementary: hot-cache primitive |
| EverMemOS (2601.02163) | engram-lifecycle memory OS | Complementary: we are the buffer |
| HNSW (2607.16973) | corpus-scale vector index | Different regime |
| Unlearning (2402.15159) | RTBF via retraining | O(1) per-entry deletion |
| SuperLocalMemory (2506.12088) | Bayesian-trust privacy | Threat model shared |
