# RotMem — Related-Work Notes

Last updated: 2026-08-27. Each entry records the citation, the
mechanism it introduces, and the *delta* from RotMem.

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
- **Why we cite them anyway.** Name overlap; readers should not be
  confused.

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
- **What we change.** We replace the cyclic orthogonal reservoir with
  a learnable rotation matrix `V_t` that is refreshed on demand, and
  we add an explicit `strength` signal that reservoir models lack.

---

## Orthogonal-to-RotMem papers

### MemOPD — arXiv:2608.07068 (2026) — Liu et al.

- **What it is.** Internal scoring *infra* for on-policy distillation
  in memory-compression agents: it records inputs/outputs, restores
  causal visibility, and packs them for efficient teacher scoring.
- **Delta.** MemOPD solves *how to score* memory updates on-policy;
  RotMem is *what the state looks like*. We could optionally adopt
  MemOPD's packing infra for our evaluation runs.

### Mem0 — arXiv:2504.19413

- **What it is.** Production-grade long-term memory that extracts
  atomic facts, embeds them, and retrieves top-k.
- **Delta.** Mem0 rebuilds memory via summarisation; we maintain a
  continuous residual state.

### HyMEM — arXiv:2603.10291 (2026)

- **What it is.** Graph-based hybrid memory for GUI agents that
  couples discrete high-level symbolic nodes with continuous
  trajectory embeddings.
- **Delta.** HyMEM is graph-structured and GUI-specific; RotMem is a
  flat residual buffer and is domain-agnostic.

### MeMento — arXiv:2608.01456 (2026)

- **What it is.** Multimodal memory compressor for embodied
  decision-making under partial observation.
- **Delta.** MeMento is multimodal + embodied; RotMem is text + code
  and CPU-only.

### StreamMeCo / Goal-Directed Search — arXiv:2511.21726 (2025)

- **What it is.** Argues that goal-agnostic compression biases memory
  toward benchmarks; goal-directed search on raw data can outperform.
- **Delta.** StreamMeCo argues against compression. RotMem *is* a
  form of compression, but one with formal retention guarantees.

### DecentMem — arXiv:2605.22721 (2026)

- **What it is.** Decentralised dual-pool memory for self-evolving
  multi-agent systems; an exploitation pool of consolidated
  trajectories + an exploration pool of LLM-generated candidates.
- **Delta.** DecentMem is decentralised + multi-agent + learned
  reweighting; RotMem is centralised + single-agent + deterministic.

### Zombie Agents — arXiv:2602.15654 (2026)

- **What it is.** Documents a persistent-attack surface for
  self-evolving agents: an attacker plants a payload in one session
  that survives and weaponises later sessions.
- **Delta.** RotMem has no learned policy → narrower attack surface
  by construction. We cite this paper as part of our threat-model
  discussion.

---

## Threat-model citations

### A-MemGuard — arXiv:2510.02373 (2025)

- **What it is.** Proactive defence framework for LLM-agent memory
  poisoning.
- **Why we cite.** We adopt their adversarial-injection test design
  as our robustness extension (§6 of the experimental design).

### Zombie Agents — arXiv:2602.15654 (2026)

- (As above.) Provides the persistence-attack framing.

---

## Internal-prior citations (this lab)

- `deepseek-harness/packages/compaction/{compaction,command-compact,compaction-basic,compaction-tool-result-pruner}` — the existing in-house compactors that RotMem is benchmarked against.

---

## Bibliography file

The full BibTeX entries are in `paper/references/references.bib`
(population deferred to paper-writing stage).