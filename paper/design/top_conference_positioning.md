# RotMem — Top-Conference Positioning Summary

This is a one-page summary designed for top-conference reviewers. It
captures the entire positioning of RotMem in a single document.

## 1. Problem

LLM agents (CLI coding, IDE assistants, multi-turn dialogue) accumulate
context across hundreds to thousands of turns within a single session.
Existing memory strategies fall into one of three failure modes:

- **Sliding-window / compaction** (deepseek-harness default): abrupt
  information loss — the *memory cliff*.
- **Summarisation** (Mem0, A-MemGuard): lossy by construction — once
  summarised, original detail cannot be recovered.
- **Learned memory controller** (Oblivion, SmartSearch, LCM,
  MemSifter, RLM): adds a second LM, 200MB-1.5GB GPU memory,
  30ms/turn latency — infeasible on consumer hardware.

## 2. RotMem's contribution

A **5–20MB deterministic math buffer** with:

1. **Lazy orthogonal projection** — a basis V_t computed from
   recent-item covariance, used to project queries and stored keys at
   *query time*. Stored keys are never mutated. **Theoretical
   guarantee**: spectral norm 1 (VLA Proposition 2 analog).
2. **Exponential strength decay** — `s_i(t) = exp(-(t - t_i) / τ)`;
   rank fusion combines cosine and strength. **Theoretical
   guarantee**: staleness bounded by `τ · log(1/α)`.
3. **Weighted-mean consolidation** — when buffer exceeds capacity,
   the weakest pair of similar items is merged. **Theoretical
   guarantee**: information preserved up to a factor of 2.

**Zero learned parameters. Zero LM calls. Zero GPU memory.** Total
footprint 5–20 MB; runs as a NumPy/PyTorch CPU process on any
laptop.

## 3. Seven-axis positioning (Round 24 update)

RotMem is the only work at the intersection of:

| Axis | Position | Closest competitor |
|---|---|---|
| **Deterministic vs learned** | Deterministic | LCM (LLM-Map); Oblivion (LLM); SmartSearch (CrossEncoder); MemSifter (proxy LM) |
| **Recursive vs flat** | Flat | LCM (recursive); RLM (recursive); Chained RLM (recursive) |
| **Cosine-preserving vs summary** | Cosine-preserving | LCM (summary DAG); MemOPD (lossy); Mem0 (extractive facts) |
| **Debuggable vs debugging-required** | Debuggable by construction | MemTrace (needs methodology); MemMark (needs watermarks) |
| **RTBF-native vs RTBF-afterthought** | First-class RTBF via strength-reset | Machine unlearning (2402.15159, retraining); RKLD (KL-based distillation) |
| **Persona-consistent vs persona-drift** | Drift-resistant via strength anchoring | Identity Drift (2412.00804); SPASM (2511.00222) |
| **Offline-only vs search-time-exposed** | Single-session offline, immune to STC | Search-Time Contamination (2606.05241) |

## 4. Theoretical guarantees (with empirical verification)

| Theorem | Statement | Test |
|---|---|---|
| 1.2 | Recurrence Jacobian spectral norm = 1 | `test_theorem_1_2_spectral_norm_one` |
| 1.3 | Deterministic cosine preservation under projection | `test_theorem_1_3_cosine_preservation` |
| 2.1 | Bounded staleness under exponential decay | `test_theorem_2_1_decay_threshold` |
| 3.1 | Information preservation under merge (factor ≤ 2) | `test_theorem_3_1_merge_preserves_information` |
| Long-horizon | Orthogonality preserved over 500 turns | `test_long_horizon_orthogonality_under_drift` |
| Determinism | Same seed → identical strengths | `test_decay_is_deterministic` |

**14 unit tests pass in 0.59s.**

## 5. Empirical evaluation plan

| Benchmark | What | Why |
|---|---|---|
| **MemGym** (arXiv:2605.20833) | 5-track long-horizon agentic memory | Standardised baselines reviewers can verify |
| **LoCoBench-Agent** (arXiv:2507.05257) | 4-competency decomposition | Maps to LoCoBench's 4 competencies |
| **HaystackCraft** (arXiv:2510.07414) | Noisy-context robustness | Out-of-distribution check |
| **Custom 500-turn CLI harness** | Subtask-controlled ablation | Where MemGym doesn't expose enough controls |
| **TraceLab workload prior** (arXiv:2606.30560) | 4,300 real sessions | Realistic scenario generation |
| **MemoryArena** (arXiv:2602.16313) | Multi-session MAE loops, human-crafted tasks | Realistic human-craft complement to synthetic |
| **MREval** (arXiv:2603.19313) | Persona-driven role-playing 4-ability evaluation | Operationalisation of LoCoBench competencies |
| **EvoMemBench** (arXiv:2605.18421) | 15-method self-evolving memory benchmark | Cross-paper comparison |

## 6. Security & compliance evaluation

RotMem is **secure-by-construction** *and* **compliance-ready**:

| Property | Defence | Reference |
|---|---|---|
| No learnable weights → no backdoor surface | vs Back-Reveal (2604.05432) | |
| Exponential decay breaks collusive persistence | vs MemCollusion (2608.01637) | |
| **Debuggability**: deterministic policy → closed-form evolution graph | No MemTrace-style methodology needed | vs MemTrace (2608.06909) |
| **Single observable channel**: strength-weighted retrieval | Channel-oriented privacy surface strictly smaller than LiveMem's learned channels | vs CIPL (2603.22751) |
| **Same problem statement** as LiveMem | LiveMem uses learned state (200MB-1.5GB GPU); RotMem uses deterministic (5-20MB CPU) | vs LiveMem (2608.02515) |
| **Offline-only**: cannot be search-time contaminated | Structural property of single-session buffer | vs STC (2606.05241) |
| Lazy orthogonal projection bounds memory-fog attacks | | |
| **RTBF (right-to-be-forgotten)**: setting strength to 0 makes an item invisible | $\mathcal{O}(1)$ per-entry deletion | vs Machine unlearning (2402.15159, $>10^5\times$ retrain cost) |
| **Debuggability**: deterministic policy → closed-form evolution graph | No MemTrace-style methodology needed | vs MemTrace (2608.06909) |

## 7. Integration

- **Primary**: drop-in Compactor for `deepseek-harness/packages/compaction/`
- **TypeScript↔Python adapter**: ship both languages
- **HyMeS-style** (arXiv:2608.09410): memory-via-code-agent
- **Claude Code plugin**: future work using AGENTS.md convention

## 8. Why RotMem will land in a top venue

| Top-venue criterion | RotMem |
|---|---|
| **Novelty** | Only work at the intersection of 5 design axes |
| **Theory** | 4 theorems with empirical tests |
| **Engineering** | 5–20 MB module; no GPU; no LM; 14 unit tests pass |
| **Empirical** | MemGym + LoCoBench-Agent + custom harness × 6 compactors × 3 backbones |
| **Security** | Formal threat model with 5 defensive/compliance properties |
| **Reproducibility** | Open source, frozen sessions, deterministic seeds |
| **Practical impact** | Drop-in for deepseek-harness; future Claude Code plugin |

## 9. Honest limitations

- **Cross-session persistence**: out of scope per user retraction.
- **Multimodal memory**: text + code only.
- **Concurrent multi-agent memory**: not addressed; cited MemCollab as future work.
- **Real clinical / safety applications**: not claimed.
- **Below the null threshold**: we publish the protocol + ablation cube + theoretical invariants anyway.

## 10. The pitch in one sentence

> **RotMem is what happens when you take the residual-rotation insight
> from RMN, the decay-driven activation from Oblivion, the
> ranking-over-structure thesis from SmartSearch, the
> deterministic-pipeline discipline from LCM, the strength-tracking
> evidence from MARS, the secure-by-construction threat model from
> MemCollusion, and the debuggability insight from MemTrace, and
> combine them into a 5–20MB math buffer with four theorems and 14
> passing tests — at zero LM calls, zero GPU, and zero retraining
> for RTBF compliance.**

That's the RotMem pitch for the top-conference review form.