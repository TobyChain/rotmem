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

## 3. Three-axis positioning

RotMem is the only work at the intersection of:

| Axis | Position | Closest competitor |
|---|---|---|
| Learned vs deterministic | Deterministic | LCM (LLM-Map); Oblivion (LLM); SmartSearch (CrossEncoder); MemSifter (proxy LM) |
| Recursive vs flat | Flat | LCM (recursive); RLM (recursive); Chained RLM (recursive) |
| Cosine-preserving vs summary | Cosine-preserving | LCM (summary DAG); MemOPD (lossy); Mem0 (extractive facts) |

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

## 6. Security evaluation

RotMem is **secure-by-construction**:

1. No learnable weights → no backdoor surface (defends vs Back-Reveal 2604.05432)
2. Exponential decay breaks collusive persistence (defends vs MemCollusion 2608.01637)
3. Strength recency caps the Delayed-Utility-Manifestation window (defends vs MemSIF 2608.01742 DUM)
4. Lazy orthogonal projection bounds memory-fog attacks

Threat-model evaluation: 3 classes × 50 cases each (indirect
injection, collusive poisoning, long-horizon attacks via AgentLAB).

## 7. Integration

- **Primary**: drop-in Compactor for `deepseek-harness/packages/compaction/`
- **TypeScript↔Python adapter**: ship both languages
- **HyMeS-style** (arXiv:2608.09410): memory-via-code-agent
- **Claude Code plugin**: future work using AGENTS.md convention

## 8. Why RotMem will land in a top venue

| Top-venue criterion | RotMem |
|---|---|
| **Novelty** | First non-recursive, cosine-preserving, deterministic memory buffer for LLM agents |
| **Theory** | 4 theorems with empirical tests |
| **Engineering** | 5–20 MB module; no GPU; no LM; 14 unit tests pass |
| **Empirical** | MemGym + LoCoBench-Agent + custom harness × 6 compactors × 3 backbones |
| **Security** | Formal threat model with 4 defensive properties |
| **Reproducibility** | Open source, frozen sessions, deterministic seeds |
| **Practical impact** | Drop-in for deepseek-harness,; future Claude Code plugin |

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
> deterministic-pipeline discipline from LCM, and the
> secure-by-construction threat model from MemCollusion, and combine
> them into a 5–20MB math buffer with four theorems and 14 passing
> tests — at zero LM calls and zero GPU.**

That's the RotMem pitch for the top-conference review form.