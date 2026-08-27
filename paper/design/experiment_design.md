# RotMem — Experimental Design

## 1. Variable Space

### Independent variables (control)

- `compactor ∈ {compaction-basic, mem0, amemguard, smartsearch, lcm, rotmem}`  — **6 compactors** after Round 12 (LCM added)
- `backbone ∈ {qwen3-2b, qwen3-9b, minimax-m3}`
- `seed ∈ {0, 1, 2}`
- `dataset_split ∈ {split_a, split_b}`

### Independent variables (sweep / ablation)

- `rotation ∈ {none, period_10, period_50, period_200, drift_triggered}`
- `decay_tau ∈ {none, 1, 5, 50, 500}` (turns)
- `merge_threshold ∈ {0.85, 0.92, 0.97, none}`
- `buffer_cap ∈ {0.5, 1.0, 2.0 × default}`
- `orthogonalisation ∈ {qr, gram_schmidt, householder}`
- `retrieval_weight ∈ {strength_only, cosine_only, strength_x_cosine}`
- `horizon ∈ {100, 500, 1000}`

### Dependent variables (metrics)

| Metric | Symbol | Direction |
|---|---|---|
| Memory-Cliff AUC | `Δ_F1(100→500)` | higher better (less drop) |
| Retrieval-Precision@5 | `P@5` | higher better |
| Long-Tail-Recall (≤50→≥400) | `R_tail` | higher better |
| Compaction-Info-Loss | `L_info` | lower better |
| Wall-clock per turn | `t_turn (ms)` | lower better |
| Memory footprint @ 5k turns | `M_bytes` | lower better |
| Adapter parameters | `θ_adapter` | lower better |

### Dependent variables (theoretical)

| Metric | Definition | Direction |
|---|---|---|
| Cosine-preservation rate | fraction of (item, query) pairs where the projected cosine equals the raw cosine (≤1e-6 error) | higher better — verifies lazy rotation correctness |
| Spectral-radius drift | `|ρ(V_t) − 1|` over time | lower better — verifies V_t remains orthogonal under repeated refresh |
| Information-preservation lower bound | `Σ s_i² · (1 − λ_max(M_t))` from the spectral bound in theory.md §4.3 | higher better |

---

## 2. Sample-Size Justification

Per-condition sample size is **100** sessions × **500 turns** = 50,000
turns per condition.

Power analysis: with paired bootstrap at α=0.05, β=0.2, the smallest
detectable difference at this sample size is **~2.5 F1 points**
(σ≈5). Our pre-registered threshold of 1 pt is therefore below the
detection limit — we explicitly accept the null-result risk.

For the *theoretical* metrics (cosine-preservation, spectral drift),
sample size is the number of query/update operations ≈ 50,000 per
condition; these are deterministic and do not require bootstrapping.

---

## 3. Pairing & Blocking

- All compactors run on the **same session seed**, paired by session_id.
- Per-turn metrics are paired by `(compactor, session_id, turn_idx)`.
- Cross-condition comparisons use **paired bootstrap** of the per-pair
  difference, never independent samples.

---

## 4. Pre-Registration

The following is locked before any analysis runs:

| Decision | Value |
|---|---|
| Primary metric | `Δ_F1(100→500)` on Memory-Cliff AUC |
| Null threshold | RotMem − best-baseline < 1 pt |
| Effect-size threshold | Cohen's d ≥ 0.5 |
| Multiple-testing correction | Holm-Bonferroni across the **6 compactors** |
| Significance level | α = 0.05 |
| Random seeds | 0, 1, 2 |
| Theoretical pre-registration | V_t must remain orthogonal (|ρ−1| < 1e-3) for all 500 turns, every seed |

If `Δ_F1(100→500)` falls below the null threshold for all three
backbones, the contribution is null and we publish the experimental
protocol + the ablation cube + the theoretical invariance results
honestly.

---

## 5. Benchmark Selection (Round 12 update)

### Primary benchmark: **MemGym** (arXiv:2605.20833)

We adopt MemGym as the **primary Stage 2 benchmark**. MemGym unifies:

- **τ²-bench** (tool-use dialogue)
- **MEMGYM-DR** (deep-research search)
- **SWE-Gym + MEMGYM-CODEQA** (coding)
- (plus 2 more tracks omitted for space)

This replaces our custom 500-turn CLI harness (§5.1 below) and gives
us **standardised baselines** that reviewers can independently verify.

### Secondary benchmark: custom 500-turn CLI harness

For ablation studies that need fine-grained control over turn structure,
we also build a synthetic 500-turn CLI harness. Subtasks:

| Subtask | Typical turns | Difficulty |
|---|---:|---|
| Refactor | 50–80 | M |
| TDD fix | 40–70 | M |
| Multi-file migration | 80–120 | H |
| Type-error whack-a-mole | 30–60 | M |
| Dependency upgrade | 60–90 | H |
| Bug hunt | 40–70 | H |
| Doc-coverage sweep | 30–50 | L |

Every session contains at least one of each. Subtask boundaries are
hidden — the agent only sees the user's natural language request.

### Tertiary benchmark: LoCoBench-Agent (arXiv:2507.05257)

For the 4-competency decomposition (accurate retrieval, test-time
learning, long-range, selective forgetting).

### TraceLab-derived workload prior

From arXiv:2606.30560 — 4,300 Claude Code + Codex sessions, 350k LLM
steps, 430k tool calls. We use the **workload distribution** (turn
count distribution, tool-call distribution) to inform our synthetic
patterns.

### Quaternary benchmark: MemoryArena (arXiv:2602.16313)

Multi-session Memory-Agent-Environment loops with human-crafted
tasks. Adopted as the second primary benchmark for Stage 2;
provides the *human-craft* test scenario that complements
MemGym's synthetic tasks.

### MemSyco-Bench (arXiv:2607.01071) — sycophancy robustness test

Robustness test (50 sycophancy-injection cases): measure whether
high-strength retrievals bias the LLM toward user-aligned (rather
than factually correct) answers. Predicted: RotMem's deterministic
strength weighting amplifies user-aligned items if they have high
retrieval history; mitigations explored in Stage 3.

---

## 6. Failure-Mode Analysis

After all runs complete, we sample **50 worst-case sessions** (lowest
per-session F1) and hand-classify each failure into:

1. Buffer overflow (rotation or merge failed)
2. Retrieval miss (top-k wrong despite correct buffer)
3. Backbone reasoning failure (correct memory retrieved, but LLM
   misuses it)
4. Harness bug
5. Adversarial / out-of-distribution input

The category distribution itself is reported.

---

## 7. Reproducibility

- Every session seed and turn sequence is hashed and stored.
- Compactor parameters are versioned in `configs/`.
- A single command `make reproduce` re-runs all main experiments.
- Teacher traces (if needed for cross-validation) are not required
  for RotMem; we re-publish the 100 session seeds.

---

## 8. The Compactor Baseline Table (Round 12 update)

| Compactor | What it is | LM calls | Adapter params | Where it fails us |
|---|---|---:|---:|---|
| `compaction-basic` | deepseek-harness default sliding-window | 0 | 0 | throws away old items → memory cliff |
| `mem0` | extractive summarisation every 50 turns | 0 (rule-based) | 0 | summarisation IS lossy; slow |
| `amemguard` | write-time summary | 0 | 0 | MemSIF's DUM problem |
| `smartsearch` | NER + CrossEncoder rank fusion | 0 (encoder runs on CPU) | ~200MB INT8 CrossEncoder | CrossEncoder = learned; ~30ms/turn |
| **`lcm`** | hierarchical summary DAG + LLM-Map | **≥1 per LLM-Map** | summary DAG | summary-DAG is lossy; recursion adds latency |
| **`rotmem`** (ours) | orthogonal basis + decay + merge | **0** | **0** | none observed; ~1ms/turn |

**LCM is now our closest head-to-head competitor** (added in R12).
LCM beats Claude Code on OOLONG (per its abstract) but uses:
- A **hierarchical summary DAG** (lossy)
- **LLM-Map primitives** (still calls the LM)

RotMem claims:
- Strictly **more information-preserving** than LCM (no lossy summary)
- Strictly **cheaper** than LCM (no LLM-Map calls)
- **Comparable or better F1** on MemGym-CODEQA (predicted)

**SmartSearch** is the next-closest (cross-encoder instead of LLM-Map,
but still a learned ranker).

---

## 9. Extension Studies (post-main)

| Study | Adds |
|---|---|
| Cross-domain transfer (MemGym all 5 tracks) | Standardised baselines for all regimes |
| 50 real-user IDE replays | Out-of-distribution leak check (uses TraceLab-style noisy distractors) |
| Information-theoretic upper bound | Theoretical analysis (spectral preservation) |
| Adversarial injection (MemCollusion 2608.01637) | Robustness test (50 cases) |
| SmartSearch head-to-head | Direct rank-fusion vs strength-cosine comparison |
| LCM head-to-head | Lossy summary-DAG vs cosine-preserving comparison |
| HyMeS (2608.09410) integration demo | Memory-via-code-agent integration with deepseek-harness |
| Cross-backbone (Qwen3 / Llama-3 / Gemma / MiniMax-M3) | Model-agnostic V_t verification |