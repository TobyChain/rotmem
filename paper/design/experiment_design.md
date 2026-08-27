# RotMem — Experimental Design

## 1. Variable Space

### Independent variables (control)

- `compactor ∈ {compaction-basic, mem0, amemguard, rotmem}`
- `backbone ∈ {qwen3-2b, qwen3-9b, minimax-m3}`
- `seed ∈ {0, 1, 2}`
- `dataset_split ∈ {split_a, split_b}`

### Independent variables (sweep / ablation)

- `rotation ∈ {none, period_10, period_50, period_200, drift_triggered}`
- `decay_tau ∈ {none, 1, 5, 50, 500}` (turns)
- `merge_threshold ∈ {0.85, 0.92, 0.97, none}`
- `buffer_cap ∈ {0.5, 1.0, 2.0} × default}`
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

---

## 2. Sample-Size Justification

Per-condition sample size is **100** sessions × **500 turns** = 50,000
turns per condition.

Power analysis: with paired bootstrap at α=0.05, β=0.2, the smallest
detectable difference at this sample size is **~2.5 F1 points**
(σ≈5). Our pre-registered threshold of 1 pt is therefore below the
detection limit — we explicitly accept the null-result risk.

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
| Multiple-testing correction | Holm-Bonferroni across the 4 compactors |
| Significance level | α = 0.05 |
| Random seeds | 0, 1, 2 |

If `Δ_F1(100→500)` falls below the null threshold for all three
backbones, the contribution is null and we publish the experimental
protocol + the ablation cube honestly.

---

## 5. Subtask Composition (500-turn CLI harness)

Each session is a composition of:

| Subtask | Typical turns | Difficulty |
|---|---:|---|
| Refactor | 50–80 | M |
| TDD fix | 40–70 | M |
| Multi-file migration | 80–120 | H |
| Type-error whack-a-mole | 30–60 | M |
| Dependency upgrade | 60–90 | H |
| Bug hunt | 40–70 | H |
| Doc-coverage sweep | 30–50 | L |

Every session must contain at least one of each. Subtask boundaries are
hidden — the agent only sees the user's natural language request.

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
- Teacher traces (if any are needed for cross-validation) are not
  required for RotMem; we re-publish the 100 session seeds.
- Compactor parameters are versioned in `configs/`.
- A single command `make reproduce` re-runs all main experiments.

---

## 8. Extension Studies (post-main)

| Study | Adds |
|---|---|
| Cross-domain transfer (LoCoMo / LongMemEval / STEM-Bench) | 3 text-dialogue benchmarks |
| SWE-Bench-Lite | Real code tasks |
| 50 real-user IDE replays | Out-of-distribution leak check |
| Information-theoretic upper bound | Theoretical analysis |
| Adversarial injection | Robustness test |