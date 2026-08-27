# RotMem — Ablation & Iteration Plan

## 1. Eight-Point Ablation Matrix

Each ablation removes or varies **one component at a time**.

| ID | Ablation | What it isolates |
|---|---|---|
| A1 | Drop orthogonal rotation, pure append+evict | Does rotation actually preserve information? |
| A2 | Drop strength decay (`s_i` fixed) | Does time-decay help retrieval ranking? |
| A3 | Replace weighted-merge with FIFO eviction | Does merging beat dropping? |
| A4 | Rotation period {10, 50, 200, never} | What's the optimal refresh frequency? |
| A5 | Drop strength weighting, cosine-only | Is strength * necessary, or only convenient? |
| A6 | Buffer cap {0.5×, 1×, 2× default} | Where does the budget wall hit? |
| A7 | QR vs Gram-Schmidt vs Householder | Is the algorithm choice consequential? |
| A8 | Horizon {100, 500, 1000} turns | Long-range scaling curve |

**Reporting format:** for each ablation, two paired-bar plots:
- Headline metric (Memory-Cliff AUC) for the variant vs. the full
  RotMem.
- Per-subtask breakdown to localise the gain.

---

## 2. The Ablation Cube (interaction effects)

Full 2 × 2 × 2 = 8 cells:

| | rotation ON | rotation OFF |
|---|---|---|
| merge ON, decay ON | **Full RotMem** | drop rotation |
| merge ON, decay OFF | drop decay | drop rotation+decay |
| merge OFF, decay ON | drop merge | drop rotation+merge |
| merge OFF, decay OFF | FIFO+decay | pure FIFO |

This exposes **second-order interactions** that the 1D ablations
cannot. Reported as a 3D heatmap of `Δ_F1(100→500)`.

---

## 3. Iteration Loop

```text
┌──────────────────────────────────────────────────┐
│                                                  │
│   Stage 1 (no GPU, week 1)                       │
│   ├─ unit tests                                  │
│   ├─ 1k-turn needle-in-haystack                  │
│   └─ hyperparameter sweep                        │
│            │                                     │
│            ▼                                     │
│   Stage 2 (3×3090, week 2–3)                     │
│   ├─ 500-turn CLI × 3 backbones × 4 compactors   │
│   ├─ 8 ablations + cube                          │
│   └─ primary-metric dashboard                    │
│            │                                     │
│            ▼                                     │
│   Stage 3 (week 4)                               │
│   ├─ cross-domain transfer                       │
│   ├─ SWE-Bench-Lite                              │
│   ├─ real-user replays                           │
│   └─ paper writing                               │
│            │                                     │
│            ▼                                     │
│   Iteration trigger:                             │
│   if headline < null threshold:                  │
│       re-examine Stage 1 hyperparameters          │
│       check unit tests for invariant violations  │
│       if still null → submit as workshop note    │
│                                                  │
└──────────────────────────────────────────────────┘
```

---

## 4. Strengthening the Headline Result

Even before a clear win, the result can be **strengthened** by:

| Lever | What it buys |
|---|---|
| Larger session seeds (100 → 500 sessions) | Tighter CIs, smaller p-values |
| Harder subtasks (add a security-9-bench) | Wider gap between compactors |
| Adversarial test as primary metric | Demonstrates robustness, not just average-case |
| Ablation cube's interaction effects | Shows our design choices compose, not just exist in isolation |
| Information-theoretic upper-bound | Justifies why further optimisation is bounded |
| Real-user replay as a separate primary metric | Bridges the synthetic-vs-real divide |

---

## 5. Failure-Mode Iteration

After Stage 2, the failure-mode analysis (see `experiment_design.md` §6)
feeds back into Stage 1:

- If "Rotation drift" is a frequent failure, tighten the re-orthog
  threshold in Stage 1.
- If "Buffer overflow" is frequent, increase the default cap.
- If "Backbone reasoning failure" dominates, swap to a stronger
  backbone for Stage 3.

---

## 6. Negative-Result Discipline

If after Stage 3 the headline is still < 1 pt:

- Publish the **ablation cube + unit-test invariants + theory** as a
  workshop note.
- Frame the contribution as: "RotMem is a deterministic,
  zero-cost, drop-in alternative to learned compaction, with formally
  analysable behaviour, even if it does not always win on a single
  benchmark."
- Submit to a workshop venue (e.g. ICML Workshop on Long Context,
  NeurIPS Workshop on Memory).