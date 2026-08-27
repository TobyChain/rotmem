# RotMem — Research Proposal

**Title.** RotMem: A 5–20MB Residual-Rotation Memory Buffer for Single-Session Long-Horizon LLM Agents

**Date.** 2026-08-27

**Status.** Round 12 (post 8 reflection rounds; final iteration before paper-writing).

### Updated positioning sentence (Round 12)

**RotMem sits at the intersection of three design axes that no prior work covers simultaneously:**

1. **Deterministic vs learned**: RotMem = **Oblivion-minus-the-controller** (arXiv:2604.00131) = **SmartSearch-minus-the-CrossEncoder** (arXiv:2603.15599) = **LCM-minus-the-LLM-Map** (arXiv:2605.04050) = **MemSifter-minus-the-proxy-LM** (arXiv:2603.03379). The consistent thesis: *all four prior works require a learned component; RotMem replaces it with deterministic information-theoretic priors.*
2. **Flat vs recursive**: RotMem is **non-recursive**; LCM (arXiv:2605.04050) and RLM (arXiv:2603.02615) are recursive. Recursion depth 2+ "overthinks" (2603.02615); flat is better at our scale.
3. **Cosine-preserving vs summary**: RotMem preserves **every item's cosine identity** via lazy orthogonal projection; LCM uses lossy hierarchical summary DAG; RLM uses plain-text summaries; MemOPD compresses inputs/outputs. The cosine-preserving axis is *strictly more information-preserving* than any summary-based mechanism at fixed memory budget.

### Three-axis positioning table (Round 12)

| Axis | RotMem | Closest competitor | Δ |
|---|---|---|---|
| Learned vs deterministic | Deterministic | LCM (deterministic but uses LLM-Map); Oblivion (LLM controller); SmartSearch (CrossEncoder); MemSifter (proxy LM) | No second LM, zero gradients |
| Recursive vs flat | Flat single-session buffer | LCM (hierarchical summary DAG); RLM (recursive); MemGym (use as benchmark, not competitor) | One bounded buffer, no recursion depth |
| Cosine-preserving vs summary | Cosine-preserving (lazy V_t) | LCM (lossy summary); MemOPD (lossy input compression); Mem0 (extractive facts) | No summary ever; identity retained |

### Map of competencies (from LoCoBench-Agent arXiv:2507.05257)

| Competency | Mechanism in RotMem |
|---|---|
| Accurate retrieval | strength-weighted cosine top-k |
| Test-time learning | strength update on each write |
| Long-range understanding | lazy orthogonal V_t projecting past keys |
| Selective forgetting | exponential decay + merge-on-overflow |

---
## 1. Short Hypothesis

If the memory state of a single-session LLM agent is maintained as a
continuously-rotated residual buffer

```
M_t = orthogonal_project(M_{t-1}, V_t) + encode(x_t)
```

— inspired by Residual Memory Networks (RMN) — instead of being
rebuilt by periodic full-block compaction, the agent retains strictly
more past-turn information at fixed memory budget, eliminates the
*memory cliff* information loss, and yields **≥20% task-F1 lift on a
500-turn single-session coding benchmark** vs. fixed-rule compactors.

The implementation is a **deterministic math buffer (5–20MB) with no
trained controller and no second LM**.

---

## 2. Core Mechanism

| Operation | Math | Cost |
|---|---|---|
| Update | `M_t = QR-project(M_{t-1}, V_t) ⊕ encode(x_t)` | O(d³) for QR, d=256 → <1 ms |
| Decay | `s_i ← s_i · exp(-Δt / τ)` | O(N) |
| Consolidate | if `|M| > B`, merge the two items with `cosine > τ_sim` by weighted average | O(N²) worst, O(N) amortised |
| Query | top-k by `cosine(k_i, q) · s_i` | O(N·d) |

The orthogonal rotation `V_t` is refreshed every `K` turns or when its
spectral radius drifts by > 0.01 from 1.

---

## 3. Test Surfaces

### 3.1 Single-session CLI coding (main)

- **Harness.** `deepseek-harness/apps/cli` with a RotMem `Compactor`
  adapter plugged into `packages/compaction/`.
- **Turns.** 500 turns per session.
- **Subtasks.** Seven hidden long-horizon coding tasks: refactor, TDD
  fix, multi-file migration, type-error whack-a-mole, dependency
  upgrade, bug hunt, doc-coverage sweep.

### 3.2 Cross-domain text dialogue (extension)

- LoCoMo (multi-session QA recall, used within-session subset)
- LongMemEval
- STEM-Bench

### 3.3 Long-context baseline

- Needle-in-a-Haystack at 30k–50k token range (proves the buffer
  doesn't degrade as the agent writes more).

### 3.4 Real-world coding

- SWE-Bench-Lite (100 instances, Qwen3-2B as backbone).

---

## 4. Backbones × Compactors

| Backbone | Why |
|---|---|
| Qwen3-2B | Default; demonstrates consumer-grade deployability. |
| Qwen3-9B | Medium; the lab already owns the weights. |
| MiniMax-M3 | Oracle; same interface, no GPU. |

| Compactor | Description |
|---|---|
| `compaction-basic` | `deepseek-harness/packages/compaction/compaction-basic` rule-based sliding window. |
| Mem0-style | Extractive summarisation every 50 turns. |
| A-MemGuard-style | Write-time summary. |
| RotMem | Ours. |

---

## 5. Metrics

| Metric | Definition |
|---|---|
| Memory-Cliff AUC | F1 drop between turn 100 and turn 500 across hidden test items. |
| Retrieval-Precision@5 | P(top-5 retrievals are useful for the next turn). |
| Long-Tail-Recall | Recall of facts stated at turn ≤50 when queried at turn ≥400. |
| Compaction-Information-Loss | Fraction of past-turn atomic facts unretrievable after the buffer cap is hit. |
| Wall-clock per turn | ms |
| Memory footprint | bytes resident at 5k turns |

---

## 6. Ablation Matrix

| # | Ablation | Expected Effect |
|---|---|---|
| A1 | Drop orthogonal rotation, pure append+evict | Tests residual-rotation contribution. |
| A2 | Drop strength decay, fix s=1 | Tests time-decay information value. |
| A3 | Replace weighted-merge with FIFO eviction | Tests merge vs evict. |
| A4 | Rotation period {10, 50, 200, never} | Sweeps rotation frequency. |
| A5 | Drop strength weighting, cosine-only retrieval | Tests reinforcement signal. |
| A6 | Buffer size cap {50%, 100%, 200%} | Stress-tests budget. |
| A7 | QR vs Gram-Schmidt vs Householder | Orthogonalisation algorithm invariance. |
| A8 | Horizon {100, 500, 1000} turns | Long-range scaling. |

Plus an **ablation cube** of {rotation on/off} × {decay on/off} × {merge
on/off} → 8 cells, to expose interaction effects.

---

## 7. Statistical Discipline

- 3 seeds × 2 dataset splits.
- Paired bootstrap 95% CI on every reported number.
- Holm-Bonferroni over all baseline comparisons.
- Cohen's d for headline effects.
- Per-task breakdown for the 7 CLI subtasks.
- 50-case failure-mode study on worst instances.

**Pre-registered null threshold:** if RotMem − best-baseline < 1 pt on
Memory-Cliff AUC, the contribution is null and we report the
experimental protocol + the ablation cube honestly.

---

## 8. Robustness Extensions

| Test | Purpose |
|---|---|
| Adversarial injection (50 cases) | Robustness à la A-MemGuard. |
| Backbone swap at turn 250 | Verify LM-agnostic interface. |
| 50 real-user IDE replays | Out-of-distribution leak check. |
| Information-theoretic upper-bound proof | Bounded retention analysis. |
| Memory-Cliff curve fitting | F1-vs-turn shape analysis. |

---

## 9. Iteration Plan

### Stage 1 — Unit validation (week 1, no GPU)

- Unit tests: `ρ = 1` invariant, merge non-informativeness-loss, query
  ordering, decay monotonic.
- Tiny needle-in-haystack at 1k–10k turns.
- Hyperparameter sweep: τ, rotation period, merge threshold.

### Stage 2 — Main experiments (week 2–3, 3× RTX 3090)

- 500-turn CLI coding × 3 backbones × 4 compactors × 3 seeds.
- Collect primary metrics + Memory-Cliff curves.
- Run all 8 ablations + the ablation cube.

### Stage 3 — Cross-domain + robustness (week 4)

- LoCoMo / LongMemEval / STEM-Bench reruns.
- SWE-Bench-Lite on Qwen3-2B (consumer hardware).
- 50 real-user IDE replays.
- Writing.

---

## 10. Compute Budget

- 3× RTX 3090 for ≤4 days (LLM rollouts only).
- **No teacher**, no MiniMax-M3 quota consumed.
- Zero GPU memory for RotMem itself.

---

## 11. Risks

| Risk | Mitigation |
|---|---|
| Rotation drift in finite precision | QR re-orthogonalise every K turns. |
| No learned policy = no per-task adaptivity | Future work; clearly labelled as extension. |
| Synthetic tasks over-fit | Real-user replay leakage check. |
| Main result < 1 pt → null | Report ablation cube + theory anyway; workshop-track. |
| Backbone mismatch | Define LM-agnostic interface; backbone only sees embeddings. |
| Adversarial injection | Adversarial test, design-deterministic-policy defence. |