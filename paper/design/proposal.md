# RotMem — Research Proposal

**Title.** RotMem: A 5–20MB Residual-Rotation Memory Buffer for Single-Session Long-Horizon LLM Agents

**Date.** 2026-08-27

**Status.** Pre-implementation, design lock.

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