# RotMem — Stage 2 Feasibility Analysis (Round 38)

This document translates the **GPU scheduling spec** (`/data4/guanbingtao/spec/all-in-tcm-spec.md` and `train-spec.md`) into concrete constraints for RotMem's empirical validation.

---

## 1. GPU Constraints (From Spec, Verified 2026-08-27)

### 1.1 Hardware Inventory

| Host | Hostname | Address | RTX 3090 count | Notes |
|---|---|---|---:|---|
| 240 | amax-new | 192.168.0.240 | 4 | priority 1 |
| 243 | amax-xp | 192.168.0.243 | 6 | priority 2 |
| 248 | sm-248 | 192.168.0.248 | 8 | priority 3 (this machine) |

- **Single-machine budget**: max 3× RTX 3090 24G per host, no cross-host DDP.
- **Reserved UUIDs** (8 per host, registered with `nvidia-smi --query-gpu=index,name,uuid`).
- SSH: `ssh -p 18518 guanbingtao@192.168.0.<host>`, shared `/data4/guanbingtao`.

### 1.2 Idle Conditions (Hard)

A GPU is launch-eligible **only** when ALL of:

1. name == `RTX 3090` (2080 Ti excluded)
2. `nvidia-smi --query-compute-apps` empty
3. `memory.used <= 512 MiB`
4. `utilization.gpu <= 5%`
5. **TWO consecutive samples ≥ 5 seconds apart**

Hard rule: `utilization.gpu > 40%` **never** usable, regardless of memory.

### 1.3 Live Status (verified 2026-08-27 via SSH)

**Host 248 (sm-248)** — this machine:
- All 8 GPUs occupied (utils 0-100%, memory 14-20 GB)
- GPU 7 has util=0% but memories not yet captured → not yet launch-eligible
- Last idle-confirmation window not met

**Host 243 (amax-xp)** — via SSH:
- GPU 0: util 64% → busy
- GPU 2/3/4/5: 14-17 GB used → busy

**Host 240 (amax-new)** — via SSH:
- GPU 2: util 50% → busy
- GPU 3: 9.4 GB used → busy

**Conclusion**: As of 2026-08-27 17:30, **no host currently has 3 launch-eligible GPUs**. All three are in active use by other tasks.

---

## 2. RotMem Stage 2 — Revised Feasibility

The spec's constraints force a **much narrower Stage 2 than my original 32-round proposal** assumed. Revised plan:

### 2.1 What is feasible under the spec

| Activity | GPU usage | Spec-feasible? |
|---|---|---|
| Stage 1 unit tests (14 already passing) | 0 GPU | ✓ done |
| Mini needle-in-haystack (1k–10k turns, dim=64) | 0 GPU (numpy only) | ✓ can run immediately |
| Synthetic 500-turn memory-cliff (numpy only) | 0 GPU | ✓ can run immediately |
| **Mosaic's exact protocol**: LLM-as-judge inference (Qwen3-2B / Qwen3-9B / MiniMax-M3 × 4 judges) | 1-2 GPU × <30min per cell | ✓ once any host is idle |
| Compactor head-to-head: RotMem vs Mosaic vs compaction-basic × MemGym-CODEQA subset | 1-2 GPU × <60min | ✓ once any host is idle |
| A1–A8 ablation cube (8 cells × 1 backbone × LoCoMo subset) | 1 GPU × ~30min | ✓ once any host is idle |

### 2.2 What is **not** feasible under the spec in Stage 2

| Activity | Why not feasible |
|---|---|
| Train any model (SFT/RL/OPD) | Out of scope; my work uses no trained controller. |
| GRPO/PPO with teacher model | Requires sustained 3-GPU occupancy; spec mandates serial stages and idle confirmation. |
| 7 compactors × 3 backbones × 5 benchmarks × 3 seeds | Total ~315 condition cells; infeasible within one-host idle window. |

### 2.3 Revised Stage 2 Plan (Spec-Compliant)

**Phase A — Zero-GPU validation (run immediately, no host needed)**:

1. Re-run `tests/test_core.py` to confirm 14/14 pass.
2. Run `python3 benchmarks/synthetic_memory_cliff.py`:
   - 1k, 5k, 10k, 50k turn sessions, dim=64
   - Per-session: Memory-Cliff AUC, Retrieval-Precision@5, Wall-clock
   - Head-to-head: RotMem vs cosine-only, vs FIFO eviction, vs sliding-window
3. Run `python3 benchmarks/theoretical_metrics.py`:
   - Cosine-preservation rate over 100k random (key, query) pairs → expect 100%
   - Spectral-radius drift over 500 turns, every seed → expect <1e-3
4. Publish numbers → enables Stage 2 budget estimation.

**Phase B — Single-GPU LLM evaluation (when any host becomes idle)**:

5. Run **Mosaic's exact 4-model × 4-judge protocol on LoCoMo subset**:
   - Backbones: Qwen3-2B (1 GPU), Qwen3-9B (2 GPU via tp=1)
   - Judges: Qwen3-2B + MiniMax-M3 (1 GPU each)
   - Compactors: compaction-basic, Mosaic, RotMem
   - Sessions: 10 (LoCoMo subset) × 3 seeds = 30 sessions
   - Wall-clock: estimated 2-4 hours total
6. Run **Memory-Cliff AUC** on:
   - compactor ∈ {compaction-basic, Mosaic, RotMem}
   - backbone ∈ {Qwen3-2B, Qwen3-9B} (subset if budget tight)
   - horizon ∈ {100, 500, 1000} turns
   - 3 seeds
   - Wall-clock: estimated 4-8 hours

**Phase C — Ablation cube (when any host becomes idle)**:

7. Run ablation cube {rotation × decay × merge} × on/off (8 cells):
   - 1 backbone (Qwen3-2B), 1 horizon (500), 3 seeds
   - Wall-clock: estimated 2-4 hours

**Phase D — Cross-domain validation (when budget allows)**:

8. MemGym-CODEQA subset × 3 compactors × 3 seeds
9. MemoryArena subset × 3 compactors × 3 seeds

**Phase E — Security evaluation (separate sessions, lightweight)**:

10. MCFA + MemCollusion + indirect-injection tests:
    - 50 cases × 3 compactors
    - These are *agent-level* attacks against a Qwen3-2B agent
    - Estimated 4-8 hours
11. Carbon-savings measurement:
    - Run RotMem vs LCM vs Mosaic on identical traces, measure wall-clock + token-cost
    - Carbon estimate via LLMCarbon (arXiv:2511.08575) methodology

### 2.4 Total Estimated Compute Budget (Spec-Compliant)

| Phase | GPU-hours | Host-hours |
|---|---:|---:|
| Phase A | 0 | 0 (any host) |
| Phase B | ~12 GPU-h | | ~6 host-h on |
| | | one |
| Phase C | | ~6 GPU-h | | ~6 host-h |
| Phase D | | ~12 GPU-h | | ~6 host-h |
| Phase E | | ~12 GPU-h | | ~6 host-h |
| **Total** | | **~42 GPU-hours** | | **~24 host-hours** |

At 24 host-hours, this is **~3 days of single-host occupation**. Under the spec's serial-stage rule, this is feasible in **~5-7 calendar days** of opportunistic scheduling.

---

## 3. Zero-GPU Validation Scripts (Phase A — run now)

### 3.1 Synthetic Memory-Cliff (no GPU needed)

```python
# benchmarks/synthetic_memory_cliff.py
# Run: python3 benchmarks/synthetic_memory_cliff.py

import sys, os, time, json, math
import numpy as np
sys.path.insert(0, 'src')
from rotmem import RotMem

def run_session(N_turns, d, mem_cap, seed):
    rng = np.random.default_rng(seed)
    keys = rng.standard_normal((N_turns, d)).astype(np.float32)
    keys /= np.linalg.norm(keys, axis=1, keepdims=True) + 1e-12
    mem = RotMem(dim=d, max_items=mem_cap, decay_tau=50.0, retrieval_k=5)
    t0 = time.time()
    for t in range(N_turns):
        mem.update(f"id_{t}", keys[t], value=t, now=t)
    elapsed = time.time() - t0
    # Memory-Cliff: query early keys, measure recall@20
    recall_at_20 = []
    for q_idx in [0, 50, 100, 200]:
        if q_idx >= N_turns: continue
        hits = mem.query(keys[q_idx], top_k=20)
        ids = [h.item.item_id for h in hits]
        # We want q_idx to be in the top-20
        recall_at_20.append(1.0 if f"id_{q_idx}" in ids else 0.0)
    return {
        "N_turns": N_turns,
        "memory_cliff_recall@20": float(np.mean(recall_at_20)),
        "wallclock_s": elapsed,
        "turns_per_sec": N_turns / elapsed,
        "buffer_size": mem.size,
    }

if __name__ == "__main__":
    results = []
    for N in [1000, 5000, 10000, 50000]:
        for cap in [5000, 10000]:
            for seed in [0, 1, 2]:
                r = run_session(N, d=64, mem_cap=cap, seed=seed)
                results.append(r)
    out = "/data4/guanbingtao/rotmem/output/synthetic_memory_cliff.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {len(results)} rows to {out}")
```

### 3.2 Theoretical Metrics (no GPU needed)

```python
# benchmarks/theoretical_metrics.py
import sys, os, json
import numpy as np
sys.path.insert(0, 'src')
from rotmem import RotMem

def cosine_preservation_test(n_pairs=10000, d=64, seed=42):
    rng = np.random.default_rng(seed)
    correct = 0
    max_err = 0.0
    mem = RotMem(dim=d, max_items=100, rotation_period=10)
    for i in range(50):
        mem.update(f"id_{i}", _unit(rng, d), now=i)
    for _ in range(n_pairs):
        k = _unit(rng, d)
        q = _unit(rng, d)
        raw = float(k @ q / (np.linalg.norm(k) * np.linalg.norm(q)))
        # Project
        V = mem._current_basis()
        pk = V @ k
        pq = V @ q
        proj = float(pk @ pq / (np.linalg.norm(pk) * np.linalg.norm(pq)))
        err = abs(raw - proj)
        max_err = max(max_err, err)
        if err < 1e-6:
            correct += 1
    return {
        "n_pairs": n_pairs,
        "correct_within_1e-6": correct,
        "rate": correct / n_pairs,
        "max_error": max_err,
    }

def spectral_drift_test(d=64, n_turns=500, seed=0):
    rng = np.random.default_rng(seed)
    mem = RotMem(dim=d, max_items=2000, rotation_period=10)
    drifts = []
    for t in range(n_turns):
        mem.update(f"id_{t}", _unit(rng, d), now=t)
        V = mem._current_basis()
        err = float(np.linalg.norm(V @ V.T - np.eye(d)))
        drifts.append(err)
    return {
        "n_turns": n_turns,
        "max_drift": float(np.max(drifts)),
        "mean_drift": float(np.mean(drifts)),
        "drift_below_1e-3": bool(np.max(drifts) < 1e-3),
    }

def _unit(rng, d):
    v = rng.standard_normal(d).astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-12)

if __name__ == "__main__":
    results = {
        "cosine_preservation": cosine_preservation_test(),
        "spectral_drift": spectral_drift_test(),
    }
    out = "/data4/guanbingtao/rotmem/output/theoretical_metrics.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))
```

### 3.3 Expected Outputs

- `output/synthetic_memory_cliff.json`: 24 rows (4 horizons × 2 caps × 3 seeds)
  - Predicted: `turns_per_sec > 1000` (numpy ops are fast)
  - Predicted: `memory_cliff_recall@20` close to 1.0 for N ≤ 5000
- `output/theoretical_metrics.json`:
  - `cosine_preservation.rate = 1.0`
  - `spectral_drift.max_drift < 1e-3`

These two files are the **first concrete empirical deliverables** for RotMem and are **runnable today** with no GPU.

---

## 4. GPU Phase B — When Idle Confirmation Triggers

When the spec's idle confirmation triggers on any host (util <5%, mem <512MiB, two consecutive samples ≥5s apart):

1. SSH to the idle host (248 first since priority order is 248, 243, 240 — wait, spec says order is [240, 243, 248]; priority 1 is 240).
2. Verify: `nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader`
3. If ≥3 cards are eligible: bind via `CUDA_VISIBLE_DEVICES=GPU-<uuid1>,GPU-<uuid2>,GPU-<uuid3>`.
4. Run `python3 benchmarks/phase_b_mosaic_protocol.py`.

`benchmarks/phase_b_mosaic_protocol.py` (to be written):

```python
# Mosaic's exact protocol on LoCoMo subset
# Backbones: Qwen3-2B (single-GPU), Qwen3-9B (tp=1)
# Judges: 4 LLMs (Qwen3-2B + Qwen3-9B + MiniMax-M3 + Llama-3-8B via MiniMax proxy if budget)
# Compactors: compaction-basic, Mosaic, RotMem
# Sessions: 10 LoCoMo × 3 seeds = 30 sessions
# Metrics: paired bootstrap, Holm-Bonferroni across 3 compactors
```

This script should follow Mosaic's published protocol exactly (4 backbones, 4 LLM judges, paired bootstrap). It can run on **1 GPU** (Qwen3-2B as both backbone and one judge).

---

## 5. Honest Assessment Under the Spec

Under the strict GPU constraints of `/data4/guanbingtao/spec/`:

- **All 32-round proposed Stage 2 work** is **not** feasible in one continuous run.
- **Phase A (zero-GPU)** is feasible *today* and gives the **first
  concrete empirical numbers** for the paper.
- **Phase B–E** require **24 host-hours** of single-machine occupation
  across the spec's idle-window detection.

This is **actually favourable** for a top-venue submission:
- The work is now sized to what the lab can realistically run.
- The headline result (Mosaic-protocol head-to-head) requires only
  *1 GPU × 2-4 hours* per cell × 9 cells = **18-36 GPU-hours** of
  *LLM inference* — feasible in **1 week of opportunistic
  scheduling**.
- The null-result risk is unchanged (theoretical bounds still hold
  either way), but the path to *publishable* result is now concrete.

---

## 6. The Next Action (Concrete)

**Tonight / today (no GPU needed)**:
1. Write `benchmarks/synthetic_memory_cliff.py` (script in §3.1).
2. Write `benchmarks/theoretical_metrics.py` (script in §3.2).
3. Run both; produce `output/synthetic_memory_cliff.json` and
   `output/theoretical_metrics.json`.
4. Commit to repo; this gives **Stage 2 Phase A results**.

**When a host becomes idle** (any of 240, 243, 248):
5. Write `benchmarks/phase_b_mosaic_protocol.py`.
6. Bind via `CUDA_VISIBLE_DEVICES` (UUID-locked per spec).
8. Run on LoCoMo subset, log wall-clock + per-cell F1 + bootstrap CI.
9. Commit results; this gives **headline result** for the paper.

This produces a **complete, publishable top-venue paper** with:
- 4 theorems + Stage 2 Phase A empirical numbers
- A concrete head-to-head with Mosaic (Phase B)
- The ablation cube (Phase C)
- Carbon-savings number (Phase E)

**Without** any teacher training, GRPO, SFT, or any sustained GPU occupation.