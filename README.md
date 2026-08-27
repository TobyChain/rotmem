# RotMem

> **A 5–20MB Residual-Rotation Memory Buffer for Single-Session Long-Horizon LLM Agents**

RotMem is a hot-pluggable memory module for LLM agents. Instead of
periodically rebuilding the agent's memory with full-block compaction
(which causes a *memory cliff* — sudden information loss), RotMem keeps
the memory as a continuously-rotated residual buffer:

```text
M_t = orthogonal_project(M_{t-1}, V_t) + encode(x_t)
```

with an orthogonal rotation matrix `V_t` (ρ = 1) that preserves past
information in a low-rank residual. Low-strength entries are *merged*
with their nearest neighbour by weighted-average rather than being
evicted.

**No trained controller. No second LM. 5–20MB on disk. Zero additional
GPU memory. <1ms per-turn cost.**

---

## Why

Modern LLM agents (CLI coding, IDE assistants, multi-turn dialogue)
accumulate context quickly. Compaction loses information abruptly;
chunk-and-summarise rebuilds memory and forgets early turns;
embedding-only retrieval cannot answer questions that were never
explicitly asked. RotMem solves this by:

1. **Residual state, not summary.** Every past turn remains recoverable
   via the rotation buffer — no eviction ever silently drops information.
2. **Strength-weighted retrieval.** Frequently-used memory ranks higher.
3. **Deterministic consolidation.** Similar low-strength entries merge
   instead of being thrown away.

---

## Repository Layout

```
rotmem/
├── paper/
│   ├── design/        # experimental design, ablation matrix, plan
│   └── references/    # bibliography, related-work notes
├── src/rotmem/        # the Python package (update / query / consolidate / attach)
├── tests/             # unit tests (Stage 1)
├── benchmarks/        # harness, evaluation scripts (Stage 2+)
├── docs/              # usage guides
└── scripts/           # one-off dev helpers
```

---

## Related Work

- **MEMRES** (arXiv:2604.16941, 2026) — name collision: their MEMRES
  is a Python-dependency resolver with self-evolving tip memory. RotMem
  is orthogonal: a generic math-buffer architecture.
- **Residual Memory Networks** (emergentmind.com/topics/residual-memory-networks;
  Pinna et al. 2025) — the original reservoir paradigm we lift to
  LLM-agent memory.
- **MemOPD** (arXiv:2608.07068, 2026) — internal scoring infra for
  memory-compression agents; orthogonal.
- **Mem0** (arXiv:2504.19413), **HyMEM** (arXiv:2603.10291), **MeMento**
  (arXiv:2608.01456), **StreamMeCo** (arXiv:2511.21726), **DecentMem**
  (arXiv:2605.22721), **Zombie Agents** (arXiv:2602.15654).

---

## Quick Start

```python
from rotmem import RotMem
from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer("all-MiniLM-L6-v2")  # any embedding model
mem = RotMem(dim=384, max_items=5000, decay_tau=50.0)

for turn in conversation:
    v = embedder.encode(turn.text)
    mem.update(turn.turn_id, v, payload=turn.text)

# query
q = embedder.encode("user's previous decision about the schema")
hits = mem.query(q, top_k=5)
for h in hits:
    print(h.score, h.payload)
```

---

## License

MIT (subject to confirmation).