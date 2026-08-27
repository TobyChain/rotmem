# RotMem — Integration Plan (Round 12 addition)

This document specifies how RotMem is **shipped as a drop-in Compactor**
in existing agent harnesses. It is the **practical deployment**
contribution that demonstrates RotMem is not just an academic
proposal.

## 1. The integration target

`deepseek-harness` is a public TypeScript CLI harness (see
`/data4/guanbingtao/deepseek-harness/`) that already has four
compactors shipped in `packages/compaction/`:

- `compaction-basic/` — sliding-window rule
- `command-compact/` — command-aware compaction
- `compaction/` — generic compaction interface
- `compaction-tool-result-pruner/` — tool-result-specific compaction

RotMem is a **6th compactor** (`packages/compaction/rotmem/`) that
implements the same `Compactor` interface, so it can be enabled with
zero changes to the agent harness.

## 2. Adapter contract

```typescript
// packages/compaction/rotmem/src/index.ts
import { Compactor, Compactions } from "../compaction";

export class RotMemCompactor extends Compactor {
  // called once per turn
  update(item_id: string, embedding: number[], value: any, now?: number): void;
  // called once per query
  query(embedding: number[], top_k?: number): Array<{item_id, score, value}>;
  // called when buffer size exceeds cap
  consolidate(): void;
  // optional hooks
  attach(agent: Agent): void;
  // size introspection
  size(): number;
  // theoretical monitoring
  spectral_drift(): number;     // |ρ(V_t) - 1|
  cosine_preservation_rate(): number;  // 1.0 expected
}
```

## 3. Server vs local

For consumer-grade deployment:

| Component | Size | Where it runs |
|---|---:|---|
| RotMem buffer (Python or TS) | 5–20 MB | client (CPU) |
| Embedding model (bge-small) | 33 MB | client (CPU) |
| Backbone LLM (Qwen3-2B INT4) | 1.5 GB | client (CPU/MPS/CUDA) |

**No GPU required for the buffer.** Only the backbone LLM needs
acceleration, and even Qwen3-2B INT4 runs on Apple Silicon M2 at ~30
tokens/sec.

## 4. HyMeS-style integration (arXiv:2608.09410)

HyMeS uses a **coding agent for high-level memory management**,
offloading low-level skills to a vision-language-action model. We
adopt the same architectural pattern:

```
┌─────────────────────────────────┐
│  deepseek-harness CLI           │  ← existing, no changes
│  + RotMem Compactor adapter     │
└────────────────┬────────────────┘
                 │ attach()
                 ▼
┌─────────────────────────────────┐
│  RotMem buffer (5–20MB, CPU)    │  ← new
│  + strength decay + V_t basis   │
└────────────────┬────────────────┘
                 │ query(top-k)
                 ▼
┌─────────────────────────────────┐
│  Qwen3-2B / Qwen3-9B backbone   │  ← unchanged
└─────────────────────────────────┘
```

The backbone never touches RotMem directly; the harness injects
top-k retrieved memories as part of the prompt context.

## 5. Claude Code integration target (for future work)

Claude Code's compaction primitives (per arXiv:2604.14228) follow a
similar pattern: a configurable compaction strategy sits between the
LLM call and the conversation buffer. RotMem can be packaged as a
**Claude Code plugin** (AGENTS.md + custom compactor) following the
**AGENTS.md** configuration convention documented in arXiv:2602.14690.

This is not the primary integration target — `deepseek-harness`
already exists in this repo — but it is the path to **broader
adoption**.

## 6. Testing the integration

After the integration is wired:

1. **Unit-level**: existing tests in `tests/test_core.py` continue
   to pass.
2. **Compactor-level**: a new test verifies that the RotMem
   Compactor implements the interface correctly.
3. **Harness-level**: a new test runs `deepseek-harness` with each
   of the 6 compactors on a 50-turn CLI task and compares F1.
4. **End-to-end**: a 500-turn session on a real coding task,
   comparing all 6 compactors on the same session seed.

## 7. Risk & mitigations

| Risk | Mitigation |
|---|---|
| TypeScript↔Python impedance (deepseek-harness is TS, RotMem core is Python) | Ship both: TS adapter that calls Python subprocess via IPC, OR port core to TS (3-day effort) |
| Embedding model cost (33MB bge-small) | Use server-side shared embedding; client-side fallback to TF-IDF |
| Backbone latency dominates | Already the bottleneck; RotMem does not change this |
| Memory buffer corruption across sessions | Per-session process; no persistence to disk |

## 8. The "Oh-My-Pi" positioning

The user's lab mentions `oh-my-pi` as a CLI agent target. RotMem is
designed to integrate with *any* CLI agent harness, including
`oh-my-pi`, by exposing the simple four-method API. The user's
specific integration choices (e.g., which embedding model, which
buffer size, which compaction threshold) are determined empirically
via the Stage 2 ablation matrix.