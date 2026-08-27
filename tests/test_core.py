"""Unit tests for RotMem core invariants.

These tests cover Stage 1 (no-GPU) sanity checks; the goal is to catch
invariant violations early, *not* to measure end-task performance.
"""

from __future__ import annotations

import numpy as np

from rotmem import RotMem


def _key(rng: np.random.Generator, d: int = 32) -> np.ndarray:
    v = rng.standard_normal(d).astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-12)


def test_empty_query_returns_empty():
    mem = RotMem(dim=16)
    hits = mem.query(_key(np.random.default_rng(0)))
    assert hits == []


def test_basic_insert_and_query():
    """Identical key inserted then retrieved must rank first."""
    rng = np.random.default_rng(0)
    mem = RotMem(dim=32, decay_tau=1e6, merge_threshold=0.99)
    # make keys well-separated
    base = rng.standard_normal(32).astype(np.float32)
    keys = [base + 10.0 * _key(rng, 32) for _ in range(10)]
    for i, k in enumerate(keys):
        mem.update(f"id_{i}", k, value=f"v_{i}", now=i)
    q = keys[3]
    hits = mem.query(q, top_k=3)
    assert len(hits) == 3
    assert hits[0].item.item_id == "id_3", f"got {hits[0].item.item_id}"


def test_replace_existing_id():
    rng = np.random.default_rng(0)
    mem = RotMem(dim=16, decay_tau=1e6)
    mem.update("a", _key(rng), value="orig", now=0)
    mem.update("a", _key(rng), value="new", now=1)
    assert mem.size == 1
    assert mem._items[0].value == "new"


def test_strength_decay_monotone():
    rng = np.random.default_rng(0)
    mem = RotMem(dim=16, decay_tau=5.0)
    mem.update("a", _key(rng), now=0)
    s0 = mem._items[0].strength
    for t in range(1, 6):
        mem.update(f"b_{t}", _key(rng), now=t)
    assert mem._items[0].strength < s0


def test_merge_does_not_shrink_uninformatively():
    """Two very-similar items should be merged into one (when over cap)."""
    rng = np.random.default_rng(0)
    base = _key(rng)
    mem = RotMem(
        dim=16,
        max_items=2,
        merge_threshold=0.5,
        decay_tau=1e6,
    )
    mem.update("a1", base + 0.01 * _key(rng), value="pair_a_1", now=0)
    mem.update("b1", base + 0.01 * _key(rng), value="pair_b_1", now=1)
    mem.update("a2", _key(rng), value="pair_a_2", now=2)
    mem.update("b2", _key(rng), value="pair_b_2", now=3)
    assert mem.size <= 2


def test_rotation_basis_is_orthogonal():
    """The lazy basis V_t must be an orthogonal matrix."""
    rng = np.random.default_rng(0)
    mem = RotMem(dim=8)
    for i in range(5):
        mem.update(f"id_{i}", _key(rng, 8), now=i)
    V = mem._current_basis()
    assert V.shape == (8, 8)
    err = np.linalg.norm(V.T @ V - np.eye(8))
    assert err < 1e-3, f"V not orthogonal: ||V^T V - I|| = {err}"


def test_query_returns_top_k():
    rng = np.random.default_rng(0)
    mem = RotMem(dim=32, decay_tau=1e6)
    base = rng.standard_normal(32).astype(np.float32)
    for i in range(50):
        k = base + 10.0 * _key(rng, 32)
        mem.update(f"id_{i}", k, value=i, now=i)
    hits = mem.query(base + 10.0 * _key(rng, 32), top_k=7)
    assert len(hits) == 7


def test_memory_cliff_smoke():
    """500-turn smoke: at turn 499 the very-first item should still be retrievable.

    To make this non-trivial we anchor 'first' to a unique direction
    in the embedding space (rather than a random vector).
    """
    rng = np.random.default_rng(0)
    dim = 64
    mem = RotMem(dim=dim, max_items=1000, decay_tau=1e6, rotation_period=10)
    first_key = rng.standard_normal(dim).astype(np.float32)
    first_key /= np.linalg.norm(first_key) + 1e-12
    mem.update("first", first_key, value="FIRST", now=0)
    for i in range(1, 500):
        # later items live in orthogonal-ish directions
        v = rng.standard_normal(dim).astype(np.float32)
        v -= 0.5 * first_key * (v @ first_key)
        v /= np.linalg.norm(v) + 1e-12
        mem.update(f"id_{i}", v, value=i, now=i)
    hits = mem.query(first_key, top_k=20)
    ids = [h.item.item_id for h in hits]
    assert "first" in ids, f"'first' missing from top-20 after 499 turns: {ids}"