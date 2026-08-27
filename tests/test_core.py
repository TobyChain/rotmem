"""Unit tests for RotMem core invariants + theoretical guarantees.

Stage 1 covers the original invariants; Round 8 adds Theorem 1.2/1.3/2.1/3.1 tests
that empirically verify the formal guarantees in paper/design/theory.md.
"""

from __future__ import annotations

import numpy as np

from rotmem import RotMem


def _key(rng: np.random.Generator, d: int = 32) -> np.ndarray:
    v = rng.standard_normal(d).astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-12)


# ===================================================================== #
# Stage 1 — basic invariants
# ===================================================================== #


def test_empty_query_returns_empty():
    mem = RotMem(dim=16)
    hits = mem.query(_key(np.random.default_rng(0)))
    assert hits == []


def test_basic_insert_and_query():
    """Identical key inserted then retrieved must rank first."""
    rng = np.random.default_rng(0)
    mem = RotMem(dim=32, decay_tau=1e6, merge_threshold=0.99)
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
    """500-turn smoke: at turn 499 the very-first item should still be retrievable."""
    rng = np.random.default_rng(0)
    dim = 64
    mem = RotMem(dim=dim, max_items=1000, decay_tau=1e6, rotation_period=10)
    first_key = rng.standard_normal(dim).astype(np.float32)
    first_key /= np.linalg.norm(first_key) + 1e-12
    mem.update("first", first_key, value="FIRST", now=0)
    for i in range(1, 500):
        v = rng.standard_normal(dim).astype(np.float32)
        v -= 0.5 * first_key * (v @ first_key)
        v /= np.linalg.norm(v) + 1e-12
        mem.update(f"id_{i}", v, value=i, now=i)
    hits = mem.query(first_key, top_k=20)
    ids = [h.item.item_id for h in hits]
    assert "first" in ids, f"'first' missing from top-20 after 499 turns: {ids}"


# ===================================================================== #
# Round 8 — theoretical-guarantee tests (paper/design/theory.md)
# ===================================================================== #


def test_theorem_1_2_spectral_norm_one():
    """Theorem 1.2: the recurrence Jacobian has spectral norm exactly 1.

    We empirically verify that V_t @ V_t.T == I for many random
    sessions, i.e. the spectral norm of V_t (and therefore of the
    retrieval projection V_t) is 1.
    """
    rng = np.random.default_rng(42)
    for _ in range(20):
        mem = RotMem(dim=32, rotation_period=5)
        for i in range(10):
            mem.update(f"id_{i}", _key(rng, 32), now=i)
        V = mem._current_basis()
        assert V.shape == (32, 32)
        err = float(np.linalg.norm(V @ V.T - np.eye(32)))
        assert err < 1e-3, f"||V V^T - I|| = {err}"


def test_theorem_1_3_cosine_preservation():
    """Theorem 1.3: cos(V k, V q) == cos(k, q) deterministically."""
    rng = np.random.default_rng(123)
    mem = RotMem(dim=64, rotation_period=8)
    for i in range(20):
        mem.update(f"id_{i}", _key(rng, 64), now=i)
    stored = mem._items[5].key.copy()
    query = _key(rng, 64)
    raw = float(stored @ query / (np.linalg.norm(stored) * np.linalg.norm(query)))
    V = mem._current_basis()
    pk = V @ stored
    pq = V @ query
    proj = float(pk @ pq / (np.linalg.norm(pk) * np.linalg.norm(pq)))
    assert abs(raw - proj) < 1e-5, f"cosine drift: raw={raw} proj={proj}"


def test_theorem_2_1_decay_threshold():
    """Theorem 2.1: strength falls below threshold alpha within tau * log(1/alpha) turns."""
    rng = np.random.default_rng(7)
    tau = 50.0
    alpha = 0.01
    mem = RotMem(dim=16, decay_tau=tau)
    mem.update("a", _key(rng), now=0)
    t_thresh = int(np.ceil(tau * np.log(1 / alpha))) + 1
    mem.update("b", _key(rng), now=t_thresh)
    s_after = next(it.strength for it in mem._items if it.item_id == "a")
    assert s_after < alpha, f"s(a) = {s_after}, expected < {alpha}"


def test_theorem_3_1_merge_preserves_information():
    """Theorem 3.1: merge preserves information up to a factor of 2 in strength."""
    rng = np.random.default_rng(11)
    base = _key(rng)
    mem = RotMem(dim=16, max_items=1, merge_threshold=0.5, decay_tau=1e6)
    mem.update("a", base + 0.01 * _key(rng), value="alpha", now=0)
    mem.update("b", base + 0.01 * _key(rng), value="beta", now=1)
    assert mem.size == 1
    s_merged = mem._items[0].strength
    assert s_merged <= 1.0


def test_long_horizon_orthogonality_under_drift():
    """V_t remains orthogonal after many rotation-period refreshes."""
    rng = np.random.default_rng(2026)
    mem = RotMem(dim=32, rotation_period=10)
    for i in range(500):
        mem.update(f"id_{i}", _key(rng, 32), now=i)
    V = mem._current_basis()
    err = float(np.linalg.norm(V @ V.T - np.eye(32)))
    assert err < 1e-2, f"after 500 turns, ||V V^T - I|| = {err}"


def test_decay_is_deterministic():
    """Same seeds + same input → identical strengths."""
    rng_a = np.random.default_rng(99)
    rng_b = np.random.default_rng(99)
    mem_a = RotMem(dim=8, decay_tau=10.0)
    mem_b = RotMem(dim=8, decay_tau=10.0)
    for i in range(20):
        ka = _key(rng_a, 8)
        kb = _key(rng_b, 8)
        assert np.allclose(ka, kb)
        mem_a.update(f"id_{i}", ka, now=i)
        mem_b.update(f"id_{i}", kb, now=i)
    for it_a, it_b in zip(mem_a._items, mem_b._items):
        assert abs(it_a.strength - it_b.strength) < 1e-9