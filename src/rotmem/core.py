"""Pure-Python implementation of the RotMem residual-rotation buffer.

The implementation is intentionally minimal:

- Uses numpy only (no torch, no GPU).
- Deterministic (no learned weights, no random seeds beyond a single
  numpy default_rng for tie-breaking during merge).
- < 200 lines of executable logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np


@dataclass
class MemoryItem:
    """A single memory record."""

    item_id: str
    key: np.ndarray          # (d,) embedding
    value: Any               # arbitrary payload (e.g. text)
    strength: float = 1.0
    last_used: int = 0
    created_at: int = 0

    def to_dict(self) -> dict:
        return {
            "item_id": self.item_id,
            "key": self.key.tolist(),
            "value": self.value,
            "strength": self.strength,
            "last_used": self.last_used,
            "created_at": self.created_at,
        }


@dataclass
class QueryHit:
    item: MemoryItem
    score: float


@dataclass
class _Config:
    dim: int = 384
    max_items: int = 5000
    decay_tau: float = 50.0
    merge_threshold: float = 0.92
    rotation_period: int = 50
    rotation_drift_tol: float = 0.01
    retrieval_k: int = 5
    rotation_enabled: bool = True
    decay_enabled: bool = True
    merge_enabled: bool = True
    use_strength_weight: bool = True


class RotMem:
    """Residual-rotation memory buffer.

    Parameters
    ----------
    dim : int
        Embedding dimensionality.
    max_items : int
        Hard cap on the number of stored items; triggers consolidation.
    decay_tau : float
        Half-life-style time constant for strength decay, in turns.
    merge_threshold : float
        Cosine similarity threshold above which two items are merged
        instead of evicting.
    rotation_period : int
        How many recent items define the residual subspace basis.
    """

    def __init__(
        self,
        dim: int = 384,
        max_items: int = 5000,
        decay_tau: float = 50.0,
        merge_threshold: float = 0.92,
        rotation_period: int = 50,
        retrieval_k: int = 5,
        **flags: bool,
    ) -> None:
        cfg = _Config(
            dim=dim,
            max_items=max_items,
            decay_tau=decay_tau,
            merge_threshold=merge_threshold,
            rotation_period=rotation_period,
            retrieval_k=retrieval_k,
            **flags,
        )
        self.cfg = cfg
        self._items: list[MemoryItem] = []
        self._turn: int = 0
        self._rng = np.random.default_rng(0)

    @property
    def size(self) -> int:
        return len(self._items)

    def attach(self, agent: Any) -> None:
        """Hot-plug into an LLM agent.

        The agent is expected to expose:
            agent.encode(text) -> np.ndarray   of shape (dim,)
        RotMem only requires the encode() surface.
        """
        self._agent = agent

    def update(
        self,
        item_id: str,
        key: np.ndarray,
        value: Any = None,
        now: Optional[int] = None,
    ) -> None:
        """Insert (or replace) a memory item.

        Stored keys are never mutated after insertion. The orthogonal
        rotation affects only the *basis* used at query time, so an
        item inserted at turn 0 remains retrievable by its original
        embedding at turn 499.
        """
        now = now if now is not None else self._turn
        key = np.asarray(key, dtype=np.float32).reshape(-1)

        if self.cfg.decay_enabled and self._items:
            for it in self._items:
                it.strength *= np.exp(-(now - it.last_used) / self.cfg.decay_tau)
                it.last_used = now

        for it in self._items:
            if it.item_id == item_id:
                it.key = key
                it.value = value
                it.strength = 1.0
                it.last_used = now
                it.created_at = now
                self._turn = now + 1
                return
        self._items.append(
            MemoryItem(
                item_id=item_id,
                key=key,
                value=value,
                strength=1.0,
                last_used=now,
                created_at=now,
            )
        )
        self._turn = now + 1

        if self.cfg.merge_enabled and len(self._items) > self.cfg.max_items:
            self._consolidate()

    def query(
        self,
        key: np.ndarray,
        top_k: Optional[int] = None,
    ) -> list[QueryHit]:
        """Return top-k items by strength-weighted cosine similarity."""
        if not self._items:
            return []
        k_arr = np.asarray(key, dtype=np.float32).reshape(-1)
        keys = np.stack([it.key for it in self._items], axis=0)
        if self.cfg.rotation_enabled and len(self._items) >= 2:
            V = self._current_basis()
            # consistent rotation: keys @ V.T = (V @ keys.T).T, and
            # V @ k_arr = column rotation. So keys[3] @ V.T equals
            # (V @ keys[3]).T == (V @ k_arr).T -> same vector.
            k_arr = V @ k_arr
            keys = keys @ V.T
        kn = k_arr / (np.linalg.norm(k_arr) + 1e-12)
        N = keys / (np.linalg.norm(keys, axis=1, keepdims=True) + 1e-12)
        cos = N @ kn
        if self.cfg.use_strength_weight:
            score = cos * np.array([it.strength for it in self._items])
        else:
            score = cos
        order = np.argsort(-score)
        k = top_k or self.cfg.retrieval_k
        return [QueryHit(item=self._items[i], score=float(score[i])) for i in order[:k]]

    def consolidate(self) -> None:
        """Run one consolidation pass."""
        self._consolidate()

    def _current_basis(self) -> np.ndarray:
        """Return the orthogonal basis V_t for the current residual subspace.

        Only the **top-`rank`** eigenvectors are kept (where `rank` is
        the numerical rank of the recent-items covariance), so vectors
        are not projected into the null space when fewer items than
        ``dim`` are present.
        """
        dim = self.cfg.dim
        if len(self._items) < 2:
            return np.eye(dim, dtype=np.float32)
        recent = np.stack(
            [it.key for it in self._items[-self.cfg.rotation_period:]],
            axis=0,
        )
        cov = recent.T @ recent
        w, Q = np.linalg.eigh(cov.astype(np.float32))
        tol = max(recent.shape[0], dim) * float(np.max(np.abs(w))) * 1e-6
        rank = max(2, min(int(np.sum(w > tol)), dim))
        Q_top = Q[:, -rank:]
        if rank < dim:
            # pad with orthonormal columns from identity
            P = np.eye(dim, dtype=np.float32)
            for i in range(dim):
                v = P[:, i].copy()
                for j in range(Q_top.shape[1]):
                    v -= (v @ Q_top[:, j]) * Q_top[:, j]
                n = float(np.linalg.norm(v))
                if n > 1e-6:
                    Q_top = np.column_stack([Q_top, v / n])
                if Q_top.shape[1] >= dim:
                    break
        return Q_top.astype(np.float32)

    def _consolidate(self) -> None:
        """Merge the weakest pair whose cosine > merge_threshold."""
        if len(self._items) <= self.cfg.max_items:
            return
        order = sorted(range(len(self._items)), key=lambda i: self._items[i].strength)
        merged: set[int] = set()
        new_items: list[MemoryItem] = []
        kept = set(range(len(self._items)))

        keys = np.stack([it.key for it in self._items], axis=0)
        N = keys / (np.linalg.norm(keys, axis=1, keepdims=True) + 1e-12)

        for i in order:
            if i in merged or i not in kept:
                continue
            it_i = self._items[i]
            best_j = -1
            best_cos = self.cfg.merge_threshold
            for j in kept - merged - {i}:
                c = float(N[i] @ N[j])
                if c > best_cos:
                    best_cos = c
                    best_j = j
            if best_j >= 0:
                it_j = self._items[best_j]
                w_i, w_j = it_i.strength, it_j.strength
                w_sum = w_i + w_j + 1e-12
                new_key = (w_i * it_i.key + w_j * it_j.key) / w_sum
                new_strength = (w_i + w_j) / 2.0
                new_items.append(
                    MemoryItem(
                        item_id=it_i.item_id,
                        key=new_key.astype(np.float32),
                        value=it_i.value if it_i.created_at >= it_j.created_at else it_j.value,
                        strength=new_strength,
                        last_used=max(it_i.last_used, it_j.last_used),
                        created_at=min(it_i.created_at, it_j.created_at),
                    )
                )
                merged.add(i)
                merged.add(best_j)
            else:
                new_items.append(it_i)
                merged.add(i)
            if len(self._items) - len(merged) + len(new_items) <= self.cfg.max_items:
                for k in kept - merged:
                    new_items.append(self._items[k])
                self._items = new_items
                return
        self._items = new_items
        if len(self._items) > self.cfg.max_items:
            self._items = self._items[-self.cfg.max_items :]