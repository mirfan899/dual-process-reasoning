"""Orthogonal Null-Space Parameter Adapter for zero-interference synaptic consolidation."""

import numpy as np


class OrthogonalAdapter:
    def __init__(self, in_features: int, out_features: int, rank: int = 16, alpha: float = 16.0, energy_threshold: float = 0.85):
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.scaling = alpha / rank
        self.energy_threshold = energy_threshold

        limit = np.sqrt(6.0 / (in_features + out_features))
        self.W0 = np.random.uniform(-limit, limit, (out_features, in_features))
        self.A = np.random.uniform(-limit, limit, (rank, in_features))
        self.B = np.zeros((out_features, rank))

        self.cov_accum = np.zeros((in_features, in_features))
        self.cov_count = 0
        self.projector = np.eye(in_features)

        self.last_x = None
        self.last_h = None
        self.grad_A = None
        self.grad_B = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.last_x = x
        self.last_h = x @ self.A.T
        return x @ self.W0.T + (self.last_h @ self.B.T) * self.scaling

    def accumulate_covariance(self, x: np.ndarray) -> None:
        self.cov_accum += x.T @ x
        self.cov_count += x.shape[0]

    def recompute_null_space(self) -> int:
        if self.cov_count == 0:
            self.projector = np.eye(self.in_features)
            return 0
        C = self.cov_accum / float(self.cov_count)
        evals, evecs = np.linalg.eigh(C)
        idx = np.argsort(evals)[::-1]
        vals, vecs = evals[idx], evecs[:, idx]
        total = np.sum(np.clip(vals, 0.0, None))
        if total == 0:
            k = 0
        else:
            cum = np.cumsum(np.clip(vals, 0.0, None)) / total
            k_match = np.where(cum >= self.energy_threshold)[0]
            k = int(k_match[0]) + 1 if len(k_match) > 0 else self.in_features

        k = min(k, self.in_features - self.rank)
        k = max(0, k)
        if k > 0:
            U_core = vecs[:, :k]
            self.projector = np.eye(self.in_features) - (U_core @ U_core.T)
        else:
            self.projector = np.eye(self.in_features)
        return k

    def backward(self, grad_out: np.ndarray, apply_projection: bool = True) -> np.ndarray:
        grad_adapter = grad_out * self.scaling
        self.grad_B = grad_adapter.T @ self.last_h
        grad_h = grad_adapter @ self.B
        raw_grad_A = grad_h.T @ self.last_x
        self.grad_A = raw_grad_A @ self.projector if apply_projection else raw_grad_A
        return grad_out @ self.W0 + (grad_h @ self.A) * self.scaling

    def consolidate_to_base(self) -> None:
        self.W0 += (self.B @ self.A) * self.scaling
        limit = np.sqrt(6.0 / (self.in_features + self.out_features))
        self.A = np.random.uniform(-limit, limit, (self.rank, self.in_features)) @ self.projector
        self.B = np.zeros((self.out_features, self.rank))
        self.grad_A, self.grad_B = None, None

    def step(self, lr: float) -> None:
        if self.grad_A is not None: self.A -= lr * self.grad_A
        if self.grad_B is not None: self.B -= lr * self.grad_B
