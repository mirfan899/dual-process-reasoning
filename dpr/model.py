"""Neural Model for Dual-Process Reasoning."""

from typing import List, Tuple
import numpy as np
from .adapter import OrthogonalAdapter


class DualProcessModel:
    def __init__(self, vocab_size: int = 100, hidden_dim: int = 64, rank: int = 16):
        np.random.seed(42)
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.embed = np.random.randn(vocab_size, hidden_dim) * 0.2
        self.layer1 = OrthogonalAdapter(hidden_dim, hidden_dim, rank=rank)
        self.layer2 = OrthogonalAdapter(hidden_dim, hidden_dim, rank=rank)
        self.head = np.random.randn(hidden_dim, vocab_size) * 0.2

    def forward(self, token_ids: List[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        x = np.mean(self.embed[token_ids], axis=0, keepdims=True)
        h1 = np.maximum(0, self.layer1.forward(x))
        h2 = np.maximum(0, self.layer2.forward(h1))
        logits = h2 @ self.head
        return logits, x, h1, h2

    def train_step(self, token_ids: List[int], target_token: int, lr: float, use_projection: bool = True) -> float:
        logits, x, h1, h2 = self.forward(token_ids)
        shift = logits - np.max(logits)
        probs = np.exp(shift) / (np.sum(np.exp(shift)) + 1e-12)
        loss = -np.log(probs[0, target_token] + 1e-12)
        dlogits = probs.copy()
        dlogits[0, target_token] -= 1.0

        dh2 = dlogits @ self.head.T
        dh1 = self.layer2.backward(dh2 * (h2 > 0), apply_projection=use_projection)
        self.layer2.step(lr)
        self.layer1.backward(dh1 * (h1 > 0), apply_projection=use_projection)
        self.layer1.step(lr)
        return float(loss)

    def accumulate_activations(self, token_ids: List[int]) -> None:
        x = np.mean(self.embed[token_ids], axis=0, keepdims=True)
        self.layer1.accumulate_covariance(x)
        h1 = np.maximum(0, self.layer1.forward(x))
        self.layer2.accumulate_covariance(h1)

    def consolidate(self) -> None:
        self.layer1.recompute_null_space()
        self.layer2.recompute_null_space()
        self.layer1.consolidate_to_base()
        self.layer2.consolidate_to_base()
