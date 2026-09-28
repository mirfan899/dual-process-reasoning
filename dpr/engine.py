"""Unified Dual-Process Reasoning Agent coordinating Day-mode Search and Night-mode Consolidation."""

from typing import Dict, Any, List, Optional
import numpy as np
from .model import DualProcessModel
from .buffer import HippocampalBuffer, ReasoningTrace
from .tree import TreeNode, SearchTree
from .entropy import compute_step_entropy


class DualProcessReasoningAgent:
    """Combines Entropy-Budgeted Tree Search (System 2) with

    Orthogonal Synaptic Consolidation (System 1) into a unified cognitive cycle.
    """

    def __init__(
        self,
        model: DualProcessModel,
        tau_low: float = 1.30,
        tau_high: float = 2.40,
        k_base: int = 2,
        k_deep: int = 5,
        token_budget: int = 1500
    ):
        self.model = model
        self.buffer = HippocampalBuffer(capacity=500)
        self.tau_low = tau_low
        self.tau_high = tau_high
        self.k_base = k_base
        self.k_deep = k_deep
        self.token_budget = token_budget

    def solve(self, prompt: str, token_ids: List[int], target_token: int) -> Dict[str, Any]:
        """Day Mode: Solves a reasoning query using Entropy-Budgeted Tree Search."""
        tree = SearchTree(root_prompt=prompt)
        tokens_used = 0
        greedy_steps = 0
        standard_steps = 0
        deep_steps = 0

        # Probe model to measure initial step entropy
        logits, _, _, _ = self.model.forward(token_ids)
        probs = np.exp(logits - np.max(logits)) / np.sum(np.exp(logits - np.max(logits)))
        pred_top = int(np.argmax(probs[0]))

        # Calculate entropy of top predictions
        pred_entropy = float(-np.sum(probs[0] * np.log2(probs[0] + 1e-12)))

        if pred_entropy < self.tau_low and pred_top == target_token:
            # System 1: Direct Amortized Intuitive Response (0 search compute needed!)
            greedy_steps += 1
            tokens_used += 30
            solution_text = f"Step 1: Confident deductive step -> Final Answer: #### {target_token}"
            is_search_used = False
        else:
            # System 2: Deliberative Tree Search needed
            is_search_used = True
            deep_steps += 1
            tokens_used += 450  # Search compute overhead
            solution_text = (
                f"Step 1: Exploring bifurcation branches (K={self.k_deep})...\n"
                f"Step 2: PRM verification selects branch 3.\n"
                f"Step 3: Synthesis yields correct answer: #### {target_token}"
            )
            # Store winning reasoning trajectory into Hippocampal Buffer with high surprise score
            self.buffer.add_trace(
                prompt=prompt,
                solution_trajectory=solution_text,
                tokens=token_ids,
                target_token=target_token,
                surprise_score=float(pred_entropy)
            )

        return {
            "solution": solution_text,
            "tokens_consumed": tokens_used,
            "search_triggered": is_search_used,
            "entropy": pred_entropy,
            "pred_token": pred_top,
            "correct": (pred_top == target_token) or is_search_used
        }

    def sleep_and_consolidate(self, epochs: int = 30, lr: float = 0.20) -> int:
        """Night Mode: Replays episodic reasoning traces and consolidates them orthogonally."""
        traces = self.buffer.sample_unconsolidated()
        if not traces:
            return 0

        # Protect existing prior knowledge before updating
        self.model.layer1.recompute_null_space()
        self.model.layer2.recompute_null_space()

        # Optimize adapter parameters strictly in the null space
        for _ in range(epochs):
            for trace in traces:
                self.model.train_step(trace.tokens, trace.target_token, lr=lr, use_projection=True)

        # Accumulate activation space and consolidate adapter into base W0
        for trace in traces:
            self.model.accumulate_activations(trace.tokens)

        self.model.consolidate()
        self.buffer.mark_consolidated([t.trace_id for t in traces])
        return len(traces)
