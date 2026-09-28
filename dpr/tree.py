"""Search Tree data structures with Upper Confidence Bounds (UCT)."""

import math
from typing import List, Optional


class TreeNode:
    _id_counter = 0

    def __init__(self, step_text: str, parent: Optional['TreeNode'] = None, prm_score: float = 0.5, is_terminal: bool = False, tokens_used: int = 0):
        TreeNode._id_counter += 1
        self.node_id = TreeNode._id_counter
        self.step_text = step_text
        self.parent = parent
        self.children: List['TreeNode'] = []
        self.depth = parent.depth + 1 if parent else 0
        self.visits = 0
        self.value_sum = 0.0
        self.prior_score = prm_score
        self.prm_score = prm_score
        self.is_terminal = is_terminal
        self.tokens_used = tokens_used

    @property
    def q_value(self) -> float:
        return self.value_sum / self.visits if self.visits > 0 else self.prior_score

    def ucb_score(self, c_puct: float = 1.414) -> float:
        if self.parent is None:
            return self.q_value
        parent_visits = max(1, self.parent.visits)
        return self.q_value + c_puct * self.prior_score * (math.sqrt(parent_visits) / (1 + self.visits))

    def add_child(self, child: 'TreeNode') -> None:
        child.parent = self
        child.depth = self.depth + 1
        self.children.append(child)

    def backpropagate(self, reward: float) -> None:
        curr: Optional['TreeNode'] = self
        while curr is not None:
            curr.visits += 1
            curr.value_sum += reward
            curr = curr.parent

    def get_full_trajectory(self) -> str:
        steps = []
        curr: Optional['TreeNode'] = self
        while curr is not None:
            if curr.step_text:
                steps.append(curr.step_text)
            curr = curr.parent
        return "\n\n".join(reversed(steps))


class SearchTree:
    def __init__(self, root_prompt: str):
        self.root = TreeNode(step_text=root_prompt, prm_score=1.0)
        self.nodes = [self.root]
        self.terminal_nodes: List[TreeNode] = []

    def register_node(self, node: TreeNode) -> None:
        self.nodes.append(node)
        if node.is_terminal:
            self.terminal_nodes.append(node)

    def select_best(self) -> TreeNode:
        candidates = self.terminal_nodes if self.terminal_nodes else [n for n in self.nodes if not n.children]
        return max(candidates, key=lambda n: n.q_value) if candidates else self.root
