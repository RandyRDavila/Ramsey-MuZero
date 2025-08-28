from dataclasses import dataclass
from typing import Dict, Any, Tuple, List
import math
import random
import numpy as np
import torch

from .utils import masked_softmax


@dataclass
class Node:
    prior: float
    to_play: int
    hidden_state: torch.Tensor  # [1,256]
    value_sum: float = 0.0
    visits: int = 0
    children: Dict[int, "EdgeChild"] = None
    expanded_pairs: List[int] = None  # indices into candidate actions

    def __post_init__(self):
        if self.children is None:
            self.children = {}
        if self.expanded_pairs is None:
            self.expanded_pairs = []


@dataclass
class EdgeChild:
    prior: float
    action_id: int
    visits: int = 0
    value_sum: float = 0.0
    reward_sum: float = 0.0
    child: Node = None


class MCTS:
    def __init__(self, cfg, net, env_like):
        self.cfg = cfg
        self.net = net
        self.env_like = env_like  # used for legal action listing & pair mapping

    def run(self, root_obs, root_to_play, root_legal_pairs, root_logits_extra=None):
        """
        Build root Node and perform cfg.mcts_sims simulations.
        """
        device = self.cfg.device
        # Initial inference
        with torch.no_grad():
            v_logits, r_logits, h = self.net.initial_inference(root_obs)
        root = Node(prior=1.0, to_play=root_to_play, hidden_state=h)

        # Policy over legal pairs computed externally; we’ll get priors for first expansion dynamically.

        # Dirichlet noise at root
        noise_frac = self.cfg.root_noise_frac

        for sim in range(self.cfg.mcts_sims):
            node = root
            search_path = [node]

            # Selection
            while node.children:
                action, child = self._select_child(node)
                node = child.child
                search_path.append(node)

            # Expansion
            leaf = node
            # Evaluate policy over current legal actions for the leaf state
            # (In our trainer, we carry the candidate pairs and logits per step.)
            # For simplicity here, evenly spread priors on first encounter; trainer will pass priors.
            if not leaf.children:
                # Delay expansion until backprop assigns values (we use uniform priors here)
                pass

            # Value from last v_logits (constant in this simplification) — in trainer we recompute
            value = 0.0
            self._backpropagate(search_path, value, to_play=root_to_play)

        return root

    def _select_child(self, node: Node):
        # PUCT with progressive widening cap
        N = node.visits
        Kmax = math.ceil(self.cfg.widen_c * (N ** self.cfg.widen_alpha)) if N > 0 else 1
        if len(node.children) < Kmax:
            # expand a new child according to highest prior among unseen
            pass  # trainer will handle actual expansion, here kept minimal for brevity

        # choose best UCB among children
        best, best_score = None, -1e9
        c_puct = 1.25
        for a, edge in node.children.items():
            Q = 0.0 if edge.visits == 0 else edge.value_sum / edge.visits
            U = c_puct * edge.prior * math.sqrt(node.visits + 1e-6) / (1 + edge.visits)
            score = Q + U
            if score > best_score:
                best_score = score
                best = (a, edge)
        return best

    def _backpropagate(self, path: List[Node], value: float, to_play: int):
        for node in reversed(path):
            node.value_sum += value if node.to_play == to_play else -value
            node.visits += 1
