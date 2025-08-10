"""
Tree node implementation for MCTS using the core library.
"""
from typing import Optional, Dict, List, Tuple
import numpy as np


class MCTSTreeNode:
    """
    A node in the MCTS tree.
    
    Each node tracks its value Q, prior probability P, and visit count for UCB calculation.

    Attributes:
        parent: Parent node (None for root)
        children: Dictionary mapping actions to child nodes
        _vis_times: Number of times this node has been visited
        _Q: Q-value (average reward)
        _U: U-value (exploration bonus)
        _P: Prior probability
    """

    def __init__(self, parent: Optional['MCTSTreeNode'], prior_prob: float):
        """Initialize MCTS tree node."""
        self.parent = parent
        self.children: Dict[int, 'MCTSTreeNode'] = {}
        self._vis_times = 0
        self._Q = 0.0  # Q = sum(rollout_results) / vis_times
        self._U = 0.0  # U = prior_prob / (1 + vis_times)
        self._P = prior_prob

    def expand(self, action_priors: List[Tuple[int, float]]) -> None:
        """
        Expand this node by creating all its children.

        Args:
            action_priors: List of (action, prior_probability) pairs
                          action is the move index (row * board_width + col)
        """
        for action, prob in action_priors:
            if action not in self.children:
                self.children[action] = MCTSTreeNode(self, prob)

    def select(self, weight_c: float) -> Tuple[int, 'MCTSTreeNode']:
        """
        Select action among children that maximizes UCB value.

        Args:
            weight_c: Exploration weight
            
        Returns:
            Tuple of (action, next_node)
        """
        return max(self.children.items(),
                   key=lambda act_node: act_node[1].evaluate(weight_c))

    def update(self, bp_value: float) -> None:
        """
        Update node values from leaf evaluation.

        Args:
            bp_value: Backpropagation value from current player's perspective
        """
        self._vis_times += 1

        # Running average: Q_{N+1} = Q_N + (v_{N+1} - Q_N) / (N+1)
        self._Q += (bp_value - self._Q) / self._vis_times

    def back_propagation(self, bp_value: float) -> None:
        """
        Backpropagate the final result from leaf to root.
        
        Args:
            bp_value: Value to backpropagate
        """
        self.update(bp_value)
        if self.parent:  # If has parent (not root)
            # Good result for one player means bad result for the other
            self.parent.back_propagation(-bp_value)

    def evaluate(self, weight_c: float) -> float:
        """
        Calculate UCB value for this node.

        Args:
            weight_c: Exploration weight controlling Q vs P balance
            
        Returns:
            UCB value combining exploitation (Q) and exploration (U)
        """
        if self.parent:
            self._U = (self._P * np.sqrt(self.parent._vis_times) / 
                      (1 + self._vis_times))
        else:
            self._U = 0.0
        return self._Q + weight_c * self._U

    def is_leaf(self) -> bool:
        """Check if this node is a leaf (has no children)."""
        return len(self.children) == 0

    def is_root(self) -> bool:
        """Check if this node is the root (has no parent)."""
        return self.parent is None

    @property
    def vis_times(self) -> int:
        """Get visit count."""
        return self._vis_times

    @property
    def Q_value(self) -> float:
        """Get Q-value."""
        return self._Q

    @property
    def prior_prob(self) -> float:
        """Get prior probability."""
        return self._P

    # Backward compatibility
    def backPropagation(self, bp_value: float) -> None:
        """Backward compatibility for backPropagation."""
        return self.back_propagation(bp_value)
