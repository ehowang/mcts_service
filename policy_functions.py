"""
Policy functions for MCTS rollout and expansion phases using core library.
"""
from typing import List, Tuple, Iterator
import numpy as np

# Setup paths for imports
try:
    from .setup_paths import setup_paths
    setup_paths()
except ImportError:
    pass

# Import directly from core modules to avoid loading the full gomoku package
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "gomoku-ai"))

from gomoku.core.models import GameState


def rollout_policy_fn(state: GameState) -> List[Tuple[int, float]]:
    """
    Random policy function for MCTS rollout phase.

    This function provides random action probabilities for the rollout simulation.
    The rollout phase plays randomly until game termination and backpropagates
    the result to all nodes from terminal to root.

    Args:
        state: Current GameState in rollout phase

    Returns:
        List of (action, probability) pairs where action is move index
    """
    legal_moves = state.get_legal_moves()
    action_probs = np.random.rand(len(legal_moves))
    action_probs = action_probs / action_probs.sum()  # Normalize
    
    # Convert (row, col) to move index
    actions = []
    for row, col in legal_moves:
        action = row * state.board_size + col
        actions.append(action)
    
    return list(zip(actions, action_probs))


def uniform_expand_policy_fn(state: GameState) -> Tuple[List[Tuple[int, float]], float]:
    """
    Uniform policy function for MCTS expansion phase.

    This function provides uniform prior probabilities for all available actions
    when expanding a node in the MCTS tree. Also returns an evaluation value of 0
    since this is pure MCTS without neural network guidance.

    Args:
        state: Current GameState

    Returns:
        Tuple of (action_probability_pairs, evaluation_value)
    """
    legal_moves = state.get_legal_moves()
    n_moves = len(legal_moves)
    
    if n_moves == 0:
        return [], 0.0
    
    # Uniform probabilities
    prob = 1.0 / n_moves
    
    # Convert (row, col) to move index and create action-prob pairs
    action_probs = []
    for row, col in legal_moves:
        action = row * state.board_size + col
        action_probs.append((action, prob))
    
    return action_probs, 0.0


def softmax(x: np.ndarray) -> np.ndarray:
    """Apply softmax function to input array."""
    probs = np.exp(x - np.max(x))  # Avoid overflow
    probs /= np.sum(probs)
    return probs


def action_to_location(action: int, board_size: int) -> Tuple[int, int]:
    """Convert action index to board location (row, col)."""
    row = action // board_size
    col = action % board_size
    return row, col


def location_to_action(row: int, col: int, board_size: int) -> int:
    """Convert board location (row, col) to action index."""
    return row * board_size + col
