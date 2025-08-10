"""
Monte Carlo Tree Search implementation using the core library.
"""
from typing import List, Tuple, Optional, Callable, Union
import copy
import numpy as np

# Import directly from core modules
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "gomoku-ai"))

from gomoku.core.models import GameState, Player, Move
from gomoku.core.game_logic import GomokuGame

try:
    # Try relative imports first (when imported as package)
    from .tree_node import MCTSTreeNode
    from .policy_functions import (
        softmax, action_to_location, location_to_action,
        rollout_policy_fn, uniform_expand_policy_fn
    )
    from .utils import ProgressBar
except ImportError:
    # Fall back to absolute imports (when run directly)
    from tree_node import MCTSTreeNode
    from policy_functions import (
        softmax, action_to_location, location_to_action,
        rollout_policy_fn, uniform_expand_policy_fn
    )
    from utils import ProgressBar


class MCTS:
    """
    Pure Monte Carlo Tree Search implementation using the core library.

    Attributes:
        root: Root node of the search tree
        _expand_policy: Function for expanding nodes
        _rollout_policy: Function for rollout simulation
        _weight_c: Exploration weight parameter
        _compute_budget: Number of MCTS simulations to run
        _silent: Whether to suppress output
        _expand_bound: Minimum visit count before expanding a node
    """

    def __init__(self, expand_policy: Callable = None, rollout_policy: Callable = None, 
                 weight_c: float = 5, compute_budget: int = 10000, 
                 expand_bound: int = 1, silent: bool = False):
        """
        Initialize MCTS.
        
        Args:
            expand_policy: Function for node expansion (default: uniform_expand_policy_fn)
            rollout_policy: Function for rollout simulation (default: rollout_policy_fn)
            weight_c: Exploration weight
            compute_budget: Number of simulations
            expand_bound: Minimum visits before expansion
            silent: Whether to suppress output
        """
        self.root = MCTSTreeNode(None, 1.0)
        self._expand_policy = expand_policy or uniform_expand_policy_fn
        self._rollout_policy = rollout_policy or rollout_policy_fn
        self._weight_c = weight_c
        self._compute_budget = int(compute_budget)
        self._silent = silent
        self._expand_bound = min(expand_bound, compute_budget)
    
    def reset(self) -> None:
        """Reset the search tree."""
        self.root = MCTSTreeNode(None, 1.0)

    def _playout(self, game: GomokuGame) -> None:
        """
        Run a single playout from root to leaf.
        
        Args:
            game: GomokuGame instance (will be modified, provide a copy)
        """
        node = self.root
        
        # Selection phase: traverse down the tree
        while not node.is_leaf():
            action, node = node.select(self._weight_c)
            row, col = action_to_location(action, game.state.board_size)
            game.make_move(row, col)

        # Check if game ended
        winner = game.check_winner()
        is_end = winner is not None or game.state.is_board_full()
        
        # Expansion phase
        if not is_end and node.vis_times >= self._expand_bound:
            action_probs, _ = self._expand_policy(game.state)
            node.expand(action_probs)

        # Evaluation phase: evaluate the leaf node by random rollout
        bp_value = self._evaluate_rollout(game)
        
        # Backpropagation phase
        node.back_propagation(-bp_value)

    def _evaluate_rollout(self, game: GomokuGame, limit: int = 1000) -> float:
        """
        Use rollout policy to simulate game to completion.
        
        Args:
            game: Current game state
            limit: Maximum number of rollout moves
            
        Returns:
            +1 if current player wins, -1 if opponent wins, 0 if tie
        """
        # Remember the player color at the leaf node
        player_at_leaf = game.state.current_player
        
        # Make a copy for rollout
        rollout_game = GomokuGame(
            board_size=game.state.board_size,
            win_condition=game.win_condition,
            board_state=game.state.copy()
        )

        for _ in range(limit):
            winner = rollout_game.check_winner()
            if winner is not None or rollout_game.state.is_board_full():
                break
                
            action_probs = self._rollout_policy(rollout_game.state)
            if not action_probs:
                break
                
            # Select move based on probabilities
            actions, probs = zip(*action_probs)
            action = np.random.choice(actions, p=probs)
            
            row, col = action_to_location(action, rollout_game.state.board_size)
            rollout_game.make_move(row, col)
        else:
            if not self._silent:
                print(f"[Warning]: Rollout exceeded limit ({limit})")

        # Determine result
        winner = rollout_game.check_winner()
        if winner is None:
            return 0.0
        else:
            return 1.0 if winner == player_at_leaf else -1.0

    def getMove(self, game: GomokuGame) -> Tuple[int, int]:
        """
        Run all playouts sequentially and return the best move.
        
        Args:
            game: Current GomokuGame instance
            
        Returns:
            Selected move as (row, col) tuple
        """
        # For empty board, place stone at center
        if not game.state.move_history:
            center = game.state.board_size // 2
            return (center, center)
        
        # Run MCTS simulations
        if self._silent:
            for _ in range(self._compute_budget):
                game_copy = GomokuGame(
                    board_size=game.state.board_size,
                    win_condition=game.win_condition,
                    board_state=game.state.copy()
                )
                self._playout(game_copy)
        else:
            print("Thinking...")
            pb = ProgressBar(self._compute_budget, total_sharp=20)
            for _ in range(self._compute_budget):
                pb.iterStart()
                game_copy = GomokuGame(
                    board_size=game.state.board_size,
                    win_condition=game.win_condition,
                    board_state=game.state.copy()
                )
                self._playout(game_copy)
                pb.iterEnd()

        # Return most visited action
        best_action, _ = max(self.root.children.items(),
                            key=lambda act_node: act_node[1].vis_times)
        
        # Convert action to (row, col) format
        row, col = action_to_location(best_action, game.state.board_size)
        return (row, col)
    
    def test_out(self) -> List[Tuple[int, MCTSTreeNode]]:
        """Return sorted list of children for debugging."""
        return sorted(list(self.root.children.items()), key=lambda x: x[-1].vis_times)

    def think(self, game: GomokuGame, decay_level: int = 100) -> Tuple[int, int]:
        """
        Consider the current game state and suggest a move.

        Similar to getMove but with reduced compute budget.

        Args:
            game: Current game state
            decay_level: Higher values mean less computation
            
        Returns:
            Suggested move as (row, col) tuple
        """
        for _ in range(self._compute_budget // decay_level):
            game_copy = GomokuGame(
                board_size=game.state.board_size,
                win_condition=game.win_condition,
                board_state=game.state.copy()
            )
            self._playout(game_copy)
            
        best_action, _ = max(self.root.children.items(),
                            key=lambda act_node: act_node[1].vis_times)
        
        row, col = action_to_location(best_action, game.state.board_size)
        return (row, col)

    def updateWithMove(self, last_move: Optional[Tuple[int, int]], board_size: int) -> None:
        """
        Reuse the tree and advance by one move.
        
        Args:
            last_move: The move that was played as (row, col) tuple
            board_size: Size of the board
        """
        if last_move is not None:
            # Convert move to action format
            action = location_to_action(last_move[0], last_move[1], board_size)
            if action in self.root.children:
                self.root = self.root.children[action]
                self.root.parent = None
                return
        
        # Rebuild the tree if we can't find the move
        self.root = MCTSTreeNode(None, 1.0)

    def __str__(self) -> str:
        """Return string representation."""
        return f"MCTS with compute budget {self._compute_budget} and weight c {self._weight_c}"

    @property
    def silent(self) -> bool:
        """Get silent mode."""
        return self._silent
    
    @silent.setter
    def silent(self, given_value: bool) -> None:
        """Set silent mode."""
        if isinstance(given_value, bool):
            self._silent = given_value

    # Backward compatibility methods
    def testOut(self) -> List[Tuple[int, MCTSTreeNode]]:
        """Backward compatibility for testOut."""
        return self.test_out()
    
    def _evaluateRollout(self, game: GomokuGame, limit: int = 1000) -> float:
        """Backward compatibility for _evaluateRollout."""
        return self._evaluate_rollout(game, limit)


class MCTSWithDNN:
    """
    Monte Carlo Tree Search with Deep Neural Network guidance using core library.

    Attributes:
        root: Root node of the search tree
        _policy_value_fn: Function that evaluates board states with neural network
        _weight_c: Exploration weight parameter
        _compute_budget: Number of MCTS simulations to run
        _silent: Whether to suppress output
        _expand_bound: Minimum visit count before expanding a node
    """

    def __init__(self, policy_value_fn: Callable, weight_c: float = 5, 
                 compute_budget: int = 10000, expand_bound: int = 10, 
                 silent: bool = False):
        """
        Initialize MCTS with DNN.
        
        Args:
            policy_value_fn: Neural network policy and value function
                            Should accept GameState and return (action_probs, value)
            weight_c: Exploration weight
            compute_budget: Number of simulations
            expand_bound: Minimum visits before expansion
            silent: Whether to suppress output
        """
        self.root = MCTSTreeNode(None, 1.0)
        self._policy_value_fn = policy_value_fn
        self._weight_c = weight_c
        self._compute_budget = int(compute_budget)
        self._silent = silent
        self._expand_bound = min(expand_bound, compute_budget)

    def _playout(self, game: GomokuGame) -> None:
        """
        Run a single playout from root to leaf with DNN guidance.
        
        Args:
            game: GomokuGame instance (will be modified, provide a copy)
        """
        node = self.root
        
        # Selection phase: traverse down the tree
        while not node.is_leaf():
            action, node = node.select(self._weight_c)
            row, col = action_to_location(action, game.state.board_size)
            game.make_move(row, col)

        # Evaluation phase: use DNN instead of rollout
        policy, value = self._policy_value_fn(game.state)
        
        # Check for game end
        winner = game.check_winner()
        is_end = winner is not None or game.state.is_board_full()
        if not is_end: 
            # Expansion phase: expand if visit count is sufficient
            if node.vis_times >= self._expand_bound:
                node.expand(policy)
        else:
            # Terminal state: use actual game result
            if winner is None:
                value = 0.0
            else:
                value = 1.0 if game.state.current_player == winner else -1.0

        # Backpropagation phase
        node.back_propagation(-value)

    def getMove(self, game: GomokuGame, exploration_level: float) -> Tuple[Tuple[int, ...], np.ndarray]:
        """
        Run all playouts sequentially and return actions with probabilities.

        Args:
            game: Current game state
            exploration_level: Temperature parameter controlling exploration
                
        Returns:
            Tuple of (actions, probabilities)
        """
        # Run MCTS simulations
        if self._silent:
            for _ in range(self._compute_budget):
                game_copy = GomokuGame(
                    board_size=game.state.board_size,
                    win_condition=game.win_condition,
                    board_state=game.state.copy()
                )
                self._playout(game_copy)
        else:
            print("Thinking...")
            pb = ProgressBar(self._compute_budget)
            for _ in range(self._compute_budget):
                pb.iterStart()
                game_copy = GomokuGame(
                    board_size=game.state.board_size,
                    win_condition=game.win_condition,
                    board_state=game.state.copy()
                )
                self._playout(game_copy)
                pb.iterEnd()

        # Calculate move probabilities based on visit counts
        act_vis = [(act, node.vis_times)
                   for act, node in self.root.children.items()]
        acts, visits = zip(*act_vis)

        # Apply temperature to visit counts and use softmax
        probs = softmax(1.0 / exploration_level * np.log(np.array(visits) + 1e-10))

        return acts, probs

    def think(self, game: GomokuGame, decay_level: int = 100) -> Tuple[Tuple[int, ...], np.ndarray]:
        """
        Consider the current game state and suggest move probabilities.

        Args:
            game: Current game state
            decay_level: Higher values mean less computation
            
        Returns:
            Tuple of (actions, probabilities)
        """
        for _ in range(self._compute_budget // decay_level):
            game_copy = GomokuGame(
                board_size=game.state.board_size,
                win_condition=game.win_condition,
                board_state=game.state.copy()
            )
            self._playout(game_copy)
            
        act_vis = [(act, node.vis_times)
                   for act, node in self.root.children.items()]
        acts, visits = zip(*act_vis)
        
        # Normalize visit counts to probabilities
        visits_array = np.array(visits)
        probs = visits_array / visits_array.sum()
        
        return acts, probs

    def updateWithMove(self, last_move: Optional[Tuple[int, int]], board_size: int) -> None:
        """
        Reuse the tree and advance by one move.
        
        Args:
            last_move: The move that was played as (row, col) tuple
            board_size: Size of the board
        """
        if last_move is not None:
            # Convert move to action format
            action = location_to_action(last_move[0], last_move[1], board_size)
            if action in self.root.children:
                # Reuse existing subtree
                self.root = self.root.children[action]
                self.root.parent = None
                return
        
        # Rebuild the tree
        self.root = MCTSTreeNode(None, 1.0)
    
    def reset(self) -> None:
        """Reset the search tree."""
        self.root = MCTSTreeNode(None, 1.0)

    def __str__(self) -> str:
        """Return string representation."""
        return f"MCTS(DNN version) with compute budget {self._compute_budget} and weight c {self._weight_c}"
    
    @property
    def silent(self) -> bool:
        """Get silent mode."""
        return self._silent
    
    @silent.setter
    def silent(self, given_value: bool) -> None:
        """Set silent mode."""
        if isinstance(given_value, bool):
            self._silent = given_value
            
    __repr__ = __str__
