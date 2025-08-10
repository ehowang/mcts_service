#!/usr/bin/env python3
"""
Simple client to test the MCTS Player Service
"""
import requests
import json
from typing import List, Dict, Optional


class MCTSPlayerClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
    
    def get_move(self, board: List[List[str]], current_player: str, 
                 board_size: int = None, win_condition: int = 5,
                 compute_budget: int = 10000, weight_c: float = 5.0,
                 move_history: Optional[List[Dict]] = None) -> Optional[Dict]:
        """
        Get MCTS move for given board state.
        
        Args:
            board: 2D list with "X", "O", "."
            current_player: "X" or "O"
            board_size: Size of board (auto-detected if None)
            win_condition: Stones needed to win
            compute_budget: Number of MCTS simulations
            weight_c: MCTS exploration parameter
            move_history: Optional move history
        
        Returns:
            {"row": int, "col": int, "computation_time": float, "confidence": float}
        """
        if board_size is None:
            board_size = len(board)
        
        payload = {
            "board": board,
            "current_player": current_player,
            "board_size": board_size,
            "win_condition": win_condition,
            "compute_budget": compute_budget,
            "weight_c": weight_c,
            "move_history": move_history
        }
        
        response = self.session.post(f"{self.base_url}/get-move", json=payload)
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error: {response.status_code} - {response.text}")
            return None
    
    def health_check(self) -> bool:
        """Check if service is healthy."""
        try:
            response = self.session.get(f"{self.base_url}/health")
            return response.status_code == 200
        except:
            return False


def create_empty_board(size: int = 8) -> List[List[str]]:
    """Create empty board."""
    return [["." for _ in range(size)] for _ in range(size)]


def print_board(board: List[List[str]]):
    """Print board in a nice format."""
    size = len(board)
    
    print("\n   ", end="")
    for i in range(size):
        print(f"{i:2}", end="")
    print()
    
    for i, row in enumerate(board):
        print(f"{i:2} ", end="")
        for cell in row:
            print(f" {cell}", end="")
        print()


def demo_simple_game():
    """Demo: Simple game using MCTS service."""
    client = MCTSPlayerClient()
    
    # Check if service is running
    if not client.health_check():
        print("❌ MCTS service is not running. Start it with: python main.py")
        return
    
    print("✅ MCTS service is running!")
    print("\n=== Simple Gomoku Demo with MCTS Service ===")
    
    # Create game
    board_size = 8
    board = create_empty_board(board_size)
    current_player = "X"
    move_history = []
    
    print(f"Board size: {board_size}x{board_size}")
    print("You are X, MCTS is O")
    
    while True:
        print_board(board)
        print(f"\nCurrent player: {current_player}")
        
        # Check for winner (simple check)
        if is_game_over(board):
            print("Game Over!")
            break
        
        if current_player == "X":
            # Human move
            try:
                move_input = input("Your turn (row,col): ")
                row, col = map(int, move_input.split(','))
                
                if 0 <= row < board_size and 0 <= col < board_size and board[row][col] == ".":
                    board[row][col] = current_player
                    move_history.append({"row": row, "col": col, "player": current_player})
                    current_player = "O"
                else:
                    print("Invalid move! Try again.")
                    continue
                    
            except (ValueError, KeyboardInterrupt):
                print("\nExiting...")
                break
        else:
            # MCTS move
            print("MCTS thinking...")
            
            result = client.get_move(
                board=board,
                current_player=current_player,
                board_size=board_size,
                compute_budget=5000,  # Fast for demo
                move_history=move_history
            )
            
            if result:
                row, col = result["row"], result["col"]
                time_taken = result["computation_time"]
                confidence = result["confidence"]
                
                board[row][col] = current_player
                move_history.append({"row": row, "col": col, "player": current_player})
                
                print(f"MCTS played ({row}, {col}) in {time_taken:.2f}s (confidence: {confidence:.1%})")
                current_player = "X"
            else:
                print("Failed to get MCTS move")
                break


def is_game_over(board: List[List[str]]) -> bool:
    """Simple game over check (just check if board is full)."""
    for row in board:
        if "." in row:
            return False
    return True


def demo_batch_requests():
    """Demo: Test multiple positions quickly."""
    client = MCTSPlayerClient()
    
    if not client.health_check():
        print("❌ MCTS service is not running")
        return
    
    print("\n=== Batch Demo ===")
    
    # Test different positions
    test_positions = [
        {
            "name": "Empty board",
            "board": create_empty_board(8),
            "player": "X"
        },
        {
            "name": "Center taken",
            "board": [
                [".", ".", ".", ".", ".", ".", ".", "."],
                [".", ".", ".", ".", ".", ".", ".", "."],
                [".", ".", ".", ".", ".", ".", ".", "."],
                [".", ".", ".", ".", ".", ".", ".", "."],
                [".", ".", ".", ".", "X", ".", ".", "."],
                [".", ".", ".", ".", ".", ".", ".", "."],
                [".", ".", ".", ".", ".", ".", ".", "."],
                [".", ".", ".", ".", ".", ".", ".", "."]
            ],
            "player": "O"
        }
    ]
    
    for test in test_positions:
        print(f"\n--- {test['name']} ---")
        print_board(test["board"])
        
        result = client.get_move(
            board=test["board"],
            current_player=test["player"],
            compute_budget=3000  # Fast for demo
        )
        
        if result:
            print(f"MCTS suggests: ({result['row']}, {result['col']}) "
                  f"in {result['computation_time']:.2f}s "
                  f"(confidence: {result['confidence']:.1%})")
        else:
            print("Failed to get move")


if __name__ == "__main__":
    print("Choose demo:")
    print("1. Interactive game")
    print("2. Batch position test")
    
    try:
        choice = input("Enter choice (1-2): ").strip()
        if choice == "1":
            demo_simple_game()
        elif choice == "2":
            demo_batch_requests()
        else:
            print("Invalid choice")
    except KeyboardInterrupt:
        print("\nExiting...")