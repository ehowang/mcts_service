#!/usr/bin/env python3
"""
MCTS Player Service - Simple API that takes board state and returns best move
"""
import sys
from pathlib import Path
from typing import List, Optional
import time

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

from models import GameState, Player, Move
from game_logic import GomokuGame
from mcts import MCTS, rollout_policy_fn, uniform_expand_policy_fn


app = FastAPI(
    title="MCTS Player Service",
    description="Simple API for MCTS move decisions",
    version="1.0.0"
)


class MCTSRequest(BaseModel):
    board: List[List[str]]  # 2D array: "X", "O", "."
    current_player: str  # "X" or "O"
    board_size: int = 15
    win_condition: int = 5
    compute_budget: int = 10000
    weight_c: float = 5.0
    move_history: Optional[List[dict]] = None  # [{"row": 0, "col": 0, "player": "X"}, ...]


class MCTSResponse(BaseModel):
    row: int
    col: int
    computation_time: float
    confidence: float  # Visit count ratio of chosen move


@app.post("/get-move", response_model=MCTSResponse)
async def get_mcts_move(request: MCTSRequest):
    """Get the best move from MCTS given current board state."""
    
    try:
        # Validate board
        if len(request.board) != request.board_size:
            raise HTTPException(status_code=400, detail="Board height doesn't match board_size")
        
        for row in request.board:
            if len(row) != request.board_size:
                raise HTTPException(status_code=400, detail="Board width doesn't match board_size")
            for cell in row:
                if cell not in ["X", "O", "."]:
                    raise HTTPException(status_code=400, detail="Invalid board cell value")
        
        # Convert current_player string to Player enum
        if request.current_player == "X":
            current_player = Player.BLACK
        elif request.current_player == "O":
            current_player = Player.WHITE
        else:
            raise HTTPException(status_code=400, detail="current_player must be 'X' or 'O'")
        
        # Create move history from request
        move_history = []
        if request.move_history:
            for move_data in request.move_history:
                player = Player.BLACK if move_data["player"] == "X" else Player.WHITE
                move_history.append(Move(move_data["row"], move_data["col"], player))
        
        # Create GameState
        game_state = GameState(
            board=request.board,
            current_player=current_player,
            move_history=move_history,
            board_size=request.board_size
        )
        
        # Create GomokuGame
        game = GomokuGame(
            board_size=request.board_size,
            win_condition=request.win_condition,
            board_state=game_state
        )
        
        # Check if game is already over
        winner = game.check_winner()
        if winner or game.state.is_board_full():
            raise HTTPException(status_code=400, detail="Game is already over")
        
        # Check if there are legal moves
        legal_moves = game.state.get_legal_moves()
        if not legal_moves:
            raise HTTPException(status_code=400, detail="No legal moves available")
        
        # Create MCTS instance
        mcts = MCTS(
            expand_policy=uniform_expand_policy_fn,
            rollout_policy=rollout_policy_fn,
            weight_c=request.weight_c,
            compute_budget=request.compute_budget,
            silent=True
        )
        
        # Get MCTS move
        start_time = time.time()
        row, col = mcts.getMove(game)
        computation_time = time.time() - start_time
        
        # Calculate confidence (visit ratio of chosen move)
        confidence = 0.0
        if mcts.root.children:
            from mcts_service.policy_functions import location_to_action
            chosen_action = location_to_action(row, col, request.board_size)
            total_visits = sum(child.vis_times for child in mcts.root.children.values())
            if chosen_action in mcts.root.children and total_visits > 0:
                confidence = mcts.root.children[chosen_action].vis_times / total_visits
        
        return MCTSResponse(
            row=row,
            col=col,
            computation_time=computation_time,
            confidence=confidence
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"MCTS computation error: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "mcts-player"}


@app.get("/")
async def root():
    """Root endpoint with usage info."""
    return {
        "service": "MCTS Player Service",
        "description": "Send board state to /get-move endpoint to get AI move",
        "docs": "/docs"
    }


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )