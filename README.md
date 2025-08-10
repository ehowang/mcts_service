# MCTS Player Service

A simple REST API service that provides MCTS (Monte Carlo Tree Search) move decisions for Gomoku. Send a board state, get the best move back.

## Quick Start

```bash
cd mcts_service
pip install fastapi uvicorn
python main.py
```

Service runs on `http://localhost:8000`  
API docs: `http://localhost:8000/docs`

## API Usage

### Get MCTS Move

**POST** `/get-move`

**Request:**
```json
{
  "board": [
    [".", ".", ".", ".", ".", ".", ".", "."],
    [".", ".", ".", ".", ".", ".", ".", "."],
    [".", ".", ".", ".", ".", ".", ".", "."],
    [".", ".", ".", ".", "X", ".", ".", "."],
    [".", ".", ".", ".", "O", ".", ".", "."],
    [".", ".", ".", ".", ".", ".", ".", "."],
    [".", ".", ".", ".", ".", ".", ".", "."],
    [".", ".", ".", ".", ".", ".", ".", "."]
  ],
  "current_player": "X",
  "board_size": 8,
  "win_condition": 5,
  "compute_budget": 10000,
  "weight_c": 5.0,
  "move_history": [
    {"row": 3, "col": 4, "player": "X"},
    {"row": 4, "col": 4, "player": "O"}
  ]
}
```

**Response:**
```json
{
  "row": 2,
  "col": 4,
  "computation_time": 1.23,
  "confidence": 0.85
}
```

### Parameters

- `board`: 2D array with "X", "O", "." 
- `current_player`: "X" or "O"
- `board_size`: Board dimensions (default: 15)
- `win_condition`: Stones needed to win (default: 5)
- `compute_budget`: MCTS simulations (default: 10000)
- `weight_c`: MCTS exploration parameter (default: 5.0)
- `move_history`: Optional move history for context

## Python Client

```python
from client import MCTSPlayerClient

client = MCTSPlayerClient("http://localhost:8000")

# Simple usage
board = [["." for _ in range(8)] for _ in range(8)]
board[4][4] = "X"

result = client.get_move(
    board=board,
    current_player="O",
    compute_budget=5000
)

print(f"Best move: ({result['row']}, {result['col']})")
print(f"Computed in: {result['computation_time']:.2f}s")
print(f"Confidence: {result['confidence']:.1%}")
```

## Demo

```bash
python client.py
# Choose option 1 for interactive game
# Choose option 2 for batch testing
```

## cURL Examples

**Health check:**
```bash
curl http://localhost:8000/health
```

**Get move for empty board:**
```bash
curl -X POST "http://localhost:8000/get-move" \
  -H "Content-Type: application/json" \
  -d '{
    "board": [
      [".", ".", ".", ".", ".", ".", ".", "."],
      [".", ".", ".", ".", ".", ".", ".", "."],
      [".", ".", ".", ".", ".", ".", ".", "."],
      [".", ".", ".", ".", ".", ".", ".", "."],
      [".", ".", ".", ".", ".", ".", ".", "."],
      [".", ".", ".", ".", ".", ".", ".", "."],
      [".", ".", ".", ".", ".", ".", ".", "."],
      [".", ".", ".", ".", ".", ".", ".", "."]
    ],
    "current_player": "X",
    "board_size": 8,
    "compute_budget": 5000
  }'
```

## Performance

- **Compute Budget**: Controls AI strength vs speed
  - 1000: ~0.2s, fast but weaker
  - 10000: ~2s, good balance  
  - 50000: ~10s, strong play
- **Stateless**: Each request is independent
- **Concurrent**: FastAPI handles multiple requests

## Integration Examples

### Game Engine Integration
```python
def get_ai_move(game_state):
    client = MCTSPlayerClient()
    
    result = client.get_move(
        board=game_state.board,
        current_player=game_state.current_player,
        board_size=game_state.size,
        compute_budget=10000
    )
    
    return (result["row"], result["col"])
```

### Web Game Integration
```javascript
async function getAIMove(board, currentPlayer) {
    const response = await fetch('/get-move', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
            board: board,
            current_player: currentPlayer,
            compute_budget: 8000
        })
    });
    
    const result = await response.json();
    return {row: result.row, col: result.col};
}
```

## Docker

```bash
docker build -f Dockerfile -t mcts-player .
docker run -p 8000:8000 mcts-player
```

## Error Responses

- `400`: Invalid board state, game over, no legal moves
- `500`: MCTS computation error

The service validates input and provides clear error messages for debugging.