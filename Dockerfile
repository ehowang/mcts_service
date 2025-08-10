FROM python:3.12-slim

WORKDIR /app

# Copy requirements from the build context root (which is the parent dir)
COPY mcts_service/requirements.txt /app/requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy and install the gomoku-ai package
COPY gomoku-ai /app/gomoku-ai
RUN pip install --no-cache-dir /app/gomoku-ai

# Copy only the service directory to preserve the package name `mcts_service`
COPY mcts_service /app/mcts_service

# Ensure `import mcts_service.*` works (namespace package)
ENV PYTHONPATH="/app"

# Expose API port
EXPOSE 8000

# Run from inside the service directory so relative imports like `from models import ...` work
WORKDIR /app/mcts_service

CMD ["python", "main.py"]
