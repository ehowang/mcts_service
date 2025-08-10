FROM python:3.11-slim

WORKDIR /app

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire gomoku project
COPY .. /app/

# Set Python path
ENV PYTHONPATH="/app:/app/gomoku-ai"

# Expose port
EXPOSE 8000

# Change to service directory
WORKDIR /app/mcts_service

# Run the service
CMD ["python", "main.py"]
