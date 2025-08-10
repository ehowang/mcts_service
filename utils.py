"""
Utility classes and functions for MCTS.
"""
import sys
import time
from typing import Optional


class ProgressBar:
    """
    Simple progress bar for MCTS simulations.
    """
    
    def __init__(self, total: int, total_sharp: int = 20):
        """
        Initialize progress bar.
        
        Args:
            total: Total number of iterations
            total_sharp: Number of # symbols to display
        """
        self.total = total
        self.total_sharp = total_sharp
        self.current = 0
        self.start_time = time.time()
        self._last_update_time = self.start_time
        self._update_interval = 0.1  # Update at most every 0.1 seconds
    
    def iterStart(self):
        """Mark the start of an iteration."""
        self.current += 1
    
    def iterEnd(self):
        """Mark the end of an iteration and update display."""
        current_time = time.time()
        
        # Only update display if enough time has passed
        if current_time - self._last_update_time < self._update_interval and self.current < self.total:
            return
            
        self._last_update_time = current_time
        
        # Calculate progress
        progress = self.current / self.total
        sharp_count = int(progress * self.total_sharp)
        
        # Calculate time estimates
        elapsed = current_time - self.start_time
        if self.current > 0:
            avg_time = elapsed / self.current
            remaining = avg_time * (self.total - self.current)
        else:
            remaining = 0
        
        # Format time strings
        elapsed_str = self._format_time(elapsed)
        remaining_str = self._format_time(remaining)
        
        # Build progress bar
        bar = '#' * sharp_count + '-' * (self.total_sharp - sharp_count)
        
        # Display
        sys.stdout.write(f'\r[{bar}] {self.current}/{self.total} '
                        f'({progress*100:.1f}%) '
                        f'Elapsed: {elapsed_str} '
                        f'Remaining: {remaining_str}')
        sys.stdout.flush()
        
        if self.current == self.total:
            print()  # New line when complete
    
    def _format_time(self, seconds: float) -> str:
        """Format seconds into human-readable string."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = int(seconds / 60)
            secs = seconds % 60
            return f"{minutes}m{secs:.0f}s"
        else:
            hours = int(seconds / 3600)
            minutes = int((seconds % 3600) / 60)
            return f"{hours}h{minutes}m"
