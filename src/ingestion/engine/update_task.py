from dataclasses import dataclass
from datetime import datetime
from typing import Literal

@dataclass
class UpdateTask:
    """Data class for stock update task
    
    Attributes:
        start (str): Start date in 'YYYY-MM-DD' format
        end (str): End date in 'YYYY-MM-DD' format
        direction (Literal['backward', 'forward', 'full']): Type of update
        
    Returns:
        UpdateTask: An instance with the specified attributes
        
    Example:
        >>> task = UpdateTask("2023-01-01", "2024-01-01", "forward")
        >>> print(task.start)
        '2023-01-01'
    """
    start: str
    end: str
    direction: Literal['backward', 'forward', 'full']

    def __post_init__(self):
        """Validate dates after initialization"""
        try:
            start_date = datetime.strptime(self.start, '%Y-%m-%d')
            end_date = datetime.strptime(self.end, '%Y-%m-%d')
            if start_date >= end_date:
                raise ValueError(f"Start date {self.start} must be before end date {self.end}")
        except ValueError as e:
            raise ValueError(f"Invalid date format or {str(e)}") 