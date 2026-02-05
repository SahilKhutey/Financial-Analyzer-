"""
Step 27: Performance Profiler.
Measures execution time of critical components.
"""
import time
from dataclasses import dataclass, field
from typing import Dict

@dataclass
class LatencyStats:
    timings: Dict[str, float] = field(default_factory=dict)
    
    def add(self, name: str, ms: float):
        self.timings[name] = ms
        
    def get_display_data(self):
        return self.timings

class Profiler:
    """
    Context Manager for timing code blocks.
    Usage:
        with Profiler("Vision", stats_obj):
            ...
    """
    def __init__(self, name: str, stats: LatencyStats):
        self.name = name
        self.stats = stats
        self.start = 0
        
    def __enter__(self):
        self.start = time.perf_counter()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = (time.perf_counter() - self.start) * 1000 # to ms
        self.stats.add(self.name, elapsed)
