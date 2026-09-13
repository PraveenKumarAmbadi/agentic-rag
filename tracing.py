from dataclasses import dataclass, field
from contextlib import contextmanager
import time
import logging
import json
import threading
from datetime import datetime

@dataclass()
class TraceContext:
    trace_id: str
    start_time: float 
    timings: dict[str, float] = field(default_factory=dict)

@contextmanager
def track_step(ctx: TraceContext, step_name: str):
    start_time = time.perf_counter()
    try:
        yield
    finally:
        ctx.timings[step_name] = (time.perf_counter() - start_time) * 1000

class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'message': record.getMessage(),
            # Safely fetch extra attributes, defaulting to None if not provided
            'trace_id': getattr(record, 'trace_id', None),
            'timings': getattr(record, 'timings', None),
        }

        # Optional: Remove keys with None values to keep JSON clean
        log_entry = {k: v for k, v in log_entry.items() if v is not None}

        return json.dumps(log_entry)

def setup_logger(name: str, formatter: logging.Formatter, level=logging.INFO, handler=None) -> logging.Logger:
    logger = logging.getLogger(name=name)
    if logger.hasHandlers():
        logger.handlers.clear()
    
    if handler is None:
        handler = logging.StreamHandler()
    
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(level)
    return logger


class CircuitBreaker():
    def __init__(self, failure_threshold: int = 3,
                 cooldown_period: float = 60.0
                 ):
      self.failure_threshold = failure_threshold
      self.cooldown_period = cooldown_period
      self.failure_count = 0
      self.last_failure_time = 0.0
      self.state = 'CLOSED'
      self._lock = threading.Lock()

    def __enter__(self):
      failure_time = (time.perf_counter() - self.last_failure_time)
      if self.state == 'OPEN':
        if failure_time >= self.cooldown_period:
          self.state = 'HALF_OPEN'
        else:
          raise RuntimeError(f'Circuit is OPEN please wait {self.cooldown_period - failure_time} seconds')

    def __exit__(self, exc_type, exc_val, exc_tb):
      with self._lock:
        if exc_type:
            self.last_failure_time = time.perf_counter()
            self.failure_count += 1
        else:
            self.failure_count = 0
            self.last_failure_time = 0.0
            self.state = 'CLOSED'

        if self.failure_count >= self.failure_threshold:
            self.state = 'OPEN'

_defaultFormatter = JsonFormatter()

class MemoryLogHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.log_records = []
        self.setFormatter(_defaultFormatter)

    def emit(self, record):
        try:
            message = self.formatter.format(record)
            json_message = json.loads(message)
            self.log_records.append(json_message)
        except Exception:
            self.handleError(record)

    def get_logs(self):
        return self.log_records