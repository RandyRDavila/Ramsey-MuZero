# ramsey/progress.py
from __future__ import annotations
from dataclasses import dataclass, field
import time
import threading
import sys

try:
    from rich.console import Console
    HAVE_RICH = True
except Exception:
    Console = None
    HAVE_RICH = False

# Shared lock with panels to avoid interleaving
_PRINT_LOCK = threading.Lock()

@dataclass
class ProgressState:
    last_line_ts: float = 0.0
    interval: float = 0.5  # seconds between prints
    window: int = 200
    # stats
    draws_in_window: int = 0
    episodes_in_window: int = 0
    avg_len_frac: float = 0.0

class ProgressTracker:
    """
    Lightweight episodic progress printer. Always prints whole lines (never '\r'),
    so it won't collide with Rich panels even when multiple threads write.
    """
    def __init__(self, window: int = 200, console: Console | None = None):
        self.state = ProgressState(window=window)
        self.console = console if (HAVE_RICH and console is not None) else None

    def _should_print(self) -> bool:
        now = time.time()
        if (now - self.state.last_line_ts) >= self.state.interval:
            self.state.last_line_ts = now
            return True
        return False

    def update_from_env(self, r: int, env, used_sat: bool, sat_success: bool | None):
        """
        Call once per episode end to update and optionally print a line.
        """
        # Update running stats
        self.state.episodes_in_window += 1
        if float(getattr(env, "last_reward", 0.0)) > 0.0:
            self.state.draws_in_window += 1

        # estimate color progress (fraction of colored edges)
        try:
            import numpy as np
            E = getattr(env, "edge_color", None)
            if E is not None:
                En = E if isinstance(E, np.ndarray) else E.cpu().numpy()
                total = En.shape[0] * (En.shape[0] - 1) // 2
                colored = int(np.triu((En > 0).astype(np.int32), k=1).sum())
                len_frac = colored / max(1, total)
            else:
                len_frac = 0.0
        except Exception:
            len_frac = 0.0

        # simple moving avg by window
        w = max(1, self.state.window)
        self.state.avg_len_frac = (self.state.avg_len_frac * (w - 1) + len_frac) / w

        if not self._should_print():
            return

        # Compose a full line; print with newline and lock
        conf = max(0.0, min(1.0, self.state.draws_in_window / max(1, self.state.episodes_in_window)))
        msg = (
            f"n={getattr(env,'n',0)} r={r} ━ "
            f"conf: {conf*100:5.1f}% | draw: {conf*100:5.1f}% | len: {self.state.avg_len_frac*100:5.1f}%"
        )

        with _PRINT_LOCK:
            if self.console is not None:
                try:
                    self.console.print(msg, soft_wrap=True)
                except Exception:
                    print(msg)
            else:
                print(msg)
            sys.stdout.flush()
