# ramsey/conjecture.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional
from pathlib import Path
import time
import threading
import sys

try:
    from rich.console import Console
    from rich.panel import Panel
    HAVE_RICH = True
except Exception:
    Console = None
    Panel = None
    HAVE_RICH = False

# Global print lock so progress lines and panels don't interleave
_PRINT_LOCK = threading.Lock()

# Known exact diagonal Ramsey numbers we want to treat specially
KNOWN_EXACT = {
    3: 6,
    4: 18,
    # 5 and 6 unknown
}

@dataclass
class ConjectureState:
    # best n (largest) with a verified draw so far, per r
    best_draw_n: Dict[int, int] = field(default_factory=dict)
    # episodes in a row with zero draws at (r, n)
    zero_draw_streak: Dict[Tuple[int, int], int] = field(default_factory=dict)
    # last time we printed a non-witness panel for (r, n) to avoid spam
    last_hint_ts: Dict[Tuple[int, int], float] = field(default_factory=dict)
    # avoid excessive CONJECTURE.md rewrites
    last_digest: str = ""
    # de-dup same witness spam (within short window)
    last_witness_key: Optional[Tuple[int, int]] = None
    last_witness_ts: float = 0.0

class ConjectureTracker:
    """
    Tracks lower bounds and equality candidates during training.
    Prints a panel for EVERY new witness (new best n), and prints equality hints
    with a small cooldown.
    """
    def __init__(self,
                 results_dir: str,
                 stall_episodes: int = 300,
                 hint_cooldown_sec: int = 30):
        self.results_dir = Path(results_dir)
        self.state = ConjectureState()
        self.stall_episodes = int(stall_episodes)
        self.hint_cooldown = int(hint_cooldown_sec)
        self.console = Console() if HAVE_RICH else None

        (self.results_dir / "witness").mkdir(parents=True, exist_ok=True)
        self.md_path = self.results_dir / "CONJECTURE.md"

    # ----------------------------- API -----------------------------

    def update(self, r: int, n: int, draw: bool, len_frac: float):
        """
        Call once per episode at terminal. Emits:
        - Immediate "Conjecture update" panel on EVERY new witness (new best n).
        - Equality (known) or equality (conjecture) panels with cooldown.
        - Keeps CONJECTURE.md up to date.
        """
        key = (r, n)
        if draw:
            self.state.zero_draw_streak[key] = 0
            prev_best = self.state.best_draw_n.get(r, -1)
            if n > prev_best:
                self.state.best_draw_n[r] = n
                # Dedup if the same witness got reported twice in quick succession
                now = time.time()
                if self.state.last_witness_key != key or (now - self.state.last_witness_ts) > 2.0:
                    self._announce_witness(r, n)
                    self.state.last_witness_key = key
                    self.state.last_witness_ts = now
                self._write_md(r, n, lower_bound=n + 1)
        else:
            self.state.zero_draw_streak[key] = self.state.zero_draw_streak.get(key, 0) + 1

        # Equality panels / hints (cooldown-controlled)
        self._maybe_print_equality_or_hint(r, n)

    # ------------------------ witnesses & panels ------------------------

    def _announce_witness(self, r: int, n: int):
        """
        Print a visible panel every time a new best witness appears.
        Try to display the most recent witness PNG path if present.
        """
        lb = n + 1
        png_hint = self._find_latest_witness_png()
        body_lines = [
            f"Conjecture update: R({r},{r}) ≥ {lb}  (new witness)",
        ]
        if png_hint is not None:
            body_lines.append(f"saved → {png_hint}")
        body = "\n".join(body_lines)
        self._print_panel(body, title="Witness Found", style="cyan")

    def _maybe_print_equality_or_hint(self, r: int, n: int):
        """
        Print an equality panel (known) or a conjectured-equality hint
        when we appear stuck at n = best+1 for many episodes.
        """
        now = time.time()
        key = (r, n)
        last = self.state.last_hint_ts.get(key, 0.0)
        if now - last < self.hint_cooldown:
            return

        # Known equality?
        if r in KNOWN_EXACT:
            R = KNOWN_EXACT[r]
            if n + 1 == R:
                self._print_panel(
                    "Equality (known)\n"
                    f"R({r},{r}) = {R}\n"
                    f"Draws exist up to n = {R-1}; no draws at n = {R}.",
                    title="Equality (known)",
                    style="green",
                )
                self._write_md(r, n, known_exact=R)
                self.state.last_hint_ts[key] = now
                return

        # Unknown equality → conjectured via stall
        best = self.state.best_draw_n.get(r, -1)
        zstreak = self.state.zero_draw_streak.get(key, 0)
        if best >= 0 and n == best + 1 and zstreak >= self.stall_episodes:
            self._print_panel(
                "Equality likely (conjecture)\n"
                f"R({r},{r}) ≈ {n+1}\n"
                f"No draws observed at n = {n} for {zstreak}+ episodes.\n"
                f"Current lower bound: R({r},{r}) ≥ {best+1}",
                title="Equality (conjecture)",
                style="yellow",
            )
            self._write_md(r, n, conjectured=n + 1, lower_bound=best + 1)
            self.state.last_hint_ts[key] = now

    # --------------------------- rendering ---------------------------

    def _print_panel(self, body: str, title: str = "Conjecture", style: str = "cyan"):
        # Ensure panels land on a clean line and are not interleaved
        with _PRINT_LOCK:
            try:
                sys.stdout.write("\n")
                sys.stdout.flush()
            except Exception:
                pass
            if self.console is not None and Panel is not None:
                try:
                    self.console.print(Panel.fit(body, title=title, border_style=style))
                except Exception:
                    print(f"==== {title} ====\n{body}")
            else:
                print(f"==== {title} ====\n{body}")
            try:
                sys.stdout.write("\n")
                sys.stdout.flush()
            except Exception:
                pass

    def _find_latest_witness_png(self) -> Optional[str]:
        try:
            wdir = self.results_dir / "witness"
            pngs = sorted(wdir.glob("*.png"), key=lambda p: p.stat().st_mtime, reverse=True)
            return str(pngs[0]) if pngs else None
        except Exception:
            return None

    # ----------------------------- I/O -----------------------------

    def _write_md(self, r: int, n: int,
                  known_exact: Optional[int] = None,
                  conjectured: Optional[int] = None,
                  lower_bound: Optional[int] = None):
        lines = ["# Ramsey Conjecture Tracker", ""]
        lines.append(f"- **Current r:** {r}")
        lines.append(f"- **Current n:** {n}")
        lines.append("")
        if known_exact is not None:
            lines.append(f"## Equality (known)")
            lines.append(f"- **R({r},{r}) = {known_exact}** — at n={known_exact} no draws exist; at n={known_exact-1} draws exist.")
        elif conjectured is not None:
            lines.append(f"## Equality (conjecture)")
            lines.append(f"- **R({r},{r}) ≈ {conjectured}** — prolonged lack of draws at n={conjectured-1} + 1.")
        if lower_bound is not None:
            lines.append("")
            lines.append("## Lower bound")
            lines.append(f"- **R({r},{r}) ≥ {lower_bound}** — witness at n={lower_bound-1}.")
        # Reference table
        if KNOWN_EXACT:
            lines.append("")
            lines.append("## Reference (known exact values)")
            for rr, exact in sorted(KNOWN_EXACT.items()):
                lines.append(f"- R({rr},{rr}) = {exact}")
        content = "\n".join(lines) + "\n"

        if content != self.state.last_digest:
            self.md_path.write_text(content)
            self.state.last_digest = content
