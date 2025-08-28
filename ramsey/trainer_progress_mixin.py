# ramsey/trainer_progress_mixin.py
from __future__ import annotations
from typing import Dict, Tuple
from collections import defaultdict
import os
import numpy as np

from .progress import ProgressTracker, Console, HAVE_RICH
from .conjecture import ConjectureTracker, KNOWN_EXACT

try:
    from rich.panel import Panel
except Exception:
    Panel = None


def _np_array(x):
    import numpy as _np
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return _np.asarray(x)


def _pretty_panel(txt: str, style: str = "green"):
    if HAVE_RICH and Panel is not None:
        try:
            Console().print(Panel.fit(txt, border_style=style))
            return
        except Exception:
            pass
    print(txt)


def attach_progress(trainer, window: int = 200, stall_episodes_for_conjecture: int = 300):
    """
    Attach episode-end progress + conjecture/equality tracking + optional
    early-exit at known (or conjectured) equality boundaries.

    Environment toggles (optional):
      - STOP_AT_KNOWN_EQUALITY=1|0          (default 1)
      - KNOWN_EQUALITY_CAP=EPISODES         (default 50)
      - STOP_AT_CONJECTURED_EQUALITY=1|0    (default 0)
      - STALL_EPISODES_FOR_CONJECTURE=EPIS  (default = stall_episodes_for_conjecture)
    """
    console = Console() if HAVE_RICH else None
    tracker = ProgressTracker(window=window, console=console)
    conject = ConjectureTracker(results_dir=trainer.cfg.results_dir,
                                stall_episodes=stall_episodes_for_conjecture)

    # Early-exit controls
    stop_at_known = os.environ.get("STOP_AT_KNOWN_EQUALITY", "1") != "0"
    cap_known = int(os.environ.get("KNOWN_EQUALITY_CAP", "50"))
    stop_at_conj = os.environ.get("STOP_AT_CONJECTURED_EQUALITY", "0") == "1"
    stall_conj = int(os.environ.get("STALL_EPISODES_FOR_CONJECTURE", str(stall_episodes_for_conjecture)))

    boundary_counts: Dict[Tuple[int, int], int] = defaultdict(int)

    orig_play = trainer.play_one_episode

    def wrapped_play_one_episode(n: int):
        result = orig_play(n)

        env = getattr(trainer, "env", None)
        cfg = getattr(trainer, "cfg", None)
        if env is not None and cfg is not None:
            used_sat = bool(getattr(env, "_last_sat_called", False))
            sat_success = getattr(env, "_last_sat_succeeded", None)

            # progress bar update
            try:
                tracker.update_from_env(r=int(cfg.r), env=env, used_sat=used_sat, sat_success=sat_success)
            except Exception as e:
                print(f"[progress] update skipped: {e}")

            # conjecture/equality tracker
            try:
                E = getattr(env, "edge_color", None)
                if E is not None:
                    E_np = _np_array(E)
                    total = E_np.shape[0] * (E_np.shape[0]-1) // 2
                    colored = int(np.triu((E_np > 0).astype(np.int32), k=1).sum())
                    len_frac = colored / max(1, total)
                else:
                    len_frac = 0.0
                rwd = float(getattr(env, "last_reward", 0.0))
                draw = (rwd > 0.0)
                conject.update(r=int(cfg.r), n=int(getattr(env, "n", 0)),
                               draw=draw, len_frac=float(len_frac))
            except Exception as e:
                print(f"[conjecture] update skipped: {e}")

            # optional early exit logic
            try:
                r_now = int(cfg.r)
                n_now = int(getattr(env, "n", 0))

                R_exact = KNOWN_EXACT.get(r_now)
                if stop_at_known and R_exact is not None and n_now >= R_exact:
                    key = (r_now, n_now)
                    boundary_counts[key] += 1
                    if boundary_counts[key] >= cap_known:
                        _pretty_panel(
                            f"Stopping at known boundary: R({r_now},{r_now}) = {R_exact}\n"
                            f"Spent {boundary_counts[key]} episodes at n={n_now} with no draws possible.",
                            style="green",
                        )
                        raise SystemExit(0)

                if stop_at_conj:
                    best = conject.state.best_draw_n.get(r_now, -1)
                    zstreak = conject.state.zero_draw_streak.get((r_now, n_now), 0)
                    if best >= 0 and n_now == best + 1 and zstreak >= stall_conj:
                        _pretty_panel(
                            f"Stopping at conjectured boundary: R({r_now},{r_now}) ≈ {n_now + 1}\n"
                            f"No draws for {zstreak}+ episodes at n={n_now}.",
                            style="yellow",
                        )
                        raise SystemExit(0)

            except SystemExit:
                raise
            except Exception as e:
                print(f"[early-exit] check skipped: {e}")

        return result

    trainer.play_one_episode = wrapped_play_one_episode  # type: ignore[attr-defined]
    return trainer
