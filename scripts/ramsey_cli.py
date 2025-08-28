#!/usr/bin/env python3
# scripts/ramsey_cli.py
from __future__ import annotations

import os
import sys
import shutil
import textwrap
import subprocess
from pathlib import Path
from typing import List, Optional
from importlib.util import find_spec

# --- Third-party (nice-to-haves, with a clear error if missing) ---
try:
    import questionary
    from questionary import Style, Choice
    import pyfiglet
    from rich.console import Console
    from rich.panel import Panel
except Exception:
    print("Please install CLI extras first: pip install questionary pyfiglet rich")
    sys.exit(1)

console = Console()

# ---------- Questionary Style (high-contrast highlight + visible pointer) ----------
CLI_STYLE = Style([
    ("qmark",        "fg:#00d1d1 bold"),
    ("question",     "bold"),
    ("answer",       "fg:#00d1d1 bold"),
    ("pointer",      "fg:#00d1d1 bold"),
    ("highlighted",  "fg:#000000 bg:#00d1d1 bold"),  # only the active row is visible
    ("selected",     ""),                            # neutral, so no default row is styled
    ("instruction",  "fg:#808080"),
    ("text",         ""),
])


POINTER_GLYPH = "»"  # nice and obvious

# ---------- Helpers ----------
def _repo_root() -> Path:
    # repo root = parent of scripts/
    return Path(__file__).resolve().parents[1]

def purge_results_contents() -> None:
    path = _safe_text("Which results directory to EMPTY (keep folder)?", "./results")
    abs_path = os.path.abspath(path)

    # Guardrails
    forbidden = {"/", "/home", "/Users", os.path.expanduser("~")}
    if abs_path in forbidden:
        console.print("[red]Refusing to operate on a system/user root.[/red]")
        return
    if not os.path.isdir(abs_path):
        console.print(f"[yellow]{abs_path} does not exist or is not a directory.[/yellow]")
        return

    sure = _safe_confirm(f"DELETE ALL contents under {abs_path} (keep the folder)?", False)
    if not sure:
        console.print("[yellow]Cancelled.[/yellow]")
        return

    # Delete children only
    errors = []
    with os.scandir(abs_path) as it:
        for entry in it:
            try:
                p = os.path.join(abs_path, entry.name)
                if entry.is_symlink() or entry.is_file():
                    os.unlink(p)
                elif entry.is_dir(follow_symlinks=False):
                    shutil.rmtree(p)
            except Exception as e:
                errors.append((entry.name, str(e)))

    if errors:
        console.print("[red]Some items could not be removed:[/red]")
        for name, msg in errors:
            console.print(f"  • {name}: {msg}")
    else:
        console.print(f"[green]Emptied {abs_path}[/green]")


def _env_summary() -> None:
    py = sys.version.split()[0]
    plat = f"{sys.platform}"
    arch = ""
    try:
        arch = f" {os.uname().machine}"
    except Exception:
        pass
    try:
        import torch
        tver = torch.__version__
        cuda = torch.cuda.is_available()
        mps = getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
    except Exception:
        tver, cuda, mps = "N/A", False, False
    console.print(Panel.fit(
        f"[bold]Environment Summary[/bold]\n\n"
        f"  Python           {py}\n"
        f"  Platform         {plat}{arch}\n"
        f"  Default Device   {'mps' if mps else ('cuda' if cuda else 'cpu')}\n"
        f"  torch            {tver}\n"
        f"  CUDA available   {cuda}\n"
        f"  MPS available    {mps}",
        border_style="cyan"
    ))

def _set_common_env(env: dict, set_kmp: bool = True) -> None:
    # mac + OpenMP nicety
    if set_kmp and sys.platform == "darwin":
        env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    env.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

# --- Safe prompts with graceful fallback to numbered input if TTY/UI breaks ---
def _safe_select(message: str, choices: List[str]) -> str:
    try:
        return questionary.select(
            message,
            choices=choices,
            pointer=POINTER_GLYPH,
            style=CLI_STYLE,
            use_shortcuts=True,
            use_indicator=False,   # no radio dot
            qmark="?",
            instruction="(↑/↓ to move, Enter to select)"
        ).unsafe_ask()
    except Exception as e:
        console.print(f"[yellow]UI degraded ({e}). Falling back to basic input.[/yellow]")
        for i, c in enumerate(choices, 1):
            console.print(f"[{i}] {c}")
        while True:
            s = input("> ").strip()
            if s.isdigit() and 1 <= int(s) <= len(choices):
                return choices[int(s) - 1]
            console.print(f"[red]Choose 1..{len(choices)}[/red]")


def _safe_confirm(message: str, default: bool = True) -> bool:
    try:
        return questionary.confirm(
            message, default=default, style=CLI_STYLE
        ).unsafe_ask()
    except Exception:
        s = input(f"{message} [{'Y/n' if default else 'y/N'}]: ").strip().lower()
        if s == "":  return default
        if s in ("y", "yes"): return True
        if s in ("n", "no"):  return False
        return default

def _safe_text(message: str, default: Optional[str] = None) -> str:
    try:
        return questionary.text(
            message, default=default or "", style=CLI_STYLE
        ).unsafe_ask()
    except Exception:
        s = input(f"{message} [{default or ''}]: ").strip()
        return s if s else (default or "")

def _ask_int(msg: str, default: int, validate_min: Optional[int] = None) -> int:
    while True:
        v = _safe_text(msg, str(default))
        try:
            x = int(v)
            if validate_min is not None and x < validate_min:
                raise ValueError()
            return x
        except Exception:
            console.print(f"[red]Enter an integer ≥ {validate_min if validate_min is not None else '-∞'}[/red]")

def _ask_float(msg: str, default: float) -> float:
    while True:
        v = _safe_text(msg, str(default))
        try:
            return float(v)
        except Exception:
            console.print("[red]Enter a number[/red]")

# ---------- Trainer launcher (prefer module, fallback to file) ----------
def _trainer_cmd(base_args: List[str]) -> List[str]:
    py = sys.executable
    # Prefer python -m ramsey.main (package entry)
    if find_spec("ramsey") and (find_spec("ramsey.main") or find_spec("ramsey.__main__")):
        if find_spec("ramsey.main"):
            return [py, "-u", "-m", "ramsey.main", *base_args]
        else:
            return [py, "-u", "-m", "ramsey", *base_args]
    # Fallback: repo-root main.py
    main_py = _repo_root() / "main.py"
    if main_py.exists():
        return [py, "-u", str(main_py), *base_args]
    raise FileNotFoundError(
        "No trainer entry point found. Expected module 'ramsey.main' / 'ramsey' "
        "or a 'main.py' at repo root."
    )

def _print_run(title: str, base_args: List[str], env: dict | None = None) -> None:
    console.print(Panel.fit(f"[bold]Launching {title}[/bold]", border_style="magenta"))
    cmd = _trainer_cmd(base_args)
    console.print(" ".join(cmd))
    console.print("Press Ctrl+C to stop.\n")
    subprocess.run(cmd, env=env, cwd=_repo_root(), check=True)

# ---------- Actions ----------
def prompt_training_custom() -> None:
    device = _safe_select("Device?", ["auto (recommended)", "cpu", "cuda", "mps"])
    r = _safe_select("Clique size r ?", ["3", "4", "5", "6"])
    n_max = _ask_int("N_MAX (padding size)?", 43, 4)
    start_n = _ask_int("Start curriculum at n =", 8, 2)
    mcts = _ask_int("MCTS sims per move", 96, 1)
    profile = _safe_select("Exploration profile?", ["default", "unstick"])
    sat_ms = _ask_int("SAT time budget per call (ms)", 5000, 10)
    sat_calls = _ask_int("Max SAT calls per game", 2, 0)
    results_dir = _safe_text("Results directory", "./results")
    set_kmp = _safe_confirm("Set KMP_DUPLICATE_LIB_OK=TRUE during run? (fixes mac OpenMP conflicts)", True)

    # Default SAT gate by r (editable)
    if r in ("3", "4"):
        sat_gate_default = 120
    elif r == "5":
        sat_gate_default = 100
    else:
        sat_gate_default = 60
    sat_gate = _ask_int("SAT gate: expose SAT_TRY_NOW when edges_left ≤", sat_gate_default, 0)

    # Profile knobs
    r_int = int(r)
    if profile == "default":
        root_noise, widen_c, widen_a, lam = 0.30, 8, 0.50, (0.10 if r_int >= 5 else 0.06)
    else:
        root_noise, widen_c, widen_a, lam = 0.45, 16, 0.60, (0.05 if r_int >= 5 else 0.03)

    env = os.environ.copy()
    _set_common_env(env, set_kmp=set_kmp)

    args = [
        "--device", "auto" if device == "auto (recommended)" else device,
        "--r", r,
        "--n_max", str(n_max),
        "--curriculum_start_n", str(start_n),
        "--mcts_sims", str(mcts),
        "--results_dir", results_dir,
        "--root_noise_frac", str(root_noise),
        "--widen_c", str(widen_c),
        "--widen_alpha", str(widen_a),
        "--lambda_shape", str(lam),
        "--sat_ms", str(sat_ms),
        "--sat_calls_per_game", str(sat_calls),
        f"--sat_edges_left_{r}", str(sat_gate),
        "--term_log_moves", "1",
        "--term_log_every", "10",
    ]
    _print_run("Custom Trainer (ramsey/*)", args, env)


def quick_sat_smoke() -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = env.get("PYTHONPATH", ".")
    code = textwrap.dedent(
        """
        import numpy as np
        from ramsey.sat_tools import bounded_draw_completion
        n,r=12,5
        R=B=E=np.zeros((n,n),dtype=np.uint8)
        ok,RR,BB,meta = bounded_draw_completion(n,r,R,B,E,time_ms=500,clause_cap=10000)
        print("SAT ok:", ok, meta)
        """
    )
    subprocess.run([sys.executable, "-c", code], env=env, cwd=_repo_root(), check=True)

def clean_results() -> None:
    path = _safe_text("Which results directory to delete? (careful!)", "./results")
    sure = _safe_confirm(f"Delete EVERYTHING under {path} ?", False)
    if not sure:
        return
    abs_path = os.path.abspath(path)
    # guardrails
    if abs_path in ("/", "/home", "/Users"):
        console.print("[red]Refusing to delete a system root.[/red]")
        return
    if os.path.isdir(abs_path):
        shutil.rmtree(abs_path)
        console.print(f"[green]Deleted {abs_path}[/green]")
    else:
        console.print(f"[yellow]{abs_path} does not exist.[/yellow]")

# ---------- Main ----------
def main() -> None:
    console.print(pyfiglet.figlet_format("Ramsey MuZero", font="Speed"))
    _env_summary()

    try:
        while True:
            choice = _safe_select(
                "🧠 What do you want to do?",
                [
                    "Train (custom GNN trainer: ramsey/*)",
                    "Quick SAT smoke",
                    "Empty a results directory (keep folder)",
                    Choice(title="Exit", value="Exit", shortcut_key="q"),
                ],
            )

            if choice == "Train (custom GNN trainer: ramsey/*)":
                prompt_training_custom()
            elif choice == "Quick SAT smoke":
                quick_sat_smoke()
            elif choice == "Empty a results directory (keep folder)":
                purge_results_contents()
            elif choice == "Exit":
                break
    except KeyboardInterrupt:
        # Graceful Ctrl+C exit from anywhere
        console.print("\n[cyan]Goodbye![/cyan]")
        return

    console.print("[cyan]Goodbye![/cyan]")



if __name__ == "__main__":
    main()
