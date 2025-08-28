#!/usr/bin/env python3
"""
Lightweight r-curriculum scheduler that runs main.py in separate stages,
auto-resuming checkpoints between stages from the SAME results_dir.

Example:
    python scripts/run_r_curriculum.py --sequence 3,4,5 \
        --results_dir ./results_curr \
        --minutes 10,20,60 \
        --profile unstick
"""
import os
import sys
import time
import shlex
import signal
import subprocess
from pathlib import Path
import argparse


def parse_int_list(s: str):
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def profile_defaults(r: int, profile: str):
    if profile == "balanced":
        return dict(root_noise_frac=0.30, widen_c=8, widen_alpha=0.50, lambda_shape=0.10 if r >= 5 else 0.06)
    # unstick
    return dict(root_noise_frac=(0.55 if r <= 4 else 0.45),
                widen_c=(24 if r <= 4 else 16),
                widen_alpha=(0.70 if r <= 4 else 0.60),
                lambda_shape=(0.03 if r <= 4 else 0.05))


def run_stage(python, main_py, device, results_dir, r, n_max, start_n, mcts_sims,
              mins, profile, sat_gate, sat_ms, sat_calls, resume):
    knobs = profile_defaults(r, profile)
    cmd = [
        python, "-u", main_py,
        "--device", device,
        "--results_dir", results_dir,
        "--r", str(r),
        "--n_max", str(n_max),
        "--curriculum_start_n", str(start_n),
        "--mcts_sims", str(mcts_sims),
        "--root_noise_frac", str(knobs["root_noise_frac"]),
        "--widen_c", str(knobs["widen_c"]),
        "--widen_alpha", str(knobs["widen_alpha"]),
        "--lambda_shape", str(knobs["lambda_shape"]),
        "--sat_ms", str(sat_ms),
        "--sat_calls_per_game", str(sat_calls),
        "--resume", "1" if resume else "0",
    ]
    # per-r SAT gate
    if r == 3: cmd += ["--sat_edges_left_3", str(sat_gate)]
    elif r == 4: cmd += ["--sat_edges_left_4", str(sat_gate)]
    elif r == 5: cmd += ["--sat_edges_left_5", str(sat_gate)]
    else: cmd += ["--sat_edges_left_6", str(sat_gate)]

    env = os.environ.copy()
    env.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    # Helpful on macs with OpenMP conflicts
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    env.setdefault("OMP_NUM_THREADS", "4")

    print(f"\n=== Stage r={r} | {mins} min | profile={profile} | resume={resume} ===")
    print(" ".join(shlex.quote(x) for x in cmd))
    proc = subprocess.Popen(cmd, env=env)

    # timed run
    deadline = time.time() + mins * 60
    ret = None
    try:
        while time.time() < deadline:
            ret = proc.poll()
            if ret is not None:
                break
            time.sleep(1)
        if ret is None:
            # timeout: terminate then kill if needed
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
    except KeyboardInterrupt:
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except Exception:
            pass
        raise

    # return code (None means forced)
    return proc.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sequence", type=str, default="3,4,5", help="Comma list of r values, e.g. 3,4,5 or 4,5")
    ap.add_argument("--minutes", type=str, default="10,20,60", help="Comma list of minutes per stage (same length or one value to broadcast)")
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--results_dir", type=str, default="./results_curr")
    ap.add_argument("--n_max", type=int, default=150)
    ap.add_argument("--start_ns", type=str, default="", help="Comma list of start n per stage; blank=defaults")
    ap.add_argument("--mcts_sims", type=int, default=96)
    ap.add_argument("--profile", type=str, choices=["balanced", "unstick"], default="unstick")
    ap.add_argument("--sat_gate", type=int, default=-1, help="Override per-r SAT gate; -1 = use r defaults")
    ap.add_argument("--sat_ms", type=int, default=5000)
    ap.add_argument("--sat_calls", type=int, default=2)
    args = ap.parse_args()

    seq = parse_int_list(args.sequence)
    mins = parse_int_list(args.minutes)
    if len(mins) == 1:
        mins = mins * len(seq)
    if len(mins) != len(seq):
        raise SystemExit("--minutes must be one value or match --sequence length")

    # defaults per r for start n and SAT gates
    default_start = {3: 4, 4: 8, 5: 8, 6: 36}
    default_gate = {3: 120, 4: 120, 5: 100, 6: 60}

    start_ns = parse_int_list(args.start_ns) if args.start_ns else []
    if start_ns and len(start_ns) != len(seq):
        raise SystemExit("--start_ns must be empty or match --sequence length")

    python = sys.executable
    main_py = str(Path(__file__).resolve().parents[1] / "main.py")

    results_dir = args.results_dir
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    # stage loop
    for idx, r in enumerate(seq):
        start_n = start_ns[idx] if start_ns else default_start.get(r, 8)
        sat_gate = args.sat_gate if args.sat_gate >= 0 else default_gate.get(r, 60)
        resume = (idx > 0)  # resume from prior stage checkpoint(s)

        rc = run_stage(
            python=python,
            main_py=main_py,
            device=args.device,
            results_dir=results_dir,
            r=r,
            n_max=args.n_max,
            start_n=start_n,
            mcts_sims=args.mcts_sims,
            mins=mins[idx],
            profile=args.profile,
            sat_gate=sat_gate,
            sat_ms=args.sat_ms,
            sat_calls=args.sat_calls,
            resume=resume,
        )
        print(f"Stage r={r} finished with return code: {rc}")

    print("All stages done.")


if __name__ == "__main__":
    main()
