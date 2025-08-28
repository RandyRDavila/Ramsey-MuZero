
import os
import argparse
from pathlib import Path
import random

import numpy as np
import torch

from ramsey.config import RamseyConfig
from ramsey.env import RamseyEnv
from ramsey.trainer import Trainer
from ramsey.model import MuZeroNetwork
from ramsey.checkpoint import find_latest_checkpoint, load_model_state_safely


def detect_device(arg: str) -> torch.device:
    if arg and arg.lower() != "auto":
        return torch.device(arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()

    # Core run config
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--results_dir", type=str, default="./results")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", type=int, default=0, help="Load latest checkpoint if available (0/1)")

    # Game / env
    ap.add_argument("--n_max", type=int, default=43)
    ap.add_argument("--r", type=int, choices=[3, 4, 5, 6], default=5)

    # Network / MuZero
    ap.add_argument("--network", type=str, default="gnn")
    ap.add_argument("--support_S", type=int, default=10)

    # GNN encoder
    ap.add_argument("--gnn_hidden", type=int, default=128)
    ap.add_argument("--gnn_layers", type=int, default=3)
    ap.add_argument("--latent_dim", type=int, default=256)
    ap.add_argument("--action_emb", type=int, default=64)

    # MCTS / search
    ap.add_argument("--mcts_sims", type=int, default=96)
    ap.add_argument("--root_noise_frac", type=float, default=0.30)
    ap.add_argument("--widen_c", type=int, default=8)
    ap.add_argument("--widen_alpha", type=float, default=0.5)

    # Training
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--unroll_steps", type=int, default=5)
    ap.add_argument("--td_steps", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--grad_clip", type=float, default=10.0)
    ap.add_argument("--draws_needed", type=int, default=2)
    ap.add_argument("--curriculum_start_n", type=int, default=8)
    ap.add_argument("--lambda_shape", type=float, default=0.1)

    # SAT / CP-SAT
    ap.add_argument("--use_cpsat_fallback", type=int, default=0)
    ap.add_argument("--cpsat_ms", type=int, default=200)

    ap.add_argument("--sat_edges_left_3", type=int, default=120)
    ap.add_argument("--sat_edges_left_4", type=int, default=120)
    ap.add_argument("--sat_edges_left_5", type=int, default=100)
    ap.add_argument("--sat_edges_left_6", type=int, default=60)

    ap.add_argument("--sat_calls_per_game", type=int, default=2)
    ap.add_argument("--sat_penalty", type=float, default=0.05)  # magnitude; env applies negative
    ap.add_argument("--sat_ms", type=int, default=5000)
    ap.add_argument("--sat_clause_cap", type=int, default=200000)

    ap.add_argument("--tri_cache", type=int, default=1)

    # TensorBoard / terminal logging switches
    ap.add_argument("--tb_log_moves", type=int, default=0)
    ap.add_argument("--tb_log_every", type=int, default=10)
    ap.add_argument("--tb_log_max_moves", type=int, default=200)
    ap.add_argument("--tb_video_per_episode", type=int, default=0)
    ap.add_argument("--tb_video_fps", type=int, default=10)
    ap.add_argument("--tb_image_size", type=int, default=640)

    ap.add_argument("--term_log_moves", type=int, default=0)
    ap.add_argument("--term_log_every", type=int, default=10)
    ap.add_argument("--term_log_compact", type=int, default=1)

    # Conjecture print toggle
    ap.add_argument("--conjecture_print", type=int, default=1)

    return ap


def main():
    ap = build_argparser()
    args = ap.parse_args()

    device = detect_device(args.device)
    if device.type == "mps":
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    # Prepare result dirs
    rp = Path(args.results_dir)
    (rp / "checkpoints").mkdir(parents=True, exist_ok=True)
    (rp / "witness").mkdir(parents=True, exist_ok=True)
    (rp / "tb").mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    print(f"Using device: {device.type}")

    # Build config
    cfg = RamseyConfig(
        # core
        device=device,
        results_dir=args.results_dir,
        seed=args.seed,

        # env
        n_max=args.n_max,
        r=args.r,

        # network / muzero
        network=args.network,
        support_S=args.support_S,
        gnn_hidden=args.gnn_hidden,
        gnn_layers=args.gnn_layers,
        latent_dim=args.latent_dim,
        action_emb=args.action_emb,

        # mcts
        mcts_sims=args.mcts_sims,
        root_noise_frac=args.root_noise_frac,
        widen_c=args.widen_c,
        widen_alpha=args.widen_alpha,

        # train
        batch_size=args.batch_size,
        unroll_steps=args.unroll_steps,
        td_steps=args.td_steps,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        draws_needed=args.draws_needed,
        curriculum_start_n=args.curriculum_start_n,
        lambda_shape=args.lambda_shape,

        # sat
        use_cpsat_fallback=bool(args.use_cpsat_fallback),
        cpsat_ms=args.cpsat_ms,
        sat_edges_left_3=args.sat_edges_left_3,
        sat_edges_left_4=args.sat_edges_left_4,
        sat_edges_left_5=args.sat_edges_left_5,
        sat_edges_left_6=args.sat_edges_left_6,
        sat_calls_per_game=args.sat_calls_per_game,
        sat_penalty=args.sat_penalty,
        sat_ms=args.sat_ms,
        sat_clause_cap=args.sat_clause_cap,
        tri_cache=bool(args.tri_cache),

        # logging
        tb_log_moves=bool(args.tb_log_moves),
        tb_log_every=args.tb_log_every,
        tb_log_max_moves=args.tb_log_max_moves,
        tb_video_per_episode=bool(args.tb_video_per_episode),
        tb_video_fps=args.tb_video_fps,
        tb_image_size=args.tb_image_size,

        term_log_moves=bool(args.term_log_moves),
        term_log_every=args.term_log_every,
        term_log_compact=bool(args.term_log_compact),

        conjecture_print=bool(args.conjecture_print),
    )

    # Build environment and network
    env = RamseyEnv(cfg)
    net = MuZeroNetwork(cfg).to(device)

    # Optional resume
    if int(args.resume) == 1:
        latest = find_latest_checkpoint(Path(args.results_dir) / "checkpoints")
        if latest is not None:
            ok, msg = load_model_state_safely(net, latest, map_location=str(device))
            print(f"[resume] {('loaded ' + latest.name) if ok else 'failed load'} :: {msg}")

    # Kick off training
    trainer = Trainer(cfg, env, net)


    from ramsey.trainer_progress_mixin import attach_progress
    attach_progress(trainer, window=200)


    print(">> Collecting initial episodes...")
    trainer.train_forever()


if __name__ == "__main__":
    main()
