# from dataclasses import dataclass
# from typing import Optional
# import torch


# @dataclass
# class RamseyConfig:
#     n_max: int = 43
#     r: int = 5
#     device: torch.device = torch.device("cpu")

#     # training
#     mcts_sims: int = 96
#     batch_size: int = 128
#     unroll_steps: int = 5
#     td_steps: int = 10
#     draws_needed: int = 2
#     curriculum_start_n: int = 8
#     lr: float = 1e-3
#     weight_decay: float = 1e-4
#     grad_clip: float = 10.0

#     # distributional support
#     support_S: int = 10

#     # MCTS
#     root_noise_frac: float = 0.30
#     widen_c: float = 8.0
#     widen_alpha: float = 0.5

#     # shaping
#     lambda_shape: float = 0.1

#     # SAT / CP-SAT
#     use_cpsat_fallback: bool = False
#     cpsat_ms: int = 200
#     sat_edges_left_3: int = 120   # r=3 – triangles are easy; SAT near end only
#     sat_edges_left_4: int = 120   # r=4 – K4 is modest; keep late for speed
#     sat_edges_left_5: int = 100
#     sat_edges_left_6: int = 60
#     sat_calls_per_game: int = 2
#     sat_penalty: float = 0.05
#     sat_ms: int = 5000
#     sat_clause_cap: int = 200000

#     # perf
#     tri_cache: bool = False

#     # network choice
#     network: str = "gnn"

#     # io
#     results_dir: str = "./results"

#     # TensorBoard move/video logging
#     tb_log_moves: bool = False          # log images during play
#     tb_log_every: int = 10              # log every k moves
#     tb_log_max_moves: int = 200         # cap images per episode
#     tb_video_per_episode: bool = False  # also log a short video per episode
#     tb_video_fps: int = 10              # video FPS if enabled
#     tb_image_size: int = 640            # image side in pixels

#     conjecture_print: bool = True
from dataclasses import dataclass
from typing import Any


@dataclass
class RamseyConfig:
    # Core / paths / device
    device: Any = None
    results_dir: str = "./results"
    seed: int = 0

    # Game / env
    n_max: int = 43
    r: int = 5
    lambda_shape: float = 0.1  # shaping weight (non-draw steps only)

    # MuZero / network
    network: str = "gnn"  # factory key picked by models.py / ramsey/model.py
    support_S: int = 10   # distributional support half-range

    # GNN details (used by the GNN encoder in ramsey/model.py)
    gnn_hidden: int = 128
    gnn_layers: int = 3
    latent_dim: int = 256
    action_emb: int = 64

    # MCTS / search
    mcts_sims: int = 96
    root_noise_frac: float = 0.30
    widen_c: int = 8
    widen_alpha: float = 0.5

    # Training
    batch_size: int = 128
    unroll_steps: int = 5
    td_steps: int = 10
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 10.0
    draws_needed: int = 2
    curriculum_start_n: int = 8

    # SAT / CP-SAT
    use_cpsat_fallback: bool = False
    cpsat_ms: int = 200

    # SAT gates per r
    sat_edges_left_3: int = 120
    sat_edges_left_4: int = 120
    sat_edges_left_5: int = 100
    sat_edges_left_6: int = 60

    sat_calls_per_game: int = 2
    sat_penalty: float = 0.05  # magnitude; env applies negative
    sat_ms: int = 5000
    sat_clause_cap: int = 200_000

    tri_cache: bool = True  # enable triangle cache if your env uses it

    # TensorBoard logging of boards/videos during play
    tb_log_moves: bool = False
    tb_log_every: int = 10
    tb_log_max_moves: int = 200
    tb_video_per_episode: bool = False
    tb_video_fps: int = 10
    tb_image_size: int = 640

    # Terminal progress logging
    term_log_moves: bool = False
    term_log_every: int = 10
    term_log_compact: bool = True

    # Conjecture line when a new-max-n draw witness is saved
    conjecture_print: bool = True
