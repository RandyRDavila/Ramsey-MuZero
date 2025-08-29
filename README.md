# Ramsey MuZero

An interactive CLI and training framework for experimenting with **misère Ramsey problems** using a MuZero-style agent.
The system learns to color edges of complete graphs in two colors while avoiding monochromatic cliques of size \(r\).
It combines reinforcement learning (MuZero + GNNs), SAT-based completion, and curriculum learning to explore new lower bounds for Ramsey numbers.

---

## Installation

```bash
# code Clone and enter
git clone https://github.com/RandyRDavila/Ramsey-MuZero.git Ramsey-MuZero
cd Ramsey-Experiment

# code Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# code Install in editable mode (creates the `ramsey` command)
pip install -U pip setuptools wheel
pip install -e .

# code (Optional) Install a PyTorch build for your machine
# code macOS (CPU/MPS):
pip install torch torchvision torchaudio
# code Linux with CUDA (example CUDA 12.4):
pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision torchaudio
```

> Prefer Conda? An `environment.yml` is included—activate the env first, then run `pip install -e .` inside that env.

---

## Run

```bash
# code Friendly console command (after `pip install -e .`)
ramsey

# code Also works:
python -m ramsey
```

You’ll see an ASCII banner, an environment summary, and an interactive menu.

---

## Menu options

- **Train (custom GNN trainer: ramsey/*)**
  Launches the MuZero-style trainer with interactive prompts for:
  - Device (`auto`, `cpu`, `cuda`, `mps`)
  - Clique size \(r\) (3, 4, 5, 6)
  - \(N_{\max}\) (padding size)
  - Curriculum start size \(n\)
  - MCTS simulations per move
  - Exploration profile (`default` or `unstick`)
  - SAT parameters (time budget, max calls, edge gate)
  - Results directory

- **Quick SAT smoke**
  Runs a fast SAT completion check (`ramsey.sat_tools.bounded_draw_completion`) to verify the toolchain.

- **Empty a results directory (keep folder)**
  Deletes all subdirectories/files under a chosen results path but **keeps** the folder.

- **Exit**
  Quit the CLI.

---

## Training details

When you select **Train**, the CLI collects hyperparameters and launches the trainer via:

- `python -m ramsey.main` (preferred package trainer shim), or
- falls back to the repo-root `main.py`.

Typical flags include:

- `--device` (`auto` / `cpu` / `cuda` / `mps`)
- `--r` (3/4/5/6)
- `--n_max`, `--curriculum_start_n`, `--mcts_sims`
- `--results_dir`
- Exploration knobs: `--root_noise_frac`, `--widen_c`, `--widen_alpha`, `--lambda_shape`
- SAT knobs: `--sat_ms`, `--sat_calls_per_game`, and `--sat_edges_left_{r}`
- Logging: `--term_log_moves`, `--term_log_every`

On macOS, the CLI can set:

- `KMP_DUPLICATE_LIB_OK=TRUE` (avoids OpenMP conflicts)
- `PYTORCH_ENABLE_MPS_FALLBACK=1` (falls back to CPU for ops lacking MPS)

---

## Results & TensorBoard

Artifacts are written to your chosen `results/` directory (default `./results`):

```bash
results/
├── checkpoints/        # model checkpoints (*.pt)
├── tb/                 # TensorBoard logs
└── witness/            # NPZ + PNG witness graphs
```

## Purpose

This project explores automated discovery of **Ramsey-theoretic lower bounds** using:

- **Reinforcement learning (MuZero)** — policy/value with MCTS
- **Graph neural networks (GNNs)** — representing colored graphs
- **SAT solvers** — detecting forced draws and completing colorings
- **Curriculum learning** — gradually increasing graph size \(n\)
- **Interactive CLI** — reproducible, user-friendly experiments

Initial targets include classical cases (e.g., \(R(3,3)\)) for validation, then scaling to \(R(4,4)\), \(R(5,5)\), and beyond.

---

## Game Play App

To play the Misere Ramsey game, run the following command in the terminal:

```bash
streamlit run app.py
```

---

## License & Citation

Add your license text here (e.g., MIT).
If you use this work in research, please cite this repository.

---

## Acknowledgments

Thanks to contributors and the open-source ecosystem (PyTorch, Questionary, Rich, OR-Tools, python-sat, etc.) that make this project possible.
EOF
