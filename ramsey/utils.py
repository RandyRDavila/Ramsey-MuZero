import os
import random
import numpy as np
import torch
from pathlib import Path
import matplotlib.pyplot as plt


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def detect_device(pref: str):
    if pref != "auto":
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def masked_softmax(logits: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    logits: [..., A]  (float32 preferred for stability)
    mask:   [..., A]  boolean, True where valid
    """
    x = logits.float()
    neg_inf = torch.full_like(x, -1e9)
    x_masked = torch.where(mask, x, neg_inf)
    return torch.softmax(x_masked, dim=dim)


def save_witness_png_npz(path_png: str, path_npz: str, red_adj: np.ndarray, blue_adj: np.ndarray, meta: dict):
    """Pretty render of a 2-coloring: nodes on a circle, red/blue edges."""
    ensure_dir(os.path.dirname(path_png) or ".")
    import math
    n = red_adj.shape[0]

    # circle layout
    angles = np.linspace(0, 2*math.pi, n, endpoint=False)
    xs = np.cos(angles)
    ys = np.sin(angles)

    # style scales for larger n
    if n <= 18:
        lw, alpha, node_size = 1.6, 0.9, 22
    elif n <= 30:
        lw, alpha, node_size = 1.0, 0.7, 16
    else:
        lw, alpha, node_size = 0.6, 0.45, 10

    fig, ax = plt.subplots(figsize=(8, 8), facecolor="white")
    ax.set_aspect("equal")
    ax.axis("off")

    # draw edges
    for i in range(n):
        xi, yi = xs[i], ys[i]
        for j in range(i+1, n):
            xj, yj = xs[j], ys[j]
            if red_adj[i, j]:
                ax.plot([xi, xj], [yi, yj], linewidth=lw, alpha=alpha, color="crimson")
            elif blue_adj[i, j]:
                ax.plot([xi, xj], [yi, yj], linewidth=lw, alpha=alpha, color="royalblue")

    # draw nodes
    ax.scatter(xs, ys, s=node_size, color="#111111", zorder=3)

    title = f"2-coloring draw witness (n={n}, r={meta.get('r', '?')})"
    ax.set_title(title, fontsize=14, pad=12)
    fig.tight_layout(pad=0.1)
    fig.savefig(path_png, dpi=300)
    plt.close(fig)

    # always save the raw arrays too
    np.savez_compressed(path_npz, red=red_adj.astype(np.uint8), blue=blue_adj.astype(np.uint8), **meta)


# --- TensorBoard rendering helpers ---

def render_board_image(red_adj: np.ndarray,
                       blue_adj: np.ndarray,
                       n = None,
                       r  = None,
                       size: int = 640) -> np.ndarray:
    """
    Render the current 2-coloring to an RGB uint8 image (H,W,3).
    Uses a circle layout; red edges = 'crimson', blue edges = 'royalblue'.
    """
    import math
    n = int(red_adj.shape[0]) if n is None else int(n)

    angles = np.linspace(0, 2*math.pi, n, endpoint=False)
    xs = np.cos(angles)
    ys = np.sin(angles)

    # style scales
    if n <= 18:
        lw, alpha, node_size = 1.6, 0.9, 22
    elif n <= 30:
        lw, alpha, node_size = 1.0, 0.7, 16
    else:
        lw, alpha, node_size = 0.6, 0.45, 10

    dpi = 100
    inches = size / dpi

    fig, ax = plt.subplots(figsize=(inches, inches), dpi=dpi, facecolor="white")
    ax.set_aspect("equal")
    ax.axis("off")

    # edges
    for i in range(n):
        xi, yi = xs[i], ys[i]
        for j in range(i+1, n):
            xj, yj = xs[j], ys[j]
            if red_adj[i, j]:
                ax.plot([xi, xj], [yi, yj], linewidth=lw, alpha=alpha, color="crimson")
            elif blue_adj[i, j]:
                ax.plot([xi, xj], [yi, yj], linewidth=lw, alpha=alpha, color="royalblue")

    # nodes
    ax.scatter(xs, ys, s=node_size, color="#111111", zorder=3)

    if r is not None:
        ax.set_title(f"n={n}, r={r}", fontsize=12, pad=8)

    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3).copy()
    plt.close(fig)
    return img
