# app.py — Misère Ramsey Edge-Coloring Game (Streamlit)
# ---------------------------------------------------
# Quick install (SAT optional; auto-detected):
#   pip install streamlit numpy networkx matplotlib python-sat[pblib,aiger]
# Run:
#   streamlit run app.py

from __future__ import annotations

import itertools
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt

# Optional SAT support
SAT_AVAILABLE = False
try:
    from pysat.formula import CNF
    from pysat.solvers import Glucose4

    SAT_AVAILABLE = True
except Exception:
    SAT_AVAILABLE = False

# -------------------------
# Game constants & typing
# -------------------------
RED, BLUE = 1, 2
Color = int
Edge = Tuple[int, int]

# -------------------------
# Helper: edge indexing & edge list
# -------------------------

def edge_index(i: int, j: int, n: int) -> int:
    """Map (i, j) with 0 <= i < j < n to index in the K_n edge list.
    Spec-provided formula for speed and sanity.
    """
    return i * (n - 1) - (i * (i - 1)) // 2 + (j - i - 1)


def make_edges(n: int) -> List[Edge]:
    return [(i, j) for i in range(n) for j in range(i + 1, n)]


# -------------------------
# Bitset clique utilities
# -------------------------

def exists_k_clique_in_mask(adj_bits: List[int], mask: int, k: int) -> bool:
    """Return True iff there exists a k-clique using only vertices in `mask`.

    Backtracking over bitsets with simple pruning. Fast enough for r <= 5, n <= 24.
    """
    if k <= 0:
        return True
    if mask == 0:
        return False
    if mask.bit_count() < k:
        return False

    def rec(cand: int, need: int) -> bool:
        if need <= 1:
            return cand != 0
        if cand.bit_count() < need:
            return False
        while cand:
            if cand.bit_count() < need:
                return False
            lsb = cand & -cand
            v = lsb.bit_length() - 1
            cand ^= lsb
            next_cand = cand & adj_bits[v]
            if rec(next_cand, need - 1):
                return True
        return False

    return rec(mask, k)


def created_Kr_by_adding_edge(adj_bits: List[int], u: int, v: int, r: int) -> bool:
    """After setting (u, v) in adj_bits (both directions), return True iff a
    monochromatic K_r was created. Optimization: reduce to checking an
    (r-2)-clique in the common neighbors of u and v.
    """
    if r <= 2:
        return True  # UI enforces r >= 3
    common = adj_bits[u] & adj_bits[v]
    return exists_k_clique_in_mask(adj_bits, common, r - 2)


def exists_Kr_global(adj_bits: List[int], r: int, n: Optional[int] = None) -> bool:
    if n is None:
        n = len(adj_bits)
    all_mask = (1 << n) - 1
    return exists_k_clique_in_mask(adj_bits, all_mask, r)


# -------------------------
# Circulant helpers
# -------------------------

def build_circulant_classes(n: int, edges: List[Edge]) -> List[List[int]]:
    """For current n, return class_edges[d-1] = list of edge indices in distance class d.
    Distance d(i,j) = min(|i-j|, n - |i-j|), d in 1..floor(n/2).
    """
    max_d = n // 2
    classes: List[List[int]] = [[] for _ in range(max_d)]
    for idx, (i, j) in enumerate(edges):
        delta = j - i  # since i < j
        d = min(delta, n - delta)
        if 1 <= d <= max_d:
            classes[d - 1].append(idx)
    return classes


# -------------------------
# State management
# -------------------------

def default_session_state() -> None:
    st.session_state.setdefault('n', 6)
    st.session_state.setdefault('r', 3)
    st.session_state.setdefault('any_kr_loss', False)
    # Circulant availability is a toggle; choose per-move when enabled
    st.session_state.setdefault('circulant_enabled', True)

    st.session_state.setdefault('to_play', RED)
    st.session_state.setdefault('edges', [])
    st.session_state.setdefault('edge_state', None)  # np.int8 array of len |E|
    st.session_state.setdefault('red_bits', [])
    st.session_state.setdefault('blue_bits', [])
    st.session_state.setdefault('class_edges', [])
    st.session_state.setdefault('last_move', None)  # ([(u,v), ...], color)
    st.session_state.setdefault('history', [])      # list of dicts per ply
    st.session_state.setdefault('terminal', False)
    st.session_state.setdefault('winner', None)
    st.session_state.setdefault('draw', False)
    st.session_state.setdefault('terminal_reason', '')

    # Rewards: track totals and a per-ply trace for plotting
    st.session_state.setdefault('rewards', {RED: 0.0, BLUE: 0.0})
    st.session_state.setdefault('reward_trace', [(0, 0.0, 0.0)])  # list of (ply, red_total, blue_total)

    # UI selections
    st.session_state.setdefault('sel_u', 0)
    st.session_state.setdefault('sel_v', 1)
    st.session_state.setdefault('sel_d', 1)


@dataclass
class MoveResult:
    ok: bool
    loss: bool = False
    draw: bool = False
    reason: str = ''


def reset_game(n: int, r: int, any_kr_loss: bool, circulant_enabled: bool) -> None:
    st.session_state.n = n
    st.session_state.r = r
    st.session_state.any_kr_loss = any_kr_loss
    st.session_state.circulant_enabled = circulant_enabled
    st.session_state.to_play = RED

    st.session_state.edges = make_edges(n)
    st.session_state.edge_state = np.zeros(len(st.session_state.edges), dtype=np.int8)
    st.session_state.red_bits = [0] * n
    st.session_state.blue_bits = [0] * n
    st.session_state.class_edges = build_circulant_classes(n, st.session_state.edges)

    st.session_state.last_move = None
    st.session_state.history = []
    st.session_state.terminal = False
    st.session_state.winner = None
    st.session_state.draw = False
    st.session_state.terminal_reason = ''

    st.session_state.rewards = {RED: 0.0, BLUE: 0.0}
    st.session_state.reward_trace = [(0, 0.0, 0.0)]

    st.session_state.sel_u = 0
    st.session_state.sel_v = 1 if n >= 2 else 0
    st.session_state.sel_d = 1


# -------------------------
# Low-level edge coloring ops
# -------------------------

def _set_edge(idx: int, color: Color) -> None:
    """Set edge at global index to color, updating bitsets. Assumes currently uncolored."""
    (u, v) = st.session_state.edges[idx]
    st.session_state.edge_state[idx] = color
    bit = 1 << v
    bit_rev = 1 << u
    if color == RED:
        st.session_state.red_bits[u] |= bit
        st.session_state.red_bits[v] |= bit_rev
    elif color == BLUE:
        st.session_state.blue_bits[u] |= bit
        st.session_state.blue_bits[v] |= bit_rev


def _unset_edge(idx: int, color: Color) -> None:
    """Unset colored edge to uncolored, updating bitsets (for undo)."""
    (u, v) = st.session_state.edges[idx]
    st.session_state.edge_state[idx] = 0
    bit = 1 << v
    bit_rev = 1 << u
    if color == RED:
        st.session_state.red_bits[u] &= ~bit
        st.session_state.red_bits[v] &= ~bit_rev
    elif color == BLUE:
        st.session_state.blue_bits[u] &= ~bit
        st.session_state.blue_bits[v] &= ~bit_rev


def _append_reward_trace():
    """Append current cumulative rewards with current ply index."""
    ply = len(st.session_state.history)
    rt = st.session_state.rewards
    st.session_state.reward_trace.append((ply, float(rt[RED]), float(rt[BLUE])))


def _apply_terminal(loss_for: Optional[Color], draw: bool, reason: str) -> None:
    st.session_state.terminal = True
    st.session_state.draw = draw
    if draw:
        # First player to force a draw wins: current mover gets +1.0
        mover = st.session_state.to_play
        st.session_state.winner = mover
        st.session_state.rewards[mover] += 1.0
        st.session_state.terminal_reason = reason
    else:
        st.session_state.winner = BLUE if loss_for == RED else RED
        st.session_state.terminal_reason = reason
        if loss_for is not None:
            st.session_state.rewards[loss_for] += -1.0


# -------------------------
# Loss rule helper
# -------------------------

def _loss_check_after_edge(color_bits: List[int], other_bits: List[int], u: int, v: int, r: int) -> bool:
    # Misère baseline: mover loses if they create K_r in *their* color
    if created_Kr_by_adding_edge(color_bits, u, v, r):
        return True
    # Optional shaping: lose if any K_r (either color) exists after move
    if st.session_state.any_kr_loss:
        n = st.session_state.n
        if exists_Kr_global(color_bits, r, n) or exists_Kr_global(other_bits, r, n):
            return True
    return False


# -------------------------
# Move application (human/agent)
# -------------------------

def apply_edge_move(u: int, v: int, color: Color, move_type: str) -> MoveResult:
    if u > v:
        u, v = v, u
    n = st.session_state.n
    idx = edge_index(u, v, n)
    if st.session_state.edge_state[idx] != 0:
        return MoveResult(ok=False, reason="Edge already colored.")

    prev = {
        'prev_last_move': st.session_state.last_move,
        'prev_terminal': st.session_state.terminal,
        'prev_winner': st.session_state.winner,
        'prev_draw': st.session_state.draw,
        'prev_reason': st.session_state.terminal_reason,
        'prev_to_play': st.session_state.to_play,
        'prev_rewards': dict(st.session_state.rewards),
    }

    _set_edge(idx, color)
    st.session_state.last_move = ([(u, v)], color)

    color_bits = st.session_state.red_bits if color == RED else st.session_state.blue_bits
    other_bits = st.session_state.blue_bits if color == RED else st.session_state.red_bits
    r = st.session_state.r

    result = MoveResult(ok=True)

    if _loss_check_after_edge(color_bits, other_bits, u, v, r):
        result.loss = True
        reason = f"Move ({u},{v}) created a monochromatic K_{r}."
        _apply_terminal(loss_for=color, draw=False, reason=reason)
        result.reason = reason
    else:
        # Draw if no uncolored edges and no mono K_r exists
        if int(np.count_nonzero(st.session_state.edge_state == 0)) == 0:
            if not exists_Kr_global(st.session_state.red_bits, r, n) and not exists_Kr_global(st.session_state.blue_bits, r, n):
                result.draw = True
                _apply_terminal(loss_for=None, draw=True, reason="All edges colored without any K_r.")

    hist_entry = {
        'ply': len(st.session_state.history) + 1,
        'move_type': move_type,
        'payload': {'u': u, 'v': v, 'idx': idx, 'color': color},
        'result': 'loss' if result.loss else ('draw' if result.draw else 'ok'),
        **prev,
    }
    st.session_state.history.append(hist_entry)

    # Toggle to_play if game not over
    if not st.session_state.terminal:
        st.session_state.to_play = BLUE if color == RED else RED

    _append_reward_trace()
    return result


def apply_class_move(d: int, color: Color, move_type: str) -> MoveResult:
    n = st.session_state.n
    class_list = st.session_state.class_edges
    if d < 1 or d > len(class_list):
        return MoveResult(ok=False, reason="Invalid distance class.")

    class_idxs = class_list[d - 1]
    todo = [idx for idx in class_idxs if st.session_state.edge_state[idx] == 0]
    if not todo:
        return MoveResult(ok=False, reason=f"Class d={d} has no uncolored edges.")

    prev = {
        'prev_last_move': st.session_state.last_move,
        'prev_terminal': st.session_state.terminal,
        'prev_winner': st.session_state.winner,
        'prev_draw': st.session_state.draw,
        'prev_reason': st.session_state.terminal_reason,
        'prev_to_play': st.session_state.to_play,
        'prev_rewards': dict(st.session_state.rewards),
    }

    colored_edges: List[Tuple[int, int, int]] = []  # (u, v, idx)
    color_bits = st.session_state.red_bits if color == RED else st.session_state.blue_bits
    other_bits = st.session_state.blue_bits if color == RED else st.session_state.red_bits
    r = st.session_state.r

    result = MoveResult(ok=True)

    for idx in todo:
        (u, v) = st.session_state.edges[idx]
        _set_edge(idx, color)
        colored_edges.append((u, v, idx))
        if _loss_check_after_edge(color_bits, other_bits, u, v, r):
            result.loss = True
            reason = f"Class d={d} move created a monochromatic K_{r} at edge ({u},{v})."
            _apply_terminal(loss_for=color, draw=False, reason=reason)
            result.reason = reason
            break

    if not st.session_state.terminal:
        if int(np.count_nonzero(st.session_state.edge_state == 0)) == 0:
            if not exists_Kr_global(st.session_state.red_bits, r, n) and not exists_Kr_global(st.session_state.blue_bits, r, n):
                result.draw = True
                _apply_terminal(loss_for=None, draw=True, reason="All edges colored without any K_r.")

    st.session_state.last_move = ([(u, v) for (u, v, _) in colored_edges], color)

    hist_entry = {
        'ply': len(st.session_state.history) + 1,
        'move_type': move_type,
        'payload': {'d': d, 'edges_colored': colored_edges, 'color': color},
        'result': 'loss' if result.loss else ('draw' if result.draw else 'ok'),
        **prev,
    }
    st.session_state.history.append(hist_entry)

    if not st.session_state.terminal:
        st.session_state.to_play = BLUE if color == RED else RED

    _append_reward_trace()
    return result


# -------------------------
# SAT finisher (optional)
# -------------------------

def attempt_sat_draw(penalty: float, gate: int) -> MoveResult:
    if not SAT_AVAILABLE:
        return MoveResult(ok=False, reason="python-sat is not installed.")

    remaining_idxs = np.where(st.session_state.edge_state == 0)[0].tolist()
    if len(remaining_idxs) > gate:
        return MoveResult(ok=False, reason=f"SAT gated: {len(remaining_idxs)} > gate={gate}.")

    prev = {
        'prev_last_move': st.session_state.last_move,
        'prev_terminal': st.session_state.terminal,
        'prev_winner': st.session_state.winner,
        'prev_draw': st.session_state.draw,
        'prev_reason': st.session_state.terminal_reason,
        'prev_to_play': st.session_state.to_play,
        'prev_rewards': dict(st.session_state.rewards),
    }

    mover = st.session_state.to_play
    st.session_state.rewards[mover] += float(penalty)

    n = st.session_state.n
    r = st.session_state.r
    edges = st.session_state.edges
    m = len(edges)

    cnf = CNF()

    # Unit clauses for already-colored edges
    for idx, col in enumerate(st.session_state.edge_state.tolist()):
        if col == RED:
            cnf.append([idx + 1])
        elif col == BLUE:
            cnf.append([-(idx + 1)])

    # For every r-vertex subset S, forbid monochromatic red or blue
    verts = list(range(n))
    for S in itertools.combinations(verts, r):
        e_idx_list = []
        for i in range(r):
            for j in range(i + 1, r):
                u, v = S[i], S[j]
                if u > v:
                    u, v = v, u
                e_idx_list.append(edge_index(u, v, n) + 1)
        cnf.append([-(v_idx) for v_idx in e_idx_list])  # not all red
        cnf.append([+(v_idx) for v_idx in e_idx_list])  # not all blue

    res = MoveResult(ok=True)
    filled_edges: List[Tuple[int, int, int, int]] = []  # (u, v, idx, color)

    with Glucose4(bootstrap_with=cnf.clauses) as solver:
        sat = solver.solve()
        if not sat:
            res.ok = False
            res.reason = "SAT: No draw completion exists (UNSAT)."
        else:
            model = solver.get_model()
            assign = {}
            for lit in model:
                v = abs(lit)
                if 1 <= v <= m:
                    assign[v] = (lit > 0)
            for idx in remaining_idxs:
                v_idx = idx + 1
                red_true = assign.get(v_idx, True)
                color = RED if red_true else BLUE
                (u, v) = edges[idx]
                _set_edge(idx, color)
                filled_edges.append((u, v, idx, color))

            # Draw forced by mover via SAT: award +1.0 (penalty already applied earlier)
            st.session_state.rewards[mover] += 1.0
            _apply_terminal(loss_for=None, draw=True, reason="SAT found a valid draw coloring.")
            res.draw = True

    hist_entry = {
        'ply': len(st.session_state.history) + 1,
        'move_type': 'sat',
        'payload': {
            'filled_edges': filled_edges,
            'mover': mover,
            'penalty': float(penalty),
        },
        'result': 'draw' if res.draw else 'unsat',
        **prev,
    }
    st.session_state.history.append(hist_entry)

    _append_reward_trace()
    if filled_edges:
        st.session_state.last_move = ([(u, v) for (u, v, *_ ) in filled_edges], None)

    return res


# -------------------------
# Undo logic
# -------------------------

def undo_last_move() -> None:
    if not st.session_state.history:
        st.warning("No moves to undo.")
        return
    entry = st.session_state.history.pop()
    move_type = entry['move_type']

    if move_type in ('edge', 'agent_edge'):
        idx = entry['payload']['idx']
        color = entry['payload']['color']
        _unset_edge(idx, color)
    elif move_type in ('class', 'agent_class'):
        for (_u, _v, idx) in entry['payload']['edges_colored']:
            color = entry['payload']['color']
            _unset_edge(idx, color)
    elif move_type == 'sat':
        for (_u, _v, idx, color) in entry['payload']['filled_edges']:
            _unset_edge(idx, color)
        mover = entry['payload']['mover']
        st.session_state.rewards[mover] -= entry['payload']['penalty']

    # Restore previous flags/state
    st.session_state.last_move = entry['prev_last_move']
    st.session_state.terminal = entry['prev_terminal']
    st.session_state.winner = entry['prev_winner']
    st.session_state.draw = entry['prev_draw']
    st.session_state.terminal_reason = entry['prev_reason']
    st.session_state.to_play = entry['prev_to_play']
    st.session_state.rewards = dict(entry['prev_rewards'])

    # Trim reward trace to match history
    if st.session_state.reward_trace:
        st.session_state.reward_trace.pop()


# -------------------------
# Drawing / Visualization
# -------------------------

def draw_graph() -> None:
    n = st.session_state.n
    edges = st.session_state.edges
    edge_state = st.session_state.edge_state

    pos = {i: (math.cos(2 * math.pi * i / n), math.sin(2 * math.pi * i / n)) for i in range(n)}

    G = nx.Graph()
    G.add_nodes_from(range(n))
    for idx, (u, v) in enumerate(edges):
        color = edge_state[idx]
        if color == 0:
            G.add_edge(u, v, color='gray', width=1.0, alpha=0.35)
        elif color == RED:
            G.add_edge(u, v, color='red', width=2.5, alpha=0.95)
        else:
            G.add_edge(u, v, color='blue', width=2.5, alpha=0.95)

    highlight_set = set()
    if st.session_state.last_move is not None:
        last_edges, _last_color = st.session_state.last_move
        for (u, v) in last_edges:
            a, b = (u, v) if u < v else (v, u)
            highlight_set.add((a, b))

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.axis('off')

    for (u, v, data) in G.edges(data=True):
        width = data.get('width', 2.0)
        if (min(u, v), max(u, v)) in highlight_set:
            width = max(width, 3.8)
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=[(u, v)],
            width=width,
            edge_color=data.get('color', 'gray'),
            alpha=data.get('alpha', 0.9),
        )

    nx.draw_networkx_nodes(G, pos, node_size=320, node_color='white', edgecolors='black', linewidths=1.2)
    nx.draw_networkx_labels(G, pos, font_size=11, font_weight='bold')

    st.pyplot(fig)
    plt.close(fig)


def draw_rewards_plot():
    trace = st.session_state.reward_trace
    if not trace or len(trace) < 1:
        return
    xs = [t[0] for t in trace]
    red_y = [t[1] for t in trace]
    blue_y = [t[2] for t in trace]

    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    ax.plot(xs, red_y, marker='o', linewidth=2.5, label='RED (You)')
    ax.plot(xs, blue_y, marker='o', linewidth=2.5, label='BLUE (Agent)')
    ax.set_xlabel('Ply (moves)')
    ax.set_ylabel('Cumulative reward')
    ax.grid(True, alpha=0.25)
    ax.legend(loc='best')
    st.pyplot(fig)
    plt.close(fig)


# -------------------------
# Random agent
# -------------------------

def agent_random_move() -> MoveResult:
    if st.session_state.terminal:
        return MoveResult(ok=False, reason="Game is over.")
    if st.session_state.to_play != BLUE:
        return MoveResult(ok=False, reason="Not agent's turn.")

    edge_state = st.session_state.edge_state
    # If circulant is enabled, flip a fair coin between edge/class when both legal
    can_edge = np.any(edge_state == 0)
    can_class = False
    if st.session_state.circulant_enabled:
        for idxs in st.session_state.class_edges:
            if any(edge_state[idx] == 0 for idx in idxs):
                can_class = True
                break

    choice = None
    if can_edge and can_class:
        choice = random.choice(['edge', 'class'])
    elif can_class:
        choice = 'class'
    elif can_edge:
        choice = 'edge'
    else:
        return MoveResult(ok=False, reason="No legal moves.")

    if choice == 'edge':
        uncolored = np.where(edge_state == 0)[0].tolist()
        idx = random.choice(uncolored)
        u, v = st.session_state.edges[idx]
        return apply_edge_move(u, v, BLUE, move_type='agent_edge')
    else:
        avail = [d for d, idxs in enumerate(st.session_state.class_edges, start=1)
                 if any(edge_state[idx] == 0 for idx in idxs)]
        d = random.choice(avail)
        return apply_class_move(d, BLUE, move_type='agent_class')


# -------------------------
# Sidebar controls
# -------------------------

def sidebar_controls():
    st.sidebar.header("Game settings")

    n = st.sidebar.number_input("n (size of K_n)", min_value=3, max_value=24, value=st.session_state.n, step=1)
    r = st.sidebar.number_input("r (target K_r)", min_value=3, max_value=6, value=st.session_state.r, step=1)

    any_kr_loss = st.sidebar.toggle(
        "Any K_r loss (shaping)", value=st.session_state.any_kr_loss,
        help="Mover loses if after their move any monochromatic K_r exists (either color)."
    )

    circulant_enabled = st.sidebar.toggle(
        "Enable circulant macro moves",
        value=st.session_state.circulant_enabled,
        help="When ON, choose per-move between single edge and distance-class coloring."
    )

    st.sidebar.subheader("SAT options")
    sat_enable = st.sidebar.checkbox(
        "Enable SAT finisher button", value=False,
        help="Attempt to finish to a draw; small penalty to the mover."
    )
    sat_penalty = st.sidebar.number_input("SAT penalty", value=-0.05, step=0.01, format="%.2f")
    sat_gate = st.sidebar.number_input("SAT gate (max uncolored edges)", value=60, step=5)

    c1, c2 = st.sidebar.columns(2)
    with c1:
        if st.sidebar.button("New game", use_container_width=True):
            reset_game(int(n), int(r), any_kr_loss, circulant_enabled)
            st.rerun()
    with c2:
        if st.sidebar.button("Undo last move", use_container_width=True):
            undo_last_move()
            st.rerun()

    return {
        'n': int(n),
        'r': int(r),
        'any_kr_loss': any_kr_loss,
        'circulant_enabled': circulant_enabled,
        'sat_enable': sat_enable,
        'sat_penalty': float(sat_penalty),
        'sat_gate': int(sat_gate),
    }


# -------------------------
# How to use (markdown)
# -------------------------

def how_to_use_md():
    st.markdown(
        r"""
# Misère Ramsey Edge-Coloring Game
Play against a **random BLUE agent** on the complete graph $K_n$ by coloring edges **RED**.

**Rules**
- **Misère baseline:** You **lose immediately** if your move creates a monochromatic $K_r$ **in your own color**.
- **Any $K_r$ loss (optional):** If toggled ON, you lose if after your move **either** color has a monoclique $K_r$.
- If all edges are colored and no $K_r$ exists in either color, the mover who **forces the draw** **wins** and gets **+1.0**.

**Moves**
- **Single edge:** Choose endpoints $u<v$ and color $(u,v)$ RED.
- **Circulant macro (optional):** Enable in the sidebar, then on your turn choose a distance class $d\in\{1,\dots,\lfloor n/2\rfloor\}$ to color **all currently uncolored** edges in that class. If any edge in the macro creates a $K_r$, you **instantly lose**; already-colored edges remain.

**SAT finisher (optional)**
- If enabled and within the gate, you may attempt a **draw completion** with a SAT solver. This applies a small **penalty** to you immediately; if SAT succeeds, the draw is **yours** (+1.0) despite the penalty.

**Rewards**
- Loss: mover **-1.0**. Draw forced by mover: **+1.0** to that mover. SAT adds its (usually small) negative penalty.

**Tips**
- Try `n=6, r=3` to see quick triangle losses.
- For `n=6, r=3`, SAT on an almost-complete graph should report **UNSAT** (since $R(3,3)=6$), not a draw.
        """
    )


# -------------------------
# Main UI
# -------------------------

def main():
    st.set_page_config(page_title="Misère Ramsey Edge-Coloring", layout="wide")
    default_session_state()

    cfg = sidebar_controls()

    st.title("Misère Ramsey Edge-Coloring — Human (RED) vs Random Agent (BLUE)")
    how_to_use_md()

    left, right = st.columns([2, 1])

    with right:
        st.subheader("Status")
        to_play = st.session_state.to_play
        st.write(f"**To move:** {'RED (You)' if to_play == RED else 'BLUE (Agent)'}")
        remaining = int(np.count_nonzero(st.session_state.edge_state == 0))
        st.write(f"**Uncolored edges:** {remaining}")
        classes_left = sum(1 for _d, idxs in enumerate(st.session_state.class_edges, start=1)
                           if any(st.session_state.edge_state[idx] == 0 for idx in idxs))
        st.write(f"**Classes with uncolored edges:** {classes_left}")
        rtot = st.session_state.rewards
        st.write(f"**Rewards** — RED (You): {rtot[RED]:.2f} | BLUE (Agent): {rtot[BLUE]:.2f}")
        if st.session_state.terminal:
            if st.session_state.draw:
                st.success(f"**Draw forced by {'You' if st.session_state.winner == RED else 'Agent'}:** {st.session_state.terminal_reason}")
            else:
                loser = 'RED (You)' if st.session_state.winner == BLUE else 'BLUE (Agent)'
                st.error(f"**Loss for {loser}:** {st.session_state.terminal_reason}")

        # SAT controls
        if cfg['sat_enable']:
            if not SAT_AVAILABLE:
                st.info("python-sat not installed; SAT button disabled.")
            else:
                rem = int(np.count_nonzero(st.session_state.edge_state == 0))
                if rem <= cfg['sat_gate'] and not st.session_state.terminal:
                    if st.button("Attempt SAT draw", type='secondary', use_container_width=True):
                        res = attempt_sat_draw(cfg['sat_penalty'], cfg['sat_gate'])
                        if not res.ok and not res.draw:
                            st.warning(res.reason)
                        st.rerun()
                else:
                    st.caption(f"SAT gated: remaining {rem} > gate {cfg['sat_gate']} or game over.")

        st.subheader("Reward trajectory")
        draw_rewards_plot()

        st.subheader("Controls")
        if st.button("Let agent move (random)", use_container_width=True):
            res = agent_random_move()
            if not res.ok and res.reason:
                st.warning(res.reason)
            st.rerun()

        st.subheader("Move history")
        if st.session_state.history:
            hist_lines = []
            for h in st.session_state.history[-14:]:
                kind = h['move_type']
                if kind in ('edge', 'agent_edge'):
                    u, v = h['payload']['u'], h['payload']['v']
                    who = 'RED' if h['payload']['color'] == RED else 'BLUE'
                    hist_lines.append(f"{h['ply']:>3}: {kind} {who} ({u},{v}) → {h['result']}")
                elif kind in ('class', 'agent_class'):
                    d = h['payload']['d']
                    who = 'RED' if h['payload']['color'] == RED else 'BLUE'
                    cnt = len(h['payload']['edges_colored'])
                    hist_lines.append(f"{h['ply']:>3}: {kind} {who} d={d} (+{cnt} edges) → {h['result']}")
                elif kind == 'sat':
                    cnt = len(h['payload']['filled_edges'])
                    hist_lines.append(f"{h['ply']:>3}: SAT attempt (+{cnt} edges) → {h['result']}")
            st.code("\n".join(hist_lines) if hist_lines else "(empty)")
        else:
            st.caption("No moves yet.")

    with left:
        draw_graph()

        if not st.session_state.terminal:
            st.subheader("Your move")
            options = ["Single edge"] + (["Circulant class"] if cfg['circulant_enabled'] else [])
            move_choice = st.radio("Move type", options, horizontal=True)

            if move_choice == "Single edge":
                n = st.session_state.n
                u = st.selectbox("u", list(range(n)), index=min(st.session_state.sel_u, n-1), key='sel_u_box')
                v_choices = [j for j in range(u + 1, n) if st.session_state.edge_state[edge_index(u, j, n)] == 0]
                if not v_choices:
                    st.info("No legal v for this u; choose a different u.")
                    v = None
                else:
                    default_v = v_choices[0]
                    if st.session_state.get('sel_v') in v_choices:
                        default_v = st.session_state['sel_v']
                    v = st.selectbox("v", v_choices, index=v_choices.index(default_v), key='sel_v_box')
                    st.session_state.sel_v = v

                if st.button("Play edge move", type='primary', disabled=(v is None)):
                    res = apply_edge_move(u, v, RED, move_type='edge')
                    if not res.ok and res.reason:
                        st.warning(res.reason)
                    st.rerun()

            else:
                n = st.session_state.n
                available = [d for d, idxs in enumerate(st.session_state.class_edges, start=1)
                             if any(st.session_state.edge_state[idx] == 0 for idx in idxs)]
                if not available:
                    st.info("No circulant classes remain with uncolored edges.")
                else:
                    default_d = available[0]
                    if st.session_state.get('sel_d') in available:
                        default_d = st.session_state.sel_d
                    d = st.selectbox("Distance class d", available, index=available.index(default_d))
                    st.session_state.sel_d = d
                    if st.button("Play class move", type='primary'):
                        res = apply_class_move(d, RED, move_type='class')
                        if not res.ok and res.reason:
                            st.warning(res.reason)
                        st.rerun()

    st.caption(
        ("python-sat available: ✅" if SAT_AVAILABLE else "python-sat available: ❌ (optional dependency)")
        + (" · Circulant moves: ON" if cfg['circulant_enabled'] else " · Circulant moves: OFF")
        + (" · Any K_r loss ON" if cfg['any_kr_loss'] else " · Misère baseline")
    )


if __name__ == "__main__":
    if 'edge_state' not in st.session_state or st.session_state.edge_state is None:
        reset_game(6, 3, any_kr_loss=False, circulant_enabled=True)
    main()
