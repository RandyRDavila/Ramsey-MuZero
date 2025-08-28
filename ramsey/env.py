import math
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Set

import numpy as np
import torch

from .config import RamseyConfig


def upper_triangle_pairs(n: int):
    for i in range(n):
        for j in range(i + 1, n):
            yield (i, j)

# --- PATCH: helpers for masks (numpy / torch / list) ---------------------------
# === mask helpers (numpy / torch / list) ======================================

try:
    import torch
except Exception:  # if torch import was already done elsewhere it's fine
    torch = None

def _count_mask(C) -> int:
    """Number of 'true' elements in mask C (np, torch, list/tuple/set)."""
    if isinstance(C, np.ndarray):
        return int(C.sum())
    if torch is not None and torch.is_tensor(C):
        return int(C.sum().item())
    if isinstance(C, (list, tuple, set)):
        return len(C)
    try:
        return int(C.count_nonzero())  # e.g., scipy sparse/bitset-like
    except Exception:
        return 0

def _mask_to_indices(C) -> np.ndarray:
    """Indices of 'true' entries of mask C as a numpy int array."""
    if isinstance(C, np.ndarray):
        return np.flatnonzero(C)
    if torch is not None and torch.is_tensor(C):
        return torch.nonzero(C, as_tuple=False).view(-1).cpu().numpy()
    if isinstance(C, (list, tuple, set)):
        return np.array(sorted(C), dtype=int)
    return np.array([], dtype=int)

def _edges_in_induced(A, idx: np.ndarray) -> int:
    """Count edges in induced subgraph A[idx, idx]; A can be np or torch 0/1 adj."""
    m = int(len(idx))
    if m < 2:
        return 0
    if isinstance(A, np.ndarray):
        sub = A[np.ix_(idx, idx)]
        return int(np.triu(sub, k=1).sum())
    if torch is not None and torch.is_tensor(A):
        idx_t = torch.as_tensor(idx, device=A.device, dtype=torch.long)
        sub = A.index_select(0, idx_t).index_select(1, idx_t)
        return int(torch.triu(sub, diagonal=1).sum().item())
    return 0
# ===============================================================================



class RamseyEnv:
    """
    Misère edge-coloring on K_n.
    - Fast r=5: triangle-in-C check via bitsets
    - Fast r=6: K4-in-C via small DFS with pruning
    """
    RED, BLUE = 0, 1

    def __init__(self, cfg: RamseyConfig):
        self.cfg = cfg
        self.n_max = cfg.n_max
        self.r = cfg.r
        self.device = cfg.device

        self.n = cfg.curriculum_start_n
        self.reset()

    def reset(self, n: Optional[int] = None):
        self.n = n or self.n
        # 0=uncolored, 1=red, 2=blue
        self.edge_color = np.zeros((self.n, self.n), dtype=np.uint8)
        self.red_adj = np.zeros((self.n, self.n), dtype=np.uint8)
        self.blue_adj = np.zeros((self.n, self.n), dtype=np.uint8)
        self.to_play = self.RED
        self.done = False
        self.last_reward = 0.0
        # SAT gating state
        self.sat_calls_left = self.cfg.sat_calls_per_game
        self.sat_tried_cache: Set[int] = set()
        return self.observation()

    # ------------------ Observations ------------------
    def observation(self):
        board = np.zeros((3, self.n_max, self.n_max), dtype=np.float32)
        board[0, :self.n, :self.n] = self.red_adj
        board[1, :self.n, :self.n] = self.blue_adj
        turn_value = 0.0 if self.to_play == self.RED else 1.0
        board[2, :self.n, :self.n] = turn_value
        return board

    def to_tensor_obs(self):
        obs = self.observation()
        return torch.from_numpy(obs).to(self.device)

    # ------------------ Legal actions ------------------
    def legal_edges(self) -> List[Tuple[int, int]]:
        edges = []
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if self.edge_color[i, j] == 0:
                    edges.append((i, j))
        return edges

    def edges_left(self) -> int:
        total_edges = self.n * (self.n - 1) // 2
        colored = int((self.edge_color > 0).sum() // 2)
        return total_edges - colored

    # ------------------ SAT_TRY_NOW gating ------------------
    def sat_action_legal(self) -> bool:
        if self.sat_calls_left <= 0:
            return False
        if   self.r == 3: gate = self.cfg.sat_edges_left_3
        elif self.r == 4: gate = self.cfg.sat_edges_left_4
        elif self.r == 5: gate = self.cfg.sat_edges_left_5
        elif self.r == 6: gate = self.cfg.sat_edges_left_6
        else: gate = 0
        return self.sat_calls_left > 0 and self.edges_left() <= int(gate)

    # ------------------ Step ------------------
    def step_edge(self, i: int, j: int):
        assert not self.done

        # who moves / which color
        to_play_before = self.to_play
        color = 1 if to_play_before == self.RED else 2
        red_move = (color == 1)

        # color the edge in state
        self.edge_color[i, j] = color
        self.edge_color[j, i] = color
        if red_move:
            self.red_adj[i, j] = 1
            self.red_adj[j, i] = 1
        else:
            self.blue_adj[i, j] = 1
            self.blue_adj[j, i] = 1

        # immediate misère loss?
        if self._moves_into_loss_after(i, j, red=red_move):
            self.done = True
            self.last_reward = -1.0
            return self.observation(), self.last_reward, True, {
                "type": "edge",
                "move": (int(i), int(j)),
                "to_play_before": int(to_play_before),
                "lost": True,
            }

        # all edges colored -> terminal (draw or guard-lose)
        if self.edges_left() == 0:
            if self._has_Kr(self.red_adj, self.r) or self._has_Kr(self.blue_adj, self.r):
                # guard (shouldn't occur given loss check)
                self.done = True
                self.last_reward = -1.0
            else:
                self.done = True
                self.last_reward = +1.0
            return self.observation(), self.last_reward, True, {
                "type": "terminal",
                "draw": bool(self.last_reward > 0.5),
            }

        # shaping (non-draw steps only)
        shape = 0.0
        if getattr(self.cfg, "lambda_shape", 0.0) > 0.0:
            C = self._common_neighbors(i, j, red=red_move)
            if self.r == 3:
                shape = -float(self.cfg.lambda_shape) * float(_count_mask(C))
            elif self.r == 4:
                A = self.red_adj if red_move else self.blue_adj
                idx = _mask_to_indices(C)
                edges_in_C = _edges_in_induced(A, idx)
                shape = -float(self.cfg.lambda_shape) * float(edges_in_C)
            elif self.r == 5:
                if hasattr(self, "_triangle_count_in_G_of_mask"):
                    tri_C = self._triangle_count_in_G_of_mask(C, red=red_move)
                else:
                    tri_C = int(self._triangle_in_G_of_mask(C, red=red_move))
                shape = -float(self.cfg.lambda_shape) * float(tri_C)
            elif self.r == 6:
                A = self.red_adj if red_move else self.blue_adj
                idx = _mask_to_indices(C)
                eC = _edges_in_induced(A, idx)
                shape = -float(0.02) * float(eC)  # keep tiny for stability

        # non-terminal: assign shaping reward, flip turn
        self.last_reward = float(shape)
        self.to_play = 1 - self.to_play
        return self.observation(), self.last_reward, False, {
            "type": "edge",
            "move": (int(i), int(j)),
            "to_play_before": int(to_play_before),
            "shape": float(shape),
        }



    def step_sat_try_now(self, sat_solver_fn):
        assert self.sat_action_legal()
        key = self._state_hash_for_sat_cache()
        if key in self.sat_tried_cache:
            # don't burn a call, give small neg and continue
            return self.observation(), -self.cfg.sat_penalty, False, {"sat_cached": True}

        self.sat_tried_cache.add(key)
        self.sat_calls_left -= 1

        # ramsey/env.py (inside step_sat_try_now)

        ok, red_full, blue_full, meta = sat_solver_fn(
            self.n, self.r, self.red_adj.copy(), self.blue_adj.copy(),
            self.edge_color.copy(), time_ms=self.cfg.sat_ms, clause_cap=self.cfg.sat_clause_cap
        )
        if ok:
            # verify again here (defensive) and import the witness into env
            if self._has_Kr(red_full, self.r) or self._has_Kr(blue_full, self.r):
                # treat as failed attempt
                return self.observation(), -self.cfg.sat_penalty, False, {"sat_invalid": True, **meta}
            # accept draw and copy full coloring to env for witness saving
            self.red_adj[:, :] = red_full
            self.blue_adj[:, :] = blue_full
            # rebuild edge_color map
            for i in range(self.n):
                for j in range(i+1, self.n):
                    if red_full[i, j]:
                        self.edge_color[i, j] = self.edge_color[j, i] = 1
                    else:
                        self.edge_color[i, j] = self.edge_color[j, i] = 2
            self.done = True
            self.last_reward = +1.0
            return self.observation(), +1.0, True, {"sat_draw": True, **meta}
        else:
            return self.observation(), -self.cfg.sat_penalty, False, {"sat_fail": True, **meta}

    # ------------------ Helpers ------------------
    def _common_neighbors(self, i, j, red: bool) -> List[int]:
        adj = self.red_adj if red else self.blue_adj
        return [k for k in range(self.n) if k != i and k != j and adj[i, k] and adj[j, k]]
    # --- r=5 helpers: triangle existence / count in G[C] ---------------------------
    def _triangle_in_G_of_mask(self, C, red: bool) -> bool:
        """
        Return True iff the induced subgraph on the common-neighbor mask C
        (in the mover's color) contains at least one triangle.
        Works with numpy or torch adjacencies.
        """
        A = self.red_adj if red else self.blue_adj

        # Turn mask into numpy indices
        idx = _mask_to_indices(C)
        m = int(len(idx))
        if m < 3:
            return False

        # Get induced adjacency as a boolean numpy array
        if isinstance(A, np.ndarray):
            sub = (A[np.ix_(idx, idx)] != 0)
        else:
            # torch or other types → convert to numpy once
            sub = np.asarray(A.cpu().numpy())[np.ix_(idx, idx)] != 0

        # Check triangles by scanning edges (u,v) and testing common neighbors
        # sub is symmetric, zero diagonal
        for u in range(m - 2):
            row_u = sub[u]
            # neighbors v of u with v>u
            vs = np.flatnonzero(row_u)
            for v in vs:
                if v <= u:
                    continue
                # any w > v that is neighbor of both u and v?
                # this avoids double/triple counting paths
                if (row_u & sub[v]).any():
                    return True
        return False

    def _k4_in_G_of_mask(self, C, red: bool) -> bool:
        """
        Return True iff the induced subgraph on the common-neighbor mask C
        (in mover's color) contains a K4 (a 4-clique).

        Fast O(|C| * deg) check using incremental neighbor intersections:
        pick a < b, require c,d both adjacent to a and b and to each other.

        Works with numpy or torch adjacency; converts to a small numpy bool submatrix.
        """
        import numpy as np
        A = self.red_adj if red else self.blue_adj

        # turn mask into indices inside the color's adjacency
        idx = _mask_to_indices(C)
        m = int(len(idx))
        if m < 4:
            return False

        # pull induced adjacency on idx as boolean numpy
        if isinstance(A, np.ndarray):
            sub = (A[np.ix_(idx, idx)] != 0)
        else:
            # torch (or similar): move to CPU once
            sub = np.asarray(A.detach().cpu().numpy())[np.ix_(idx, idx)] != 0

        # quick degree pruning: any vertex with <3 neighbors in sub cannot be in a K4
        degs = sub.sum(axis=1)
        if (degs >= 3).sum() < 4:
            return False

        # main search: choose a<b, then look for c,d in N(a)∩N(b) with edge (c,d)
        # use upper-triangle traversal to avoid duplicates
        for a in range(m - 3):
            if degs[a] < 3:
                continue
            neigh_a = sub[a]  # boolean row
            bs = np.flatnonzero(neigh_a)
            for b in bs:
                if b <= a or degs[b] < 3:
                    continue
                # candidates common to a and b, and strictly after b to keep order a<b<c<d
                inter_ab = neigh_a & sub[b]
                if inter_ab.sum() < 2:
                    continue
                # restrict to indices > b
                if b + 1 < m:
                    inter_ab[: b + 1] = False
                else:
                    continue
                if inter_ab.sum() < 2:
                    continue
                cs = np.flatnonzero(inter_ab)
                # need an edge among cs; check by incremental intersection:
                # for each c in cs, see if there exists d>c in cs with sub[c,d]=1
                for t, c in enumerate(cs[:-1]):
                    # nodes that are in inter_ab AND adjacent to c, with index > c
                    cand_cd = inter_ab & sub[c]
                    if c + 1 < m:
                        cand_cd[: c + 1] = False
                    # if any remain → edge (c,d) exists inside inter_ab ⇒ K4 found
                    if cand_cd.any():
                        return True
        return False


    def _triangle_count_in_G_of_mask(self, C, red: bool) -> int:
        """
        Count (approximately) triangles in the induced subgraph on C.
        Exact count with de-duplication by enforcing u<v<w.
        Used for shaping at r=5 (can be small and fast).
        """
        A = self.red_adj if red else self.blue_adj

        idx = _mask_to_indices(C)
        m = int(len(idx))
        if m < 3:
            return 0

        if isinstance(A, np.ndarray):
            sub = (A[np.ix_(idx, idx)] != 0)
        else:
            sub = np.asarray(A.cpu().numpy())[np.ix_(idx, idx)] != 0

        tri = 0
        for u in range(m - 2):
            row_u = sub[u]
            vs = np.flatnonzero(row_u)
            for v in vs:
                if v <= u:
                    continue
                common = (row_u & sub[v])
                if common.any():
                    # only count w > v to avoid double counting
                    tri += int(common[v + 1 :].sum())
        return tri
    # -------------------------------------------------------------------------------

    def _subgraph_edges(self, adj: np.ndarray, nodes: List[int]) -> int:
        s = 0
        for idx_a in range(len(nodes)):
            a = nodes[idx_a]
            for idx_b in range(idx_a+1, len(nodes)):
                b = nodes[idx_b]
                s += int(adj[a, b] == 1)
        return s

    def _triangle_count_in_C(self, C: List[int], red: bool) -> int:
        """Count triangles in G[C] for shaping (r=5)."""
        if len(C) < 3:
            return 0
        adj = self.red_adj if red else self.blue_adj
        count = 0
        # bitset acceleration
        # Build bitsets per node: rows as bit arrays
        bitrows = [0] * len(C)
        idx_map = {v: t for t, v in enumerate(C)}
        for t, v in enumerate(C):
            row = 0
            for u in C:
                if u != v and adj[v, u]:
                    row |= (1 << idx_map[u])
            bitrows[t] = row
        # count triangles: for each (a<b<c) check edges (a,b),(b,c),(a,c)
        m = len(C)
        for a in range(m):
            for b in range(a+1, m):
                if (bitrows[a] >> b) & 1:
                    # neighbors common to a and b among indices > b
                    ab = bitrows[a] & bitrows[b]
                    # mask to indices > b
                    mask = ~((1 << (b+1)) - 1)
                    count += ((ab & mask) != 0) * 1  # early exit variant: count triangles loosely
        return count

    def _moves_into_loss_after(self, i: int, j: int, red: bool) -> bool:
        """
        True iff coloring (i,j) in the given color immediately creates a mono-K_r
        in that color (misère loss). Uses fast r-specific checks on C = N(i)∩N(j).
        """
        C = self._common_neighbors(i, j, red=red)

        if self.r == 3:
            # triangle close with (i,j) if there exists any vertex in C
            return _count_mask(C) > 0

        if self.r == 4:
            # need an edge inside G[C] to complete a K4 with {i,j}
            A = self.red_adj if red else self.blue_adj
            idx = _mask_to_indices(C)
            return _edges_in_induced(A, idx) > 0

        if self.r == 5:
            # your existing fast triangle-in-G[C] predicate
            return self._triangle_in_G_of_mask(C, red=red)

        if self.r == 6:
            # your existing fast K4-in-G[C] predicate
            return self._k4_in_G_of_mask(C, red=red)

        raise ValueError(f"Unsupported r={self.r}; supported r∈{{3,4,5,6}}")



    def _has_triangle_in_C(self, C: List[int], red: bool) -> bool:
        if len(C) < 3:
            return False
        adj = self.red_adj if red else self.blue_adj
        # quick bitset check with early exit
        idx = {v: k for k, v in enumerate(C)}
        rows = [0] * len(C)
        for t, v in enumerate(C):
            row = 0
            for u in C:
                if u != v and adj[v, u]:
                    row |= (1 << idx[u])
            rows[t] = row
        m = len(C)
        for a in range(m):
            for b in range(a+1, m):
                if (rows[a] >> b) & 1:
                    ab = rows[a] & rows[b]
                    # any c > b?
                    mask = ~((1 << (b+1)) - 1)
                    if (ab & mask) != 0:
                        return True
        return False

    def _has_K4_in_C(self, C: List[int], red: bool) -> bool:
        if len(C) < 4:
            return False
        adj = self.red_adj if red else self.blue_adj
        # DFS with pivoting & simple deg cutoff
        Cset = set(C)
        deg = {v: int(adj[v, C].sum()) for v in C}
        ordered = sorted(C, key=lambda v: -deg[v])
        def dfs(clique: List[int], cand: List[int]) -> bool:
            if len(clique) == 4:
                return True
            # pivot: choose u with max degree among cand
            if not cand:
                return False
            u = max(cand, key=lambda x: deg[x])
            # try nodes with edge to u
            nxt = []
            for v in cand:
                # pruning: must connect to all in clique
                ok = True
                for w in clique:
                    if not adj[v, w]:
                        ok = False
                        break
                if ok and adj[v, u]:
                    nxt.append(v)
            # degree cutoff heuristic
            nxt = [v for v in nxt if deg[v] >= 3 - len(clique)]
            for v in nxt:
                if dfs(clique + [v], [w for w in nxt if w > v]):
                    return True
            return False
        return dfs([], ordered)

    def _has_Kr(self, adj: np.ndarray, r: int) -> bool:
        """Full Kr check (used only on terminal verification / SAT witness)."""
        n = adj.shape[0]
        vertices = list(range(n))
        # small r only (5 or 6)
        r_needed = r
        clique = []

        def backtrack(start: int, cand: List[int]) -> bool:
            if len(clique) == r_needed:
                return True
            for idx in range(start, len(cand)):
                v = cand[idx]
                # must connect to all in clique
                if all(adj[v, u] for u in clique):
                    clique.append(v)
                    nxt = [w for w in cand[idx + 1:] if adj[v, w]]
                    if backtrack(0, nxt):
                        return True
                    clique.pop()
            return False

        return backtrack(0, vertices)

    def _state_hash_for_sat_cache(self) -> int:
        # simple hash: color bitstring (0/1/2) flattened + r + n
        return hash((self.n, self.r, self.edge_color.tobytes()))
