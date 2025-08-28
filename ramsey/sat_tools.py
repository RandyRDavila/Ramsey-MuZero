# ramsey/sat_tools.py
from typing import Tuple, Dict, List, Optional
import time
import numpy as np

try:
    from pysat.formula import CNF
    from pysat.solvers import Minisat22
except Exception:
    CNF = None
    Minisat22 = None


def _edge_var(i: int, j: int, n: int):
    """Two vars per edge (i<j): xr (red), xb (blue). Force built-in ints."""
    i = int(i); j = int(j); n = int(n)
    idx = int(i * n + j)
    xr = int(2 * idx + 1)
    xb = int(2 * idx + 2)
    return xr, xb


def _append_int_clause(cnf: CNF, lits):
    cnf.append([int(x) for x in lits])


def _find_Kr(adj: np.ndarray, r: int) -> Optional[List[int]]:
    """Return a list of vertices forming a K_r, or None."""
    n = adj.shape[0]
    cand = list(range(n))
    clique: List[int] = []

    def backtrack(cset: List[int], start: int, remain: List[int]) -> Optional[List[int]]:
        if len(cset) == r:
            return cset[:]
        # simple pruning: need enough vertices left
        need = r - len(cset)
        if len(remain) < need:
            return None
        for idx in range(len(remain)):
            v = remain[idx]
            ok = True
            for u in cset:
                if not adj[v, u]:
                    ok = False
                    break
            if not ok:
                continue
            nxt = [w for w in remain[idx+1:] if adj[v, w]]
            cset.append(v)
            got = backtrack(cset, 0, nxt)
            if got is not None:
                return got
            cset.pop()
        return None

    return backtrack(clique, 0, cand)


def bounded_draw_completion(n: int, r: int, red_adj: np.ndarray, blue_adj: np.ndarray,
                            edge_color: np.ndarray, time_ms: int = 5000, clause_cap: int = 200000
                            ) -> Tuple[bool, np.ndarray, np.ndarray, Dict]:
    """
    Try to SAT-complete remaining edges to avoid mono-Kr in both colors.
    Heuristic seed of anti-mono-Kr clauses, then a lazy loop:
      solve -> verify -> if mono-Kr found, add a blocking clause and re-solve (budgeted).
    Returns: (ok, red_full, blue_full, meta)
    """
    t0 = time.time()
    if CNF is None or Minisat22 is None:
        return False, red_adj, blue_adj, {"reason": "pysat_not_available"}

    cnf = CNF()
    clause_count = 0
    n_int = int(n)

    # Per-edge variables & exactly-one for uncolored
    uncolored = []
    for i in range(n_int):
        for j in range(i + 1, n_int):
            xr, xb = _edge_var(i, j, n_int)
            c = int(edge_color[i, j])
            if c == 1:
                _append_int_clause(cnf, [xr]); clause_count += 1
            elif c == 2:
                _append_int_clause(cnf, [xb]); clause_count += 1
            else:
                _append_int_clause(cnf, [xr, xb]); clause_count += 1
                _append_int_clause(cnf, [-xr, -xb]); clause_count += 1
                uncolored.append((i, j, xr, xb))

    # Heuristic anti-mono-Kr seeds
    import random, itertools
    vertices = list(range(n_int))
    deg_red = red_adj.sum(1)
    deg_blue = blue_adj.sum(1)
    hot = sorted(vertices, key=lambda v: int(deg_red[v] + deg_blue[v]), reverse=True)[:min(n_int, 12)]

    def add_no_mono_clique(S):
        nonlocal clause_count
        xr_list, xb_list = [], []
        for a in range(len(S)):
            for b in range(a + 1, len(S)):
                i, j = int(S[a]), int(S[b])
                if i > j: i, j = j, i
                xr, xb = _edge_var(i, j, n_int)
                xr_list.append(xr); xb_list.append(xb)
        _append_int_clause(cnf, xb_list); clause_count += 1  # forbid all-red
        _append_int_clause(cnf, xr_list); clause_count += 1  # forbid all-blue

    seeds = []
    if r == 5:
        for v in hot:
            nbrs = list(set(np.where((red_adj[v] | blue_adj[v]) > 0)[0].tolist()))
            if len(nbrs) >= 4:
                picks = random.sample(nbrs, min(len(nbrs), 7))
                for a, b, c, d in itertools.combinations(picks, 4):
                    seeds.append(tuple(sorted({int(v), int(a), int(b), int(c), int(d)})))
        while len(seeds) < 200 and len(vertices) >= 5:
            seeds.append(tuple(sorted(random.sample(vertices, 5))))
    else:  # r == 6
        for v in hot:
            nbrs = list(set(np.where((red_adj[v] | blue_adj[v]) > 0)[0].tolist()))
            if len(nbrs) >= 5:
                picks = random.sample(nbrs, min(len(nbrs), 8))
                for a, b, c, d, e in itertools.combinations(picks, 5):
                    seeds.append(tuple(sorted({int(v), int(a), int(b), int(c), int(d), int(e)})))
        while len(seeds) < 200 and len(vertices) >= 6:
            seeds.append(tuple(sorted(random.sample(vertices, 6))))

    # Dedup seeds
    seen = set(); uniq = []
    for S in seeds:
        if S not in seen:
            uniq.append(S); seen.add(S)
    for S in uniq:
        add_no_mono_clique(S)
        if clause_count >= int(clause_cap):
            break

    # --- Solve with lazy verification & blocking ---
    def build_full_from_model(model_set):
        redF = red_adj.copy()
        blueF = blue_adj.copy()
        for i in range(n_int):
            for j in range(i + 1, n_int):
                if int(edge_color[i, j]) == 0:
                    xr, xb = _edge_var(i, j, n_int)
                    if xr in model_set:
                        redF[i, j] = redF[j, i] = 1
                    else:
                        blueF[i, j] = blueF[j, i] = 1
        return redF, blueF

    block_adds = 0
    time_limit = (time_ms / 1000.0) if time_ms else None

    with Minisat22(bootstrap_with=cnf.clauses) as m:
        while True:
            if time_limit is not None and (time.time() - t0) > time_limit:
                return False, red_adj, blue_adj, {"sat": False, "reason": "timeout", "clauses": clause_count, "blocks": block_adds, "time_ms": int((time.time()-t0)*1000)}
            # conflict budget as crude time control
            if time_ms is not None:
                m.conf_budget(int(max(1000, time_ms)))
            sat = m.solve()
            if not sat:
                return False, red_adj, blue_adj, {"sat": False, "reason": "unsat", "clauses": clause_count, "blocks": block_adds, "time_ms": int((time.time()-t0)*1000)}
            model = set(int(v) for v in (m.get_model() or []))
            redF, blueF = build_full_from_model(model)

            # Verify
            Rk = _find_Kr(redF, r)
            if Rk is not None:
                # add blocking clause: at least one edge among Rk must be BLUE
                xb_list = []
                for a in range(len(Rk)):
                    for b in range(a+1, len(Rk)):
                        i, j = Rk[a], Rk[b]
                        if i > j: i, j = j, i
                        _, xb = _edge_var(i, j, n_int)
                        xb_list.append(int(xb))
                m.add_clause(xb_list)
                clause_count += 1
                block_adds += 1
                continue

            Bk = _find_Kr(blueF, r)
            if Bk is not None:
                # add blocking clause: at least one edge among Bk must be RED
                xr_list = []
                for a in range(len(Bk)):
                    for b in range(a+1, len(Bk)):
                        i, j = Bk[a], Bk[b]
                        if i > j: i, j = j, i
                        xr, _ = _edge_var(i, j, n_int)
                        xr_list.append(int(xr))
                m.add_clause(xr_list)
                clause_count += 1
                block_adds += 1
                continue

            # Passed verification
            return True, redF, blueF, {"sat": True, "clauses": clause_count, "blocks": block_adds, "time_ms": int((time.time()-t0)*1000)}
