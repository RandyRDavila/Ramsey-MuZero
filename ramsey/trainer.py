from dataclasses import dataclass
from typing import List, Tuple, Dict, Any
import os
import time
import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter

from .config import RamseyConfig
from .env import RamseyEnv, upper_triangle_pairs
from .model import MuZeroNetwork
from .mcts import MCTS, Node, EdgeChild
from .sat_tools import bounded_draw_completion
from .utils import ensure_dir, masked_softmax, save_witness_png_npz


try:
    from rich.console import Console as RichConsole
    from rich.panel import Panel as RichPanel
    HAVE_RICH = True
except Exception:
    HAVE_RICH = False

class ReplayBuffer:
    def __init__(self, cap:int=50000):
        self.cap = cap
        self.data = []

    def add_episode(self, episode):
        self.data.append(episode)
        if len(self.data) > self.cap:
            self.data = self.data[-self.cap:]

    def sample_batch(self, B:int):
        episodes = random.choices(self.data, k=B)
        obs, target_value, target_reward, policy_target = [], [], [], []
        for ep in episodes:
            t = random.randrange(len(ep["obs"]))
            obs.append(ep["obs"][t])
            target_value.append(ep["value"][t])
            target_reward.append(ep["reward"][t])
            policy_target.append(ep["policy"][t])
        return obs, target_value, target_reward, policy_target


class Trainer:
    def __init__(self, cfg: RamseyConfig, env: RamseyEnv, net: MuZeroNetwork):
        self.cfg = cfg
        self.env = env
        self.net = net
        self.optim = optim.Adam(self.net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        self.replay = ReplayBuffer()
        self.results_dir = cfg.results_dir
        ensure_dir(self.results_dir)
        ensure_dir(f"{self.results_dir}/checkpoints")
        ensure_dir(f"{self.results_dir}/witness")

        # ---- TensorBoard ----
        tb_dir = os.path.join(self.results_dir, "tb")
        ensure_dir(tb_dir)
        self.tb = SummaryWriter(log_dir=tb_dir)
        self.episode_idx = 0
        self.optim_steps = 0

        self.curriculum_n = cfg.curriculum_start_n
        self.draws_at_n = 0
        self.console = RichConsole() if HAVE_RICH else None
        self.best_conjecture_n = 0  # largest n with a verified draw in this run

    # -------------------- Training Loop --------------------
    def train_forever(self):
        step = 0
        while True:
            episode = self.play_one_episode(self.curriculum_n)
            self.replay.add_episode(episode)

            # ----- TensorBoard episode logs -----
            ep_len = len(episode["reward"])
            self.tb.add_scalar("episode/length", ep_len, self.episode_idx)
            self.tb.add_scalar("episode/draw", 1.0 if episode.get("draw") else 0.0, self.episode_idx)
            self.tb.add_scalar("episode/sat_attempts", episode.get("sat_attempts", 0), self.episode_idx)
            self.tb.add_scalar("curriculum/n", self.curriculum_n, self.episode_idx)
            self.episode_idx += 1
            # ------------------------------------

            if episode["draw"]:
                self.draws_at_n += 1
                if self.draws_at_n >= self.cfg.draws_needed:
                    self.curriculum_n = min(self.curriculum_n + 1, self.cfg.n_max)
                    self.env.reset(n=self.curriculum_n)
                    self.draws_at_n = 0
                    print(f"[curriculum] advancing to n={self.curriculum_n}")
                    self.tb.add_scalar("curriculum/advance", self.curriculum_n, self.episode_idx)

            # optimize a few minibatches
            for _ in range(16):
                if len(self.replay.data) < 4:
                    break
                self.update_mini()

            if step % 50 == 0:
                self.save_ckpt(step)
                self.tb.flush()
            step += 1

    # -------------------- Episode Rollout --------------------
    def play_one_episode(self, n:int):
        ep = dict(obs=[], action=[], policy=[], value=[], reward=[], done=[])
        env = self.env
        env.reset(n=n)

        # symmetry augmentation: random permutation of vertex labels for the whole episode
        perm = np.random.permutation(n)
        invperm = np.argsort(perm)

        def apply_perm_obs(board):
            red = board[0,:n,:n].copy()
            blue = board[1,:n,:n].copy()
            red = red[perm][:,perm]
            blue = blue[perm][:,perm]
            out = board.copy()
            out[0,:n,:n] = red
            out[1,:n,:n] = blue
            return out

        mcts = MCTS(self.cfg, self.net, env_like=env)
        sat_fn = lambda n_, r_, R, B, EC, time_ms, clause_cap: bounded_draw_completion(
            n_, r_, R, B, EC, time_ms=time_ms, clause_cap=clause_cap
        )

        to_play = 0  # 0=RED, 1=BLUE
        done = False
        move_count = 0
        sat_attempts = 0

        while not done:
            board = env.observation()
            board_perm = apply_perm_obs(board)
            obs_t = torch.from_numpy(board_perm).to(self.cfg.device)

            legal = env.legal_edges()
            # build legal mask (upper triangle) in permuted indexing
            legal_mask_upper = torch.zeros((n,n), dtype=torch.bool, device=self.cfg.device)
            for (i,j) in legal:
                ip, jp = perm[i], perm[j]
                ii, jj = (ip, jp) if ip < jp else (jp, ip)
                legal_mask_upper[ii, jj] = True

            # Optionally allow SAT_TRY_NOW
            sat_allowed = env.sat_action_legal()

            # Node embeddings proxy for pairwise policy
            red = obs_t[0,:n,:n]
            blue = obs_t[1,:n,:n]
            deg_r = red.sum(-1, keepdim=True)
            deg_b = blue.sum(-1, keepdim=True)
            ids = torch.arange(n, device=self.cfg.device)
            id_emb = self.net.encoder.id_emb(ids)
            H_nodes = torch.cat([deg_r, deg_b, id_emb], dim=-1)
            H_nodes = torch.relu(self.net.encoder.inp(H_nodes))

            # Pairwise logits
            pair_logits = []
            pairs = []
            scorer = nn.Sequential(
                nn.Linear(H_nodes.shape[1]*4, H_nodes.shape[1]), nn.ReLU(),
                nn.Linear(H_nodes.shape[1], 1)
            ).to(self.cfg.device)
            for i in range(n):
                for j in range(i+1, n):
                    if legal_mask_upper[i, j]:
                        hi, hj = H_nodes[i], H_nodes[j]
                        feats = torch.cat([hi, hj, (hi-hj).abs(), hi*hj], dim=-1)
                        pair_logits.append(scorer(feats))
                        pairs.append((i, j))
            if len(pair_logits) == 0 and not sat_allowed:
                break
            logits_vec = torch.cat(pair_logits, dim=0).view(-1) if pair_logits else torch.empty(0, device=self.cfg.device)
            pi_edges = torch.softmax(logits_vec.float(), dim=0) if logits_vec.numel() > 0 else None

            # Combine with SAT_TRY_NOW
            if sat_allowed:
                if pi_edges is None:
                    pi = torch.ones(1, device=self.cfg.device)
                    action_space = [("sat", -1, -1)]
                else:
                    sat_logit = torch.tensor([0.0], device=self.cfg.device)
                    logits_all = torch.cat([logits_vec.float(), sat_logit], dim=0)
                    pi = torch.softmax(logits_all, dim=0)
                    action_space = [("edge", i, j) for (i,j) in pairs] + [("sat", -1, -1)]
            else:
                pi = pi_edges
                action_space = [("edge", i, j) for (i,j) in pairs]

            # Temperature schedule by move count (simple)
            T = 1.0 if move_count < 500 else (0.5 if move_count < 1500 else 0.25)
            probs = (pi ** (1.0 / T)) / (pi ** (1.0 / T)).sum()
            a_idx = torch.multinomial(probs, num_samples=1).item()
            a_type, ii, jj = action_space[a_idx]

            ep["obs"].append(board_perm.copy())
            ep["policy"].append(probs.detach().cpu().numpy())

            if a_type == "sat":
                sat_attempts += 1
                obs2, r, done, info = env.step_sat_try_now(sat_fn)
                ep["reward"].append(r)
                if done and r > 0 and info.get("sat_draw", False):
                    self._save_witness(env)
            else:
                # map back to original labels
                invperm = np.argsort(perm)
                i = invperm[ii]; j = invperm[jj]
                if i > j:
                    i, j = j, i
                obs2, r, done, info = env.step_edge(i, j)
                ep["reward"].append(r)
                if done and r > 0.9:
                    self._save_witness(env)

            ep["done"].append(done)
            ep["value"].append(0.0)  # placeholder
            move_count += 1

        ep["draw"] = bool(self.env.last_reward > 0.9)
        ep["sat_attempts"] = sat_attempts
        return ep

    def _save_witness(self, env):
        red = env.red_adj.copy()
        blue = env.blue_adj.copy()
        stamp = int(time.time() * 1000)
        png = f"{self.results_dir}/witness/draw_{env.n}_{stamp}.png"
        npz = f"{self.results_dir}/witness/draw_{env.n}_{stamp}.npz"
        save_witness_png_npz(png, npz, red, blue, {"n": env.n, "r": env.r})

        # TB text pointer to last PNG
        self.tb.add_text("witness/last_png", png, self.episode_idx)

        # --- Conjecture print: only when this is a NEW max n ---
        if self.cfg.conjecture_print and env.n > getattr(self, "best_conjecture_n", 0):
            self.best_conjecture_n = env.n
            msg = f"Conjecture update: R({env.r},{env.r}) \u2265 {env.n + 1}  (new witness)"
            # Add to TensorBoard as a text event too
            self.tb.add_text("conjecture/update", msg, self.episode_idx)
            # Pretty terminal output
            if self.console:
                box = RichPanel.fit(
                    f"[bold white]{msg}[/bold white]\n[dim]saved → {png}[/dim]",
                    title="[green]Witness Found[/green]",
                    border_style="green",
                )
                self.console.print(box)
            else:
                print(f"[Conjecture] {msg} | saved → {png}")


    # ------------------- Optimization -------------------
    def update_mini(self):
        obs_list, v_target, r_target, pi_target = self.replay.sample_batch(self.cfg.batch_size)
        obs = torch.stack([torch.from_numpy(x).to(self.cfg.device) for x in obs_list], dim=0)
        B = obs.shape[0]

        v_logits, r_logits, h = self.net.initial_inference(obs[0])

        v_scalar = torch.zeros((B,), device=self.cfg.device)
        r_scalar = torch.tensor([t for t in r_target], device=self.cfg.device, dtype=torch.float32)
        v_dist = self.net.scalar_to_support(v_scalar)
        r_dist = self.net.scalar_to_support(r_scalar)

        v_loss = self.net.dist_ce(v_logits.expand(B, -1), v_dist)
        r_loss = self.net.dist_ce(r_logits.expand(B, -1), r_dist)
        loss = v_loss + r_loss

        self.optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.cfg.grad_clip)
        self.optim.step()

        # ---- TensorBoard loss logs ----
        self.tb.add_scalar("loss/value", float(v_loss.item()), self.optim_steps)
        self.tb.add_scalar("loss/reward", float(r_loss.item()), self.optim_steps)
        self.tb.add_scalar("loss/total", float(loss.item()), self.optim_steps)
        # learning rate (in case of schedulers later)
        for i, pg in enumerate(self.optim.param_groups):
            self.tb.add_scalar(f"opt/lr_group_{i}", float(pg.get("lr", self.cfg.lr)), self.optim_steps)
        self.optim_steps += 1

    def save_ckpt(self, step: int):
        path = f"{self.results_dir}/checkpoints/ckpt_{step}.pt"
        torch.save({"step": step, "model": self.net.state_dict()}, path)
