# models_gnn.py
from __future__ import annotations
import math
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

# Utilities: value/reward distributional support
def support_to_scalar(logits: torch.Tensor, support_size: int) -> torch.Tensor:
    # logits: (B, 2S+1)
    probs = logits.softmax(dim=-1)
    support = torch.arange(-support_size, support_size + 1, device=logits.device, dtype=probs.dtype)
    return (probs * support).sum(dim=-1, keepdim=True)

def scalar_to_support(x: torch.Tensor, support_size: int) -> torch.Tensor:
    # typically the engine handles targets; this helper is here for completeness
    x = torch.clamp(x, -support_size, support_size)
    return x

def _safe_softmax_masked(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # logits, mask same shape; mask==0 means illegal
    x = logits.float()
    big_neg = torch.finfo(x.dtype).min if x.dtype in (torch.float32, torch.float64) else -1e9
    x = x + (mask.float() - 1.0) * 1e9  # mask=1 keeps, mask=0 adds -1e9
    return F.softmax(x, dim=-1)

# Edge indexing helpers for fixed N_MAX upper triangle
def _edge_count(nmax: int) -> int:
    return (nmax * (nmax - 1)) // 2

def _enumerate_pairs(nmax: int, device=None):
    # returns (E,2) tensor of (i,j) pairs i<j
    pairs = []
    for i in range(nmax - 1):
        for j in range(i + 1, nmax):
            pairs.append((i, j))
    return torch.tensor(pairs, dtype=torch.long, device=device)

class MessageLayer(nn.Module):
    def __init__(self, d_in: int, d_hid: int):
        super().__init__()
        self.W_red = nn.Linear(d_in, d_hid, bias=False)
        self.W_blue = nn.Linear(d_in, d_hid, bias=False)
        self.proj = nn.Linear(d_in + d_hid, d_hid)
        self.ln = nn.LayerNorm(d_hid)

    def forward(self, H, A_red, A_blue):
        # Row-normalized neighbor aggregation
        deg_r = torch.clamp(A_red.sum(-1, keepdim=True), min=1.0)
        deg_b = torch.clamp(A_blue.sum(-1, keepdim=True), min=1.0)
        m_r = A_red @ self.W_red(H) / deg_r
        m_b = A_blue @ self.W_blue(H) / deg_b
        out = torch.relu(self.ln(self.proj(torch.cat([H, m_r + m_b], dim=-1))))
        return out

class PairwiseScorer(nn.Module):
    def __init__(self, d: int, hidden: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(4 * d, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, H, pairs):
        # H: (B,N,D), pairs: (E,2)
        i = pairs[:, 0]; j = pairs[:, 1]
        Hi = H[:, i, :]  # (B,E,D)
        Hj = H[:, j, :]
        feat = torch.cat([Hi, Hj, torch.abs(Hi - Hj), Hi * Hj], dim=-1)  # (B,E,4D)
        s = self.mlp(feat).squeeze(-1)  # (B,E)
        return s

class GNNRepresentation(nn.Module):
    def __init__(self, nmax: int, d_node: int = 96, layers: int = 3, d_latent: int = 256):
        super().__init__()
        self.nmax = nmax
        self.id_embed = nn.Embedding(nmax, 32)
        self.in_proj = nn.Linear(2 + 32, d_node)  # deg_red, deg_blue, id_emb
        self.layers = nn.ModuleList([MessageLayer(d_node, d_node) for _ in range(layers)])
        self.readout = nn.Sequential(
            nn.Linear(d_node + 1, 256), nn.ReLU(),
            nn.Linear(256, d_latent),
        )

    def forward(self, obs):
        # obs: (B,3,N,N) float32 (0:red adj, 1:blue adj, 2:turn plane {0,1})
        B, C, N, _ = obs.shape
        assert N == self.nmax, "obs N must equal configured N_MAX"
        A_red = obs[:, 0]
        A_blue = obs[:, 1]
        turn = obs[:, 2, 0, 0:1]  # (B,1)

        # degrees
        deg_r = A_red.sum(-1)  # (B,N)
        deg_b = A_blue.sum(-1)

        ids = torch.arange(N, device=obs.device).unsqueeze(0).repeat(B, 1)  # (B,N)
        id_emb = self.id_embed(ids)  # (B,N,32)
        H0 = torch.cat([deg_r.unsqueeze(-1), deg_b.unsqueeze(-1), id_emb], dim=-1)  # (B,N,34)
        H = torch.relu(self.in_proj(H0))  # (B,N,D)

        for layer in self.layers:
            H = layer(H, A_red, A_blue)  # (B,N,D)

        # mean pool + turn scalar
        g = H.mean(dim=1)  # (B,D)
        g = torch.cat([g, turn], dim=-1)  # (B,D+1)
        z = self.readout(g)  # (B,latent)
        return z, H  # return node states for policy scorer

class GNNPrediction(nn.Module):
    def __init__(self, latent: int, support_size: int, nmax: int, d_node: int = 96):
        super().__init__()
        self.support_size = support_size
        self.val_head = nn.Sequential(
            nn.Linear(latent, 256), nn.ReLU(),
            nn.Linear(256, 2 * support_size + 1),
        )
        self.rew_head = nn.Sequential(
            nn.Linear(latent, 256), nn.ReLU(),
            nn.Linear(256, 2 * support_size + 1),
        )
        self.nmax = nmax
        self.pairs = _enumerate_pairs(nmax)
        self.scorer = PairwiseScorer(d=d_node, hidden=128)
        self.sat_head = nn.Linear(latent, 1)

    def forward(self, z, H):
        # Value & reward distributions
        v_logits = self.val_head(z)
        r_logits = self.rew_head(z)
        # Policy for edges via pairwise scorer from node states
        if H is None:
            raise RuntimeError("Node states required for pairwise policy")
        device = z.device
        if self.pairs.device != device:
            self.pairs = self.pairs.to(device)
        edge_logits = self.scorer(H, self.pairs)  # (B,E)
        sat_logit = self.sat_head(z).squeeze(-1)  # (B,)
        policy_logits = torch.cat([edge_logits, sat_logit.unsqueeze(-1)], dim=-1)  # (B,E+1)
        return v_logits, r_logits, policy_logits

class GNNDynamics(nn.Module):
    def __init__(self, latent: int, nmax: int, support_size: int):
        super().__init__()
        self.nmax = nmax
        self.support_size = support_size
        self.node_emb = nn.Embedding(nmax, 64)
        self.color_emb = nn.Embedding(3, 16)  # 0:edge, 1:sat, 2:other
        self.mlp = nn.Sequential(
            nn.Linear(latent + 64 * 2 + 16, 256), nn.ReLU(),
            nn.Linear(256, latent),
        )
        self.rew_head = nn.Sequential(
            nn.Linear(latent, 256), nn.ReLU(),
            nn.Linear(256, 2 * support_size + 1),
        )

    def forward(self, z, action_idx: torch.Tensor):
        # action_idx: (B,) in [0..E] where E=_edge_count(nmax), E==SAT
        B = action_idx.shape[0]
        E = _edge_count(self.nmax)
        device = z.device
        # decode edge or SAT
        ij = action_idx.clamp(0, E)  # keep in range
        # build (i,j) for all actions; SAT uses zeros
        # fast integer math: precompute triangular counts
        # Simple loop (B small in MuZero)
        I = torch.zeros(B, dtype=torch.long, device=device)
        J = torch.zeros(B, dtype=torch.long, device=device)
        for b in range(B):
            a = int(ij[b].item())
            if a == E:
                I[b] = 0; J[b] = 0
            else:
                # invert enumeration
                # row by row
                left = a
                for i in range(self.nmax - 1):
                    row = self.nmax - i - 1
                    if left < row:
                        I[b] = i
                        J[b] = i + 1 + left
                        break
                    left -= row

        i_emb = self.node_emb(I)  # (B,64)
        j_emb = self.node_emb(J)  # (B,64)
        col = torch.where(ij == E, torch.ones_like(ij), torch.zeros_like(ij))  # 1 if SAT
        c_emb = self.color_emb(col)  # (B,16)

        h = torch.cat([z, i_emb, j_emb, c_emb], dim=-1)
        z_next = self.mlp(h)
        r_logits = self.rew_head(z_next)
        return z_next, r_logits

class GNNNetwork(nn.Module):
    """
    Minimal MuZero-compatible GNN network. Plugged by:
      if config.network == "gnn": return GNNNetwork(config)
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        nmax = config.observation_shape[1]
        self.repr = GNNRepresentation(nmax=nmax, d_node=96, layers=3, d_latent=256)
        self.pred = GNNPrediction(latent=256, support_size=config.support_size, nmax=nmax, d_node=96)
        self.dyn = GNNDynamics(latent=256, nmax=nmax, support_size=config.support_size)

    # MuZero API
    def initial_inference(self, observation: torch.Tensor):
        # observation: (B, C, N, N)
        z, H = self.repr(observation)
        v_logits, r_logits, p_logits = self.pred(z, H)
        return (v_logits, r_logits, p_logits, z)

    def recurrent_inference(self, hidden_state: torch.Tensor, action: torch.Tensor):
        # hidden_state: (B, latent), action: (B,)
        z_next, r_logits = self.dyn(hidden_state, action)
        # prediction uses z_next; we don't have node states after dynamics, pass H=None
        v_logits, _, p_logits = self.pred(z_next, H=None)  # reward from dyn
        return (v_logits, r_logits, p_logits, z_next)
