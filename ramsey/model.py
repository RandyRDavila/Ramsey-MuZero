from dataclasses import dataclass
from typing import Tuple, Dict, List
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import RamseyConfig
from .utils import masked_softmax


class LayerNormRelu(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.ln = nn.LayerNorm(d)
    def forward(self, x):
        return F.relu(self.ln(x))


class EdgeMP(nn.Module):
    """Two-edge message passing with separate red/blue weights, row-normalized."""
    def __init__(self, d_in: int, d_h: int):
        super().__init__()
        self.Wself = nn.Linear(d_in, d_h)
        self.Wr = nn.Linear(d_in, d_h)
        self.Wb = nn.Linear(d_in, d_h)
        self.out = nn.Linear(3*d_h, d_h)
        self.act = LayerNormRelu(d_h)

    def forward(self, X, red_adj, blue_adj):
        # X: [n, d_in]
        n = X.shape[0]
        # row-normalize adj
        red_deg = red_adj.sum(-1, keepdim=True).clamp(min=1)
        blue_deg = blue_adj.sum(-1, keepdim=True).clamp(min=1)
        redP = red_adj / red_deg
        blueP = blue_adj / blue_deg

        m_self = self.Wself(X)
        m_red = self.Wr(torch.matmul(redP, X))
        m_blue = self.Wb(torch.matmul(blueP, X))
        h = torch.cat([m_self, m_red, m_blue], dim=-1)
        return self.act(self.out(h))


class GNNEncoder(nn.Module):
    def __init__(self, n_max: int, d_id: int = 16, d_h: int = 128, layers: int = 3):
        super().__init__()
        self.n_max = n_max
        self.id_emb = nn.Embedding(n_max, d_id)
        self.inp = nn.Linear(2 + d_id, d_h)
        self.gnn = nn.ModuleList([EdgeMP(d_h, d_h) for _ in range(layers)])
        self.readout = nn.Sequential(
            nn.Linear(d_h + 1, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU()
        )

    def forward(self, red_adj, blue_adj, turn_scalar: torch.Tensor):
        """
        red_adj, blue_adj: [n, n], float (0/1)
        turn_scalar: float tensor shape []
        """
        n = red_adj.shape[0]
        deg_r = red_adj.sum(-1, keepdim=True)
        deg_b = blue_adj.sum(-1, keepdim=True)
        ids = torch.arange(n, device=red_adj.device)
        ide = self.id_emb(ids)  # [n, d_id]
        X = torch.cat([deg_r, deg_b, ide], dim=-1)
        X = F.relu(self.inp(X))
        for layer in self.gnn:
            X = X + layer(X, red_adj, blue_adj)
        # pool
        H = X.mean(dim=0, keepdim=True)  # [1, d]
        h = torch.cat([H, turn_scalar.view(1, 1)], dim=-1)
        return self.readout(h)  # [1, 256], latent


class PairwisePolicyHead(nn.Module):
    """Compute logits for each uncolored edge via pairwise scoring over node embeddings."""
    def __init__(self, d_h: int, d_node: int = 128):
        super().__init__()
        self.W = nn.Linear(d_node, d_node, bias=False)
        self.scorer = nn.Sequential(
            nn.Linear(4 * d_node, d_node), nn.ReLU(),
            nn.Linear(d_node, 1)
        )

    def forward(self, H_nodes, legal_mask_upper):
        # H_nodes: [n, d], we need pairwise scores for i<j
        n, d = H_nodes.shape
        # Bilinear + MLP combo. For simplicity, we use only MLP over [Hi, Hj, |Hi-Hj|, Hi*Hj].
        scores = []
        pairs = []
        for i in range(n):
            for j in range(i+1, n):
                if legal_mask_upper[i, j]:
                    hi = H_nodes[i]
                    hj = H_nodes[j]
                    feats = torch.cat([hi, hj, (hi-hj).abs(), hi*hj], dim=-1)
                    logit = self.scorer(feats)
                    scores.append(logit)
                    pairs.append((i, j))
        if len(scores) == 0:
            return torch.empty(0, device=H_nodes.device), []
        return torch.cat(scores, dim=0).view(-1), pairs


class MuZeroNetwork(nn.Module):
    def __init__(self, cfg: RamseyConfig):
        super().__init__()
        self.cfg = cfg
        self.n_max = cfg.n_max
        self.support_S = cfg.support_S

        self.encoder = GNNEncoder(n_max=cfg.n_max, d_id=16, d_h=128, layers=3)
        # Node projector to expose node embeddings for policy
        self.node_readout = nn.Identity()  # reuse pre-gnn X; for simplicity we re-encode small graph in representation()

        self.value_head = self._dist_head(256)
        self.reward_head = self._dist_head(256)
        # dynamics
        self.action_emb = nn.Embedding(self.n_max * self.n_max * 3, 64)  # (i,j,color) coarse id
        self.dynamics = nn.Sequential(nn.Linear(256 + 64, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU())
        # policy scorer will be run outside using node embeddings (re-encode per observation)

    def support(self):
        S = self.support_S
        return torch.arange(-S, S+1, device=self.cfg.device, dtype=torch.float32)

    def _dist_head(self, d):
        S = 2 * self.support_S + 1
        return nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, S))

    # ----------- Representation -----------
    def initial_inference(self, obs: torch.Tensor):
        """
        obs: [3, Nmax, Nmax]
        returns: (policy_logits (deferred), value_logits, reward_logits, latent h)
        We compute value/reward from latent; policy computed later with pairwise scorer for legal edges.
        """
        red = obs[0, :,:]
        blue = obs[1, :,:]
        # derive n
        n = int((red.sum() + blue.sum() > 0).nonzero().numel() > 0)  # not robust; better pass n externally
        # safer: detect n by leading non-zero in turn plane block
        # we assume trainer passes (red, blue) and n separately for speed.
        turn_scalar = torch.tensor(float(obs[2,0,0].item()), device=obs.device)
        # For simplicity, compute with actual n from trainer; here assume full Nmax with zeros okay.
        h = self.encoder(red, blue, turn_scalar)  # [1,256]
        v = self.value_head(h).squeeze(0)
        r = self.reward_head(h).squeeze(0)
        return v, r, h

    def recurrent_inference(self, h: torch.Tensor, action_id: int):
        aemb = self.action_emb.weight[action_id].unsqueeze(0)
        h2 = self.dynamics(torch.cat([h, aemb], dim=-1))
        v = self.value_head(h2).squeeze(0)
        r = self.reward_head(h2).squeeze(0)
        return v, r, h2

    # ------ helpers ------
    def scalar_to_support(self, x: torch.Tensor):
        S = self.support()
        x = x.clamp(S[0], S[-1])
        # nearest integer projection on support
        idx = torch.round(x - S[0]).long()
        onehot = torch.zeros((*x.shape, S.numel()), device=x.device)
        onehot.scatter_(-1, idx.unsqueeze(-1), 1.0)
        return onehot

    def dist_ce(self, logits: torch.Tensor, target_dist: torch.Tensor):
        return -(target_dist * torch.log_softmax(logits.float(), dim=-1)).sum(-1).mean()
