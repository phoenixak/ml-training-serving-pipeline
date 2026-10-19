"""Two-tower retrieval and a DLRM-style ranker.

Retrieval: an item tower (id + genres + decade) and a user tower that attends over the user's
liked items. Because users are built from their history rather than an id, a brand-new visitor
who picks a few movies gets recommendations immediately.

Ranking: the top candidates are re-scored with explicit feature interactions (DLRM): several
feature vectors are projected to a shared size, every pair is dot-producted, and an MLP scores
the result together with the dense features.
"""

import torch
from torch import nn
from torch.nn import functional as F

MAX_HISTORY = 50
RETRIEVAL_TEMPERATURE = 0.05  # same temperature the two-tower model is trained with


class ItemTower(nn.Module):
    def __init__(self, genre_matrix: torch.Tensor, decade_index: torch.Tensor, n_decades: int, dim: int = 64) -> None:
        super().__init__()
        n_items, n_genres = genre_matrix.shape
        self.register_buffer("genres", genre_matrix)
        self.register_buffer("decades", decade_index)
        self.id = nn.Embedding(n_items, dim, padding_idx=0)
        self.genre = nn.Linear(n_genres, dim, bias=False)
        self.decade = nn.Embedding(n_decades, dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
        nn.init.normal_(self.id.weight, std=0.05)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        x = self.id(idx) + self.genre(self.genres[idx]) + self.decade(self.decades[idx])
        return F.normalize(x + self.mlp(x), dim=-1)

    def all_items(self) -> torch.Tensor:
        return self.forward(torch.arange(self.genres.shape[0], device=self.genres.device))


class UserTower(nn.Module):
    """One pre-norm self-attention block over the most recent likes (newest first), then masked mean-pooling.

    Written out explicitly (not nn.TransformerEncoderLayer) so PyTorch inference and the ONNX export
    run the same maths: the built-in layer takes a fused fast path for padded batches in eval mode.
    """

    def __init__(self, dim: int = 64, heads: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        self.heads = heads
        self.pos = nn.Embedding(MAX_HISTORY, dim)
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.ffn = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
        self.drop = nn.Dropout(dropout)
        self.out = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))

    def attend(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        q, k, v = self.qkv(x).reshape(b, n, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        scores = q @ k.transpose(-1, -2) / (d // self.heads) ** 0.5
        scores = scores.masked_fill(~mask[:, None, None, :], -1e9)
        out = (self.drop(scores.softmax(-1)) @ v).transpose(1, 2).reshape(b, n, d)
        return self.proj(out)

    def forward(self, item_vecs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """item_vecs: (B, L, d) newest first; mask: (B, L) True where there is an item."""
        pos = torch.arange(item_vecs.shape[1], device=item_vecs.device)
        x = item_vecs + self.pos(pos)[None]
        x = x + self.drop(self.attend(self.norm1(x), mask))
        x = x + self.drop(self.ffn(self.norm2(x)))
        m = mask.unsqueeze(-1).float()
        pooled = (x * m).sum(1) / m.sum(1).clamp(min=1.0)
        return F.normalize(pooled + self.out(pooled), dim=-1)


class TwoTower(nn.Module):
    def __init__(self, genre_matrix, decade_index, n_decades, dim: int = 64) -> None:
        super().__init__()
        self.items = ItemTower(genre_matrix, decade_index, n_decades, dim)
        self.users = UserTower(dim)

    def user_vectors(self, history: torch.Tensor) -> torch.Tensor:
        """history: (B, L) item indexes, newest first, 0 = padding."""
        return self.users(self.items(history), history > 0)


class DLRMRanker(nn.Module):
    """Pairwise dot-product interactions between projected feature vectors (DLRM), then an MLP."""

    N_DENSE = 7

    def __init__(self, n_items: int, n_genres: int, n_decades: int, tower_dim: int = 64, dim: int = 16) -> None:
        super().__init__()
        self.item_id = nn.Embedding(n_items, dim, padding_idx=0)
        self.genre = nn.Linear(n_genres, dim, bias=False)
        self.decade = nn.Embedding(n_decades, dim)
        self.user_proj = nn.Linear(tower_dim, dim)
        self.item_proj = nn.Linear(tower_dim, dim)
        self.bottom = nn.Sequential(nn.Linear(self.N_DENSE, 32), nn.ReLU(), nn.Linear(32, dim))
        n_vec = 6
        iu = torch.triu_indices(n_vec, n_vec, offset=1)  # every pair of feature vectors, computed once
        self.register_buffer("pair_i", iu[0], persistent=False)
        self.register_buffer("pair_j", iu[1], persistent=False)
        self.top = nn.Sequential(
            nn.Linear(n_vec * (n_vec - 1) // 2 + dim, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1)
        )
        # residual ranking: the score is the retrieval logit plus a learned correction, and the correction
        # starts at zero, so an untrained ranker reproduces the retrieval order exactly
        nn.init.zeros_(self.top[-1].weight)
        nn.init.zeros_(self.top[-1].bias)

    def forward(self, item_idx, item_genres, item_decade, user_vec, item_vec, dense) -> torch.Tensor:
        d = self.bottom(dense)
        vecs = torch.stack(
            [
                self.item_id(item_idx),
                self.genre(item_genres),
                self.decade(item_decade),
                self.user_proj(user_vec),
                self.item_proj(item_vec),
                d,
            ],
            dim=1,
        )  # (B, 6, dim)
        inter = vecs @ vecs.transpose(1, 2)
        correction = self.top(torch.cat([inter[:, self.pair_i, self.pair_j], d], dim=1)).squeeze(-1)
        return dense[:, 0] / RETRIEVAL_TEMPERATURE + correction  # dense[:, 0] is the retrieval cosine


def in_batch_softmax_loss(
    user: torch.Tensor, item: torch.Tensor, targets: torch.Tensor, log_pop: torch.Tensor, temperature: float = 0.05
) -> torch.Tensor:
    """Sampled softmax over the batch's own targets, with logQ correction for popularity bias."""
    logits = user @ item.T / temperature - log_pop[targets][None, :]
    same = targets[None, :] == targets[:, None]
    logits = logits.masked_fill(same & ~torch.eye(len(targets), dtype=torch.bool, device=logits.device), float("-inf"))
    return F.cross_entropy(logits, torch.arange(len(targets), device=logits.device))
