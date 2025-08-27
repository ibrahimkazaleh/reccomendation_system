"""
Hybrid Recommender (Collaborative + Content)
-------------------------------------------
- Extends your NCF-style collaborative model with item-content features:
  * year_of_make (numeric)
  * manufacturer (categorical)
  * part_name (categorical/text label)

Design goals:
- Modular, clean architecture for future growth
- Backward-compatible data pipeline (ratings + orders)
- Cold-start friendly: can score unseen items using content tower

Main components:
- IdMaps: mapping between raw ids and contiguous indices
- Interactions preparation (explicit + implicit + negatives)
- ItemFeatureEncoder: builds dense tensors for item metadata
- HybridNCF: two-tower fusion (user/item embeddings + item content tower)
- RecommenderSystemHybrid: trainer & inference

Note: Replace any TODOs with your I/O as needed.
"""

from __future__ import annotations
import math
import json
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ===============================================================
# Utilities & Mappings
# ===============================================================

@dataclass
class IdMaps:
    user2idx: Dict[Any, int]
    item2idx: Dict[Any, int]
    idx2user: Dict[int, Any]
    idx2item: Dict[int, Any]


def build_id_maps(ratings_df: pd.DataFrame, orders_df: pd.DataFrame) -> IdMaps:
    user_ids = pd.Index(pd.concat([ratings_df["user_id"], orders_df["user_id"]]).unique())
    item_ids = pd.Index(pd.concat([ratings_df["item_id"], orders_df["item_id"]]).unique())
    user2idx = {u: i for i, u in enumerate(user_ids)}
    item2idx = {p: i for i, p in enumerate(item_ids)}
    idx2user = {i: u for u, i in user2idx.items()}
    idx2item = {i: p for p, i in item2idx.items()}
    return IdMaps(user2idx, item2idx, idx2user, idx2item)


def normalize_rating(r, min_r=0.0, max_r=5.0):
    return (r - min_r) / (max_r - min_r + 1e-8)


# ===============================================================
# Interactions (explicit + implicit)
# ===============================================================

def prepare_interactions(
    ratings_df: pd.DataFrame,
    orders_df: pd.DataFrame,
    idmaps: IdMaps,
) -> pd.DataFrame:
    rows = []
    if not ratings_df.empty:
        for _, row in ratings_df.iterrows():
            if row["user_id"] in idmaps.user2idx and row["item_id"] in idmaps.item2idx:
                rows.append({
                    "user_idx": idmaps.user2idx[row["user_id"]],
                    "item_idx": idmaps.item2idx[row["item_id"]],
                    "y_explicit": normalize_rating(float(row["rating"])),
                    "y_implicit": np.nan,
                    "seen": True,
                })
    if not orders_df.empty:
        agg = orders_df.groupby(["user_id", "item_id"]).size().reset_index(name="count")
        for _, row in agg.iterrows():
            if row["user_id"] in idmaps.user2idx and row["item_id"] in idmaps.item2idx:
                rows.append({
                    "user_idx": idmaps.user2idx[row["user_id"]],
                    "item_idx": idmaps.item2idx[row["item_id"]],
                    "y_explicit": np.nan,
                    "y_implicit": 1.0,
                    "seen": True,
                })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = (
            df.groupby(["user_idx", "item_idx"], as_index=False)
              .agg({"y_explicit": "max", "y_implicit": "max", "seen": "max"})
        )
    return df


def build_seen_sets(inter_df: pd.DataFrame) -> Dict[int, set]:
    seen = {}
    for _, r in inter_df.iterrows():
        seen.setdefault(int(r["user_idx"]), set()).add(int(r["item_idx"]))
    return seen


# ===============================================================
# Item Metadata Encoding
# ===============================================================

@dataclass
class ItemFeatureSpaces:
    n_manufacturer: int
    n_part_name: int


class ItemFeatureEncoder:
    """Encodes item metadata -> dense tensors
    Inputs DataFrame schema (example):
      item_id, year_of_make, manufacturer, part_name
    """

    def __init__(self, items_df: pd.DataFrame, idmaps: IdMaps):
        self.idmaps = idmaps

        # Build vocabularies for categorical fields
        manu_vals = items_df["manufacturer"].fillna("<UNK>").astype(str).unique()
        part_vals = items_df["part_name"].fillna("<UNK>").astype(str).unique()
        self.manufacturer2idx = {v: i for i, v in enumerate(manu_vals)}
        self.partname2idx = {v: i for i, v in enumerate(part_vals)}

        self.idx2manufacturer = {i: v for v, i in self.manufacturer2idx.items()}
        self.idx2partname = {i: v for v, i in self.partname2idx.items()}

        # Year normalization (min-max)
        y = items_df["year_of_make"].fillna(items_df["year_of_make"].median())
        self.year_min = float(np.nanmin(y)) if len(y) else 0.0
        self.year_max = float(np.nanmax(y)) if len(y) else 1.0

        # Build a per-item feature table aligned to item_idx
        n_items = len(idmaps.item2idx)
        self.features = {
            "year": np.zeros((n_items, 1), dtype=np.float32),
            "manufacturer": np.zeros((n_items,), dtype=np.int64),
            "partname": np.zeros((n_items,), dtype=np.int64),
        }

        # Fill rows
        by_id = items_df.set_index("item_id")
        for raw_item_id, item_idx in idmaps.item2idx.items():
            if raw_item_id in by_id.index:
                row = by_id.loc[raw_item_id]
                year = row.get("year_of_make", np.nan)
                manu = str(row.get("manufacturer", "<UNK>"))
                pname = str(row.get("part_name", "<UNK>"))
            else:
                year = np.nan
                manu = "<UNK>"
                pname = "<UNK>"

            year_norm = self._norm_year(year)
            manu_idx = self.manufacturer2idx.get(manu, 0)
            pname_idx = self.partname2idx.get(pname, 0)

            self.features["year"][item_idx, 0] = year_norm
            self.features["manufacturer"][item_idx] = manu_idx
            self.features["partname"][item_idx] = pname_idx

        self.spaces = ItemFeatureSpaces(
            n_manufacturer=len(self.manufacturer2idx),
            n_part_name=len(self.partname2idx),
        )

    def _norm_year(self, y):
        if pd.isna(y):
            y = (self.year_min + self.year_max) * 0.5
        if self.year_max == self.year_min:
            return 0.0
        return float((y - self.year_min) / (self.year_max - self.year_min))

    def get_all_tensors(self, device: str):
        return {
            "year": torch.from_numpy(self.features["year"]).to(device),
            "manufacturer": torch.from_numpy(self.features["manufacturer"]).to(device),
            "partname": torch.from_numpy(self.features["partname"]).to(device),
        }


# ===============================================================
# Dataset (with negatives for implicit)
# ===============================================================

class CollabDataset(Dataset):
    def __init__(self, inter_df: pd.DataFrame, num_items: int, neg_ratio: float = 1.0):
        self.num_items = num_items
        self.pos = inter_df.copy()
        self.pos["has_explicit"] = (~self.pos["y_explicit"].isna()).astype(np.float32)
        self.pos["has_implicit"] = (~self.pos["y_implicit"].isna()).astype(np.float32)
        self.seen = build_seen_sets(self.pos)

        neg_rows = []
        total_pos_impl = int(self.pos["has_implicit"].sum())
        num_negs = int(neg_ratio * total_pos_impl)
        rng = np.random.default_rng(42)
        users = list(self.seen.keys()) if self.seen else []

        for _ in range(num_negs):
            if not users:
                break
            u = int(rng.choice(users))
            # sample unseen item
            tries = 0
            while True:
                i = int(rng.integers(0, self.num_items))
                if i not in self.seen[u]:
                    break
                tries += 1
                if tries > 50:
                    # fallback in dense users
                    unseen = list(set(range(self.num_items)) - set(self.seen[u]))
                    if unseen:
                        i = int(rng.choice(unseen))
                    break
            neg_rows.append({
                "user_idx": u, "item_idx": i,
                "y_explicit": np.nan, "y_implicit": 0.0,
                "has_explicit": 0.0, "has_implicit": 1.0, "seen": False,
            })

        self.all = pd.concat([self.pos, pd.DataFrame(neg_rows)], ignore_index=True) if neg_rows else self.pos
        self.all = self.all.sample(frac=1.0, random_state=123).reset_index(drop=True)

    def __len__(self):
        return len(self.all)

    def __getitem__(self, idx):
        r = self.all.iloc[idx]
        return {
            "user": torch.tensor(int(r["user_idx"]), dtype=torch.long),
            "item": torch.tensor(int(r["item_idx"]), dtype=torch.long),
            "y_explicit": torch.tensor(0.0 if math.isnan(r["y_explicit"]) else float(r["y_explicit"]), dtype=torch.float32),
            "y_implicit": torch.tensor(0.0 if math.isnan(r["y_implicit"]) else float(r["y_implicit"]), dtype=torch.float32),
            "has_explicit": torch.tensor(float(r["has_explicit"]), dtype=torch.float32),
            "has_implicit": torch.tensor(float(r["has_implicit"]), dtype=torch.float32),
        }


# ===============================================================
# Model: Hybrid NCF (Collaborative + Content)
# ===============================================================

class ItemContentTower(nn.Module):
    def __init__(self, spaces: ItemFeatureSpaces, dim: int = 64, year_dim: int = 8, manu_dim: int = 32, part_dim: int = 32):
        super().__init__()
        # Embeddings for categorical
        self.emb_manu = nn.Embedding(spaces.n_manufacturer, manu_dim)
        self.emb_part = nn.Embedding(spaces.n_part_name, part_dim)
        # Year as small MLP
        self.year_mlp = nn.Sequential(
            nn.Linear(1, year_dim),
            nn.ReLU(),
            nn.Linear(year_dim, year_dim),
            nn.ReLU(),
        )
        # Projection to common space
        self.proj = nn.Sequential(
            nn.Linear(year_dim + manu_dim + part_dim, dim),
            nn.ReLU(),
        )
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.emb_manu.weight, std=0.02)
        nn.init.normal_(self.emb_part.weight, std=0.02)

    def forward(self, year: torch.Tensor, manu: torch.Tensor, part: torch.Tensor) -> torch.Tensor:
        y = self.year_mlp(year)          # (B, year_dim)
        m = self.emb_manu(manu)          # (B, manu_dim)
        p = self.emb_part(part)          # (B, part_dim)
        x = torch.cat([y, m, p], dim=-1)
        return self.proj(x)              # (B, dim)


class HybridNCF(nn.Module):
    """Two-tower hybrid model
    - Collaborative tower: user/item id embeddings + MLP
    - Content tower: item metadata -> embedding
    - Fusion: concat(collab_item, content_item) with user, then MLP
    Heads: explicit rating (0..1) & implicit click/purchase logit
    """
    def __init__(self, num_users: int, num_items: int, spaces: ItemFeatureSpaces, dim: int = 64):
        super().__init__()
        # Collaborative embeddings
        self.user_emb = nn.Embedding(num_users, dim)
        self.item_emb = nn.Embedding(num_items, dim)

        # Content tower for items
        self.content = ItemContentTower(spaces, dim=dim)

        # Fusion MLP (user + item_collab + item_content)
        self.mlp = nn.Sequential(
            nn.Linear(dim*3, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.head_explicit = nn.Linear(64, 1)
        self.head_implicit = nn.Linear(64, 1)

        # Gating to handle cold-start (learn how much to trust content vs id)
        self.gate = nn.Sequential(
            nn.Linear(dim*2, 1),
            nn.Sigmoid(),
        )

        # init
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)

    def forward(self, users, items, item_year, item_manu, item_part):
        # Collaborative embeddings
        u = self.user_emb(users)                # (B, dim)
        i_collab = self.item_emb(items)        # (B, dim)
        # Content embedding for item
        i_cont = self.content(item_year, item_manu, item_part)   # (B, dim)

        # Learnable gate (0..1) to mix collab & content item vectors
        g = self.gate(torch.cat([i_collab, i_cont], dim=-1))     # (B,1)
        i = g * i_collab + (1 - g) * i_cont

        x = torch.cat([u, i_collab, i_cont], dim=-1)  # keep both for richer signal
        h = self.mlp(x)
        out_exp = self.head_explicit(h)      # sigmoid later
        out_impl_logit = self.head_implicit(h)
        return out_exp, out_impl_logit

    @torch.no_grad()
    def score_user_all_items(self, user_idx: int, item_feats: Dict[str, torch.Tensor], device: str) -> torch.Tensor:
        """Returns implicit probabilities for all items for a given user.
        Works even for cold-start items because content tower doesn't depend on item id.
        """
        self.eval()
        n_items = item_feats["year"].shape[0]
        users = torch.full((n_items,), user_idx, dtype=torch.long, device=device)
        items = torch.arange(n_items, dtype=torch.long, device=device)
        out_exp, out_impl_logit = self.forward(
            users,
            items,
            item_feats["year"],
            item_feats["manufacturer"],
            item_feats["partname"],
        )
        return torch.sigmoid(out_impl_logit).squeeze(1)  # (n_items,)


# ===============================================================
# Trainer / Recommender
# ===============================================================

class RecommenderSystemHybrid:
    def __init__(self, num_users: int, num_items: int, item_encoder: ItemFeatureEncoder, dim: int = 64, lr: float = 1e-3, device: Optional[str]=None):
        self.num_users = num_users
        self.num_items = num_items
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = "cpu"  # force CPU if needed

        self.item_encoder = item_encoder
        self.item_feats = {k: v.to(self.device) for k, v in self.item_encoder.get_all_tensors(self.device).items()}

        self.model = HybridNCF(num_users, num_items, item_encoder.spaces, dim=dim).to(self.device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.mse = nn.MSELoss(reduction="none")
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def _batch_item_content(self, items: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Gather item content rows for batch items
        year = self.item_feats["year"][items]
        manu = self.item_feats["manufacturer"][items]
        part = self.item_feats["partname"][items]
        return year, manu, part

    def fit(self, dataset: CollabDataset, epochs: int = 5, batch_size: int = 1024):
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        self.model.train()
        for ep in range(epochs):
            total_loss = 0.0
            for batch in loader:
                users = batch["user"].to(self.device)
                items = batch["item"].to(self.device)
                y_exp = batch["y_explicit"].to(self.device)
                y_impl = batch["y_implicit"].to(self.device)
                m_exp = batch["has_explicit"].to(self.device)
                m_impl = batch["has_implicit"].to(self.device)

                year, manu, part = self._batch_item_content(items)

                out_exp, out_impl_logit = self.model(users, items, year, manu, part)
                out_exp = torch.sigmoid(out_exp).squeeze(1)
                out_impl_logit = out_impl_logit.squeeze(1)

                loss_exp = (self.mse(out_exp, y_exp) * m_exp).sum() / (m_exp.sum() + 1e-8)
                loss_impl = (self.bce(out_impl_logit, y_impl) * m_impl).sum() / (m_impl.sum() + 1e-8)

                loss = loss_exp + loss_impl

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                total_loss += float(loss.detach().cpu())

            print(f"[Epoch {ep+1}/{epochs}] loss={total_loss / max(1, len(loader)):.4f}")

    @torch.no_grad()
    def recommend(self, user_idx: int, top_n: int = 5, seen_items: Optional[set] = None) -> List[int]:
        self.model.eval()
        seen_items = seen_items or set()

        scores = self.model.score_user_all_items(user_idx, self.item_feats, self.device).cpu().numpy()
        if len(seen_items) > 0:
            scores[list(seen_items)] = -np.inf
        if top_n <= 0:
            return []
        k = min(top_n, max(1, len(scores)-1))
        top_indices = np.argpartition(-scores, k-1)[:k]
        top_indices = top_indices[np.argsort(-scores[top_indices])]
        return top_indices.tolist()

    def save(self, path: str):
        torch.save({
            "state_dict": self.model.state_dict(),
            "num_users": self.num_users,
            "num_items": self.num_items,
            "spaces": {
                "n_manufacturer": self.item_encoder.spaces.n_manufacturer,
                "n_part_name": self.item_encoder.spaces.n_part_name,
            },
            "item_encoder": {
                "manufacturer2idx": self.item_encoder.manufacturer2idx,
                "partname2idx": self.item_encoder.partname2idx,
                "year_min": self.item_encoder.year_min,
                "year_max": self.item_encoder.year_max,
            },
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.num_users = ckpt["num_users"]
        self.num_items = ckpt["num_items"]
        # Reconstruct minimal spaces (the full encoder should be rebuilt from original items_df in practice)
        spaces = ItemFeatureSpaces(
            n_manufacturer=ckpt["spaces"]["n_manufacturer"],
            n_part_name=ckpt["spaces"]["n_part_name"],
        )
        self.model = HybridNCF(self.num_users, self.num_items, spaces).to(self.device)
        self.model.load_state_dict(ckpt["state_dict"]) 
        self.model.eval()
        # NOTE: caller must also restore/assign self.item_encoder & self.item_feats


# ===============================================================
# Example wiring (usage)
# ===============================================================

if __name__ == "__main__":
    # --- Example input schemas ---
    # ratings_df: columns = [user_id, item_id, rating]
    # orders_df:  columns = [user_id, item_id, order_id, ts]
    # items_df:   columns = [item_id, year_of_make, manufacturer, part_name]

    # TODO: Load your real dataframes here
    ratings_df = pd.DataFrame([
        {"user_id": 1, "item_id": 101, "rating": 4.0},
        {"user_id": 1, "item_id": 102, "rating": 5.0},
        {"user_id": 2, "item_id": 101, "rating": 3.0},
    ])
    orders_df = pd.DataFrame([
        {"user_id": 1, "item_id": 103},
        {"user_id": 2, "item_id": 102},
        {"user_id": 2, "item_id": 103},
    ])
    items_df = pd.DataFrame([
        {"item_id": 101, "year_of_make": 2018, "manufacturer": "Toyota", "part_name": "Oil Filter"},
        {"item_id": 102, "year_of_make": 2020, "manufacturer": "Ford", "part_name": "Engine Oil"},
        {"item_id": 103, "year_of_make": 2019, "manufacturer": "Toyota", "part_name": "Air Filter"},
        {"item_id": 104, "year_of_make": 2021, "manufacturer": "Nissan", "part_name": "Brake Pads"},
    ])

    # Build maps & interactions
    idmaps = build_id_maps(ratings_df, orders_df)
    inter_df = prepare_interactions(ratings_df, orders_df, idmaps)
    seen_sets = build_seen_sets(inter_df)

    # Build item feature encoder
    item_encoder = ItemFeatureEncoder(items_df, idmaps)

    # Dataset & model
    dataset = CollabDataset(inter_df, num_items=len(idmaps.item2idx), neg_ratio=1.0)
    rec = RecommenderSystemHybrid(
        num_users=len(idmaps.user2idx),
        num_items=len(idmaps.item2idx),
        item_encoder=item_encoder,
        dim=64,
        lr=1e-3,
        device="cpu",
    )

    # Train
    rec.fit(dataset, epochs=5, batch_size=8)

    # Recommend for user 1
    uidx = idmaps.user2idx[1]
    recs = rec.recommend(uidx, top_n=3, seen_items=seen_sets.get(uidx, set()))
    print("Top recs (item_idx):", recs)
    print("Top recs (raw item_id):", [idmaps.idx2item[i] for i in recs])
