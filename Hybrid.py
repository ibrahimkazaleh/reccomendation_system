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

class IdMaps:
    def __init__(self, user2idx=None, item2idx=None, idx2user=None, idx2item=None):
        self.user2idx = user2idx or {}
        self.item2idx = item2idx or {}
        self.idx2user = idx2user or {}
        self.idx2item = idx2item or {}

    def fit(self, user_ids, item_ids):
        """item and user to id"""
        self.user2idx = {u: i for i, u in enumerate(user_ids)}
        self.item2idx = {it: j for j, it in enumerate(item_ids)}
        self.idx2user = {i: u for u, i in self.user2idx.items()}
        self.idx2item = {j: it for it, j in self.item2idx.items()}

    def num_users(self):
        return len(self.user2idx)

    def num_items(self):
        return len(self.item2idx)


    @staticmethod
    def from_dfs(ratings_df: pd.DataFrame, orders_df: pd.DataFrame) -> "IdMaps":
        user_ids = pd.Index(pd.concat([ratings_df["user_id"], orders_df["user_id"]]).unique())
        item_ids = pd.Index(pd.concat([ratings_df["item_id"], orders_df["item_id"]]).unique())
        user2idx = {u: i for i, u in enumerate(user_ids)}
        item2idx = {p: i for i, p in enumerate(item_ids)}
        idx2user = {i: u for u, i in user2idx.items()}
        idx2item = {i: p for p, i in item2idx.items()}
        return IdMaps(user2idx, item2idx, idx2user, idx2item)

    def add_user(self, raw_user_id: Any) -> int:
        """Add a new user id if not existing. Return user_idx."""
        if raw_user_id in self.user2idx:
            return self.user2idx[raw_user_id]
        new_idx = max(self.idx2user.keys()) + 1 if self.idx2user else 0
        self.user2idx[raw_user_id] = new_idx
        self.idx2user[new_idx] = raw_user_id
        return new_idx

    def add_item(self, raw_item_id: Any) -> int:
        """Add a new item id if not existing. Return item_idx."""
        if raw_item_id in self.item2idx:
            return self.item2idx[raw_item_id]
        new_idx = max(self.idx2item.keys()) + 1 if self.idx2item else 0
        self.item2idx[raw_item_id] = new_idx
        self.idx2item[new_idx] = raw_item_id
        return new_idx

    def to_dict(self) -> dict:
        return {
            "user2idx": self.user2idx,
            "item2idx": self.item2idx,
            "idx2user": self.idx2user,
            "idx2item": self.idx2item,
        }

    @staticmethod
    def from_dict(d: dict) -> "IdMaps":
        return IdMaps(d["user2idx"], d["item2idx"], {int(k): v for k, v in d["idx2user"].items()}, {int(k): v for k, v in d["idx2item"].items()})


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
    if ratings_df is not None and not ratings_df.empty:
        for _, row in ratings_df.iterrows():
            if row["user_id"] in idmaps.user2idx and row["item_id"] in idmaps.item2idx:
                rows.append({
                    "user_idx": idmaps.user2idx[row["user_id"]],
                    "item_idx": idmaps.item2idx[row["item_id"]],
                    "y_explicit": normalize_rating(float(row["rating"])),
                    "y_implicit": np.nan,
                    "seen": True,
                })
    if orders_df is not None and not orders_df.empty:
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
    if inter_df is None or inter_df.empty:
        return seen
    for _, r in inter_df.iterrows():
        seen.setdefault(int(r["user_idx"]), set()).add(int(r["item_idx"]))
    return seen


# ===============================================================
# Item Metadata Encoding (extendable)
# ===============================================================

@dataclass
class ItemFeatureSpaces:
    n_manufacturer: int
    n_part_name: int


class ItemFeatureEncoder:
    """Encodes item metadata -> dense tensors
    Inputs DataFrame schema (example):
      item_id, year, manufacturer, part_name
    Supports extension with new items and new categorical values.
    """

    def __init__(self, items_df: pd.DataFrame, idmaps: IdMaps):
        # initial vocabularies (preserve order)
        self.manufacturer2idx: Dict[str, int] = {}
        self.partname2idx: Dict[str, int] = {}
        # build initial mappings
        manu_vals = items_df["manufacturer"].fillna("<UNK>").astype(str).unique()
        part_vals = items_df["part_name"].fillna("<UNK>").astype(str).unique()
        for v in manu_vals:
            if v not in self.manufacturer2idx:
                self.manufacturer2idx[v] = len(self.manufacturer2idx)
        for v in part_vals:
            if v not in self.partname2idx:
                self.partname2idx[v] = len(self.partname2idx)

        # year scaling values
        y = items_df["year"].fillna(items_df["year"].median())
        self.year_min = float(np.nanmin(y)) if len(y) else 0.0
        self.year_max = float(np.nanmax(y)) if len(y) else 1.0

        self.idmaps = idmaps
        # build features matrix aligned to idmaps.item2idx
        self.rebuild_features(items_df)

    def _norm_year(self, y):
        if pd.isna(y):
            y = (self.year_min + self.year_max) * 0.5
        if self.year_max == self.year_min:
            return 0.0
        return float((y - self.year_min) / (self.year_max - self.year_min))

    def rebuild_features(self, items_df: pd.DataFrame):
        """(Re)build full features arrays aligned to current idmaps.
        Extends manufacturer/partname vocabs if new categories appear.
        """
        # incorporate any new categories found in items_df
        for _, row in items_df.iterrows():
            manu = str(row.get("manufacturer", "<UNK>"))
            pname = str(row.get("part_name", "<UNK>"))
            if manu not in self.manufacturer2idx:
                self.manufacturer2idx[manu] = len(self.manufacturer2idx)
            if pname not in self.partname2idx:
                self.partname2idx[pname] = len(self.partname2idx)

        # update year scaling
        y_all = items_df["year"].fillna(items_df["year"].median())
        if len(y_all):
            self.year_min = min(self.year_min, float(np.nanmin(y_all)))
            self.year_max = max(self.year_max, float(np.nanmax(y_all)))

        n_items = len(self.idmaps.item2idx)
        # allocate arrays
        year_arr = np.zeros((n_items, 1), dtype=np.float32)
        manu_arr = np.zeros((n_items,), dtype=np.int64)
        part_arr = np.zeros((n_items,), dtype=np.int64)

        by_id = items_df.set_index("item_id")
        for raw_item_id, item_idx in self.idmaps.item2idx.items():
            if raw_item_id in by_id.index:
                row = by_id.loc[raw_item_id]
                year = row.get("year", np.nan)
                manu = str(row.get("manufacturer", "<UNK>"))
                pname = str(row.get("part_name", "<UNK>"))
            else:
                year = np.nan
                manu = "<UNK>"
                pname = "<UNK>"
            year_norm = self._norm_year(year)
            manu_idx = self.manufacturer2idx.get(manu, 0)
            pname_idx = self.partname2idx.get(pname, 0)
            year_arr[item_idx, 0] = year_norm
            manu_arr[item_idx] = manu_idx
            part_arr[item_idx] = pname_idx

        self.features = {
            "year": year_arr,
            "manufacturer": manu_arr,
            "partname": part_arr,
        }
        self.spaces = ItemFeatureSpaces(
            n_manufacturer=len(self.manufacturer2idx),
            n_part_name=len(self.partname2idx),
        )

    def extend_with_items(self, new_items_df: pd.DataFrame, idmaps: IdMaps):
        """Add new items into idmaps externally and then call this to rebuild features."""
        self.idmaps = idmaps
        self.rebuild_features(new_items_df)

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
        self.pos = inter_df.copy() if inter_df is not None else pd.DataFrame()
        if not self.pos.empty:
            self.pos["has_explicit"] = (~self.pos["y_explicit"].isna()).astype(np.float32)
            self.pos["has_implicit"] = (~self.pos["y_implicit"].isna()).astype(np.float32)
        else:
            self.pos["has_explicit"] = pd.Series(dtype=np.float32)
            self.pos["has_implicit"] = pd.Series(dtype=np.float32)

        self.seen = build_seen_sets(self.pos)

        neg_rows = []
        total_pos_impl = int(self.pos["has_implicit"].sum()) if not self.pos.empty else 0
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
                if i not in self.seen.get(u, set()):
                    break
                tries += 1
                if tries > 50:
                    unseen = list(set(range(self.num_items)) - set(self.seen.get(u, set())))
                    if unseen:
                        i = int(rng.choice(unseen))
                    break
            neg_rows.append({
                "user_idx": u, "item_idx": i,
                "y_explicit": np.nan, "y_implicit": 0.0,
                "has_explicit": 0.0, "has_implicit": 1.0, "seen": False,
            })

        self.all = pd.concat([self.pos, pd.DataFrame(neg_rows)], ignore_index=True) if neg_rows else self.pos
        if not self.all.empty:
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
# Model: Hybrid NCF (Collaborative + Content) - enhanced dims
# ===============================================================

class ItemContentTower(nn.Module):
    def __init__(self, spaces: ItemFeatureSpaces, content_dim: int = 64, year_dim: int = 8, manu_dim: int = 32, part_dim: int = 32):
        super().__init__()
        self.content_dim = content_dim
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
        # Projection to common content space
        self.proj = nn.Sequential(
            nn.Linear(year_dim + manu_dim + part_dim, content_dim),
            nn.ReLU(),
        )
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.emb_manu.weight, std=0.02)
        nn.init.normal_(self.emb_part.weight, std=0.02)

    def forward(self, year: torch.Tensor, manu: torch.Tensor, part: torch.Tensor) -> torch.Tensor:
        # year: (B,1), manu: (B,), part: (B,)
        y = self.year_mlp(year)          # (B, year_dim)
        m = self.emb_manu(manu)          # (B, manu_dim)
        p = self.emb_part(part)          # (B, part_dim)
        x = torch.cat([y, m, p], dim=-1)
        return self.proj(x)              # (B, content_dim)


class HybridNCF(nn.Module):
    """Two-tower hybrid model with configurable dims"""
    def __init__(self, num_users: int, num_items: int,
                 spaces: ItemFeatureSpaces,
                 user_dim: int = 64, item_dim: int = 64, content_dim: int = 64):
        super().__init__()
        self.user_dim = user_dim
        self.item_dim = item_dim
        self.content_dim = content_dim

        # Collaborative embeddings
        self.user_emb = nn.Embedding(num_users, user_dim)
        self.item_emb = nn.Embedding(num_items, item_dim)

        # Content tower for items (produces content_dim vector)
        self.content = ItemContentTower(spaces, content_dim=content_dim)

        # Fusion MLP (user + item_collab + item_content)
        fusion_in = user_dim + item_dim + content_dim
        self.mlp = nn.Sequential(
            nn.Linear(fusion_in, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.head_explicit = nn.Linear(64, 1)
        self.head_implicit = nn.Linear(64, 1)

        # Gating to handle cold-start (learn how much to trust content vs id)
        # gate input: item_collab (item_dim) + item_content (content_dim) --> sigmoid scalar
        self.gate = nn.Sequential(
            nn.Linear(item_dim + content_dim, 1),
            nn.Sigmoid(),
        )

        # init
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)

    def forward(self, users, items, item_year, item_manu, item_part):
        # Collaborative embeddings
        u = self.user_emb(users)                # (B, user_dim)
        i_collab = self.item_emb(items)         # (B, item_dim)
        # Content embedding for item
        i_cont = self.content(item_year, item_manu, item_part)   # (B, content_dim)

        # Learnable gate (0..1) to mix collab & content item vectors
        # For gating we need matching dims, so project or pad small dims: simplest approach: 
        # if item_dim != content_dim, project both to min_dim for gating input
        if i_collab.shape[-1] != i_cont.shape[-1]:
            # linear projections (lazy creation to avoid fixed module complexity). For simplicity, concatenate and let linear accept it.
            return []
        g = self.gate(torch.cat([i_collab, i_cont], dim=-1))     # (B,1)
        # Mix
        # If dims differ, need to bring them to same size for mixing; we'll expand smaller with linear projection.
        if i_collab.shape[-1] == i_cont.shape[-1]:
            i = g * i_collab + (1 - g) * i_cont
        else:
            # project both to content_dim for final mixing (safe choice)
            # define on the fly layers? We'll create simple linear layers attached to module for projection
            if not hasattr(self, "_proj_collab_to_cont"):
                self._proj_collab_to_cont = nn.Linear(i_collab.shape[-1], i_cont.shape[-1]).to(i_collab.device)
            proj_collab = self._proj_collab_to_cont(i_collab)
            i = g * proj_collab + (1 - g) * i_cont

        x = torch.cat([u, i_collab, i_cont], dim=-1)  # keep both for richer signal
        h = self.mlp(x)
        out_exp = self.head_explicit(h)      # sigmoid later
        out_impl_logit = self.head_implicit(h)
        return out_exp, out_impl_logit

    @torch.no_grad()
    def score_user_all_items(self, user_idx: int, item_feats: Dict[str, torch.Tensor], device: str) -> torch.Tensor:
        """Returns implicit probabilities for all items for a given user.
        Works also for cold-start items because content tower doesn't depend on historical item embedding.
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
# Trainer / Recommender - with addition APIs
# ===============================================================

class RecommenderSystemHybrid:
    def __init__(self,
                 idmaps: IdMaps,
                 item_encoder: ItemFeatureEncoder,
                 user_dim: int = 64, item_dim: int = 64, content_dim: int = 64,
                 lr: float = 1e-3, device: Optional[str] = None):
        self.idmaps = idmaps
        self.item_encoder = item_encoder
        # self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device=device
        self.user_dim = user_dim
        self.item_dim = item_dim
        self.content_dim = content_dim
        self.lr = lr

        self.num_users = len(self.idmaps.user2idx)
        self.num_items = len(self.idmaps.item2idx)

        # item features as tensors (on device)
        self.item_feats = {k: v.to(self.device) for k, v in self.item_encoder.get_all_tensors(self.device).items()}

        # Build model
        self.model = HybridNCF(self.num_users, self.num_items, self.item_encoder.spaces,
                               user_dim=user_dim, item_dim=item_dim, content_dim=content_dim).to(self.device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.mse = nn.MSELoss(reduction="none")
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def _batch_item_content(self, items: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        year = self.item_feats["year"][items]
        manu = self.item_feats["manufacturer"][items].long()
        part = self.item_feats["partname"][items].long()
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

    # ---------------------------
    # --- add_user / Extensions
    # ---------------------------

    def _reinit_optimizer(self):
        self.opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def add_user(self, raw_user_id: Any) -> int:
        """Add a new user mapping and extend user embedding weights without breaking old ones.
        Returns the new user_idx.
        """
        if raw_user_id in self.idmaps.user2idx:
            return self.idmaps.user2idx[raw_user_id]
        new_idx = self.idmaps.add_user(raw_user_id)
        old_num = self.num_users
        new_num = old_num + 1
        self.num_users = new_num

        # extend embedding
        old_emb = self.model.user_emb
        old_weight = old_emb.weight.data
        new_emb = nn.Embedding(new_num, old_emb.embedding_dim).to(self.device)
        # copy old weights
        new_emb.weight.data[:old_num] = old_weight
        # init new row
        nn.init.normal_(new_emb.weight.data[old_num:], std=0.01)
        self.model.user_emb = new_emb

        # re-init optimizer to capture new params
        self._reinit_optimizer()
        return new_idx

    def add_items(self, new_items_df: pd.DataFrame):
        """Add new items (raw item ids must be present in new_items_df['item_id']).
        This will update idmaps, item_encoder, item_feats and extend item embedding matrix.
        """
        # 1) add new raw ids to idmaps if not present
        added = []
        for raw_id in new_items_df["item_id"].unique():
            if raw_id not in self.idmaps.item2idx:
                idx = self.idmaps.add_item(raw_id)
                added.append((raw_id, idx))

        if not added:
            # nothing to add, but might still have new categories -> rebuild features
            self.item_encoder.rebuild_features(new_items_df)
            self.item_feats = {k: v.to(self.device) for k, v in self.item_encoder.get_all_tensors(self.device).items()}
            return

        # 2) update item encoder (rebuild features aligned to new idmaps)
        # to preserve previous category indices, pass a combined items_df of known + new
        # (in practice, keep a catalog; here we assume new_items_df contains both existing & new rows needed)
        self.item_encoder.rebuild_features(new_items_df)

        # 3) extend item embedding matrix
        old_num = self.num_items
        new_num = len(self.idmaps.item2idx)
        self.num_items = new_num

        old_emb = self.model.item_emb
        old_weight = old_emb.weight.data
        new_emb = nn.Embedding(new_num, old_emb.embedding_dim).to(self.device)
        new_emb.weight.data[:old_num] = old_weight
        # initialize new rows
        if new_num > old_num:
            nn.init.normal_(new_emb.weight.data[old_num:], std=0.01)

        self.model.item_emb = new_emb

        # 4) update item_feats tensors
        self.item_feats = {k: v.to(self.device) for k, v in self.item_encoder.get_all_tensors(self.device).items()}

        # re-init optimizer to capture new params
        self._reinit_optimizer()

    def recommend_cold_start_by_content(self, user_pref: Optional[dict] = None, top_n: int = 5, exclude_seen: Optional[set] = None) -> List[int]:
        """
        Recommend using only content tower based on user preference dict.
        user_pref can be:
         - {"manufacturer": ["Toyota", "Ford"], "part_name": ["Oil Filter"], "year": 2019}
         - or None -> fallback: recommend by content model scores averaged or popularity proxy.
        """
        self.model.eval()
        exclude_seen = exclude_seen or set()
        # compute per-item content embeddings (n_items, content_dim)
        year_t = self.item_feats["year"]
        manu_t = self.item_feats["manufacturer"].long()
        part_t = self.item_feats["partname"].long()
        with torch.no_grad():
            items_content = self.model.content(year_t, manu_t, part_t)  # (n_items, content_dim)

            # build preference vector in content space
            if user_pref:
                pieces = []
                device = next(self.model.parameters()).device

                # manufacturer preference
                manu_pref = user_pref.get("manufacturer")
                if manu_pref:
                    # map manufacturer names to indices known in encoder
                    idxs = [self.item_encoder.manufacturer2idx.get(str(m), None) for m in manu_pref]
                    idxs = [i for i in idxs if i is not None]
                    if idxs:
                        emb_manu = self.model.content.emb_manu(torch.tensor(idxs, device=device))
                        pieces.append(emb_manu.mean(dim=0, keepdim=True))  # (1, manu_dim)

                # part_name preference
                part_pref = user_pref.get("part_name") or user_pref.get("partname")
                if part_pref:
                    idxs = [self.item_encoder.partname2idx.get(str(p), None) for p in part_pref]
                    idxs = [i for i in idxs if i is not None]
                    if idxs:
                        emb_part = self.model.content.emb_part(torch.tensor(idxs, device=device))
                        pieces.append(emb_part.mean(dim=0, keepdim=True))  # (1, part_dim)

                # year preference
                year_pref = user_pref.get("year")
                if year_pref is not None:
                    year_val = torch.tensor([[self.item_encoder._norm_year(year_pref)]], device=device)
                    year_vec = self.model.content.year_mlp(year_val)  # (1, year_dim)
                    pieces.append(year_vec)

                if pieces:
                    # concat pieces in the same order the tower does: year, manu, part
                    # but some pieces may be missing; we fill missing with zeros of correct size
                    # determine dims:
                    year_dim = self.model.content.year_mlp[0].out_features if hasattr(self.model.content.year_mlp[0], 'out_features') else self.model.content.year_mlp[0].out_features
                    manu_dim = self.model.content.emb_manu.embedding_dim
                    part_dim = self.model.content.emb_part.embedding_dim
                    # build full vector of shape (1, year_dim + manu_dim + part_dim)
                    ys = torch.zeros((1, year_dim), device=device)
                    ms = torch.zeros((1, manu_dim), device=device)
                    ps = torch.zeros((1, part_dim), device=device)
                    # fill from pieces by type detection (rough but sufficient)
                    for p in pieces:
                        if p.shape[-1] == manu_dim:
                            ms = p
                        elif p.shape[-1] == part_dim:
                            ps = p
                        else:
                            # assume year_dim
                            ys = p
                    pref_cat = torch.cat([ys, ms, ps], dim=-1)  # (1, sum)
                    pref_vec = self.model.content.proj(pref_cat)  # (1, content_dim)
                    pref_vec = pref_vec.squeeze(0)  # (content_dim,)
                else:
                    # no specific pref categories -> fallback: average content vector
                    pref_vec = items_content.mean(dim=0)
            else:
                # no user_pref -> fallback average item content
                pref_vec = items_content.mean(dim=0)

            # score items by similarity with pref_vec (cosine or dot)
            # use cosine similarity:
            pref_norm = pref_vec / (pref_vec.norm() + 1e-8)
            items_norm = items_content / (items_content.norm(dim=1, keepdim=True) + 1e-8)
            sims = torch.matmul(items_norm, pref_norm.unsqueeze(-1)).squeeze(-1).cpu().numpy()  # (n_items,)

            # exclude seen
            if exclude_seen:
                sims[list(exclude_seen)] = -np.inf
            # top N
            if top_n <= 0:
                return []
            k = min(top_n, max(1, len(sims)-1))
            top_indices = np.argpartition(-sims, k-1)[:k]
            top_indices = top_indices[np.argsort(-sims[top_indices])]
            return top_indices.tolist()

    # Save/load (also saves idmaps + encoder metadata)
    def save(self, path: str):
        torch.save({
            "version":1,
            "state_dict": self.model.state_dict(),
            "num_users": self.num_users,
            "num_items": self.num_items,
            "user_dim": self.user_dim,
            "item_dim": self.item_dim,
            "content_dim": self.content_dim,
            "idmaps": self.idmaps.to_dict(),
            "item_encoder": {
                "manufacturer2idx": self.item_encoder.manufacturer2idx,
                "partname2idx": self.item_encoder.partname2idx,
                "year_min": self.item_encoder.year_min,
                "year_max": self.item_encoder.year_max,
                # NOTE: we do not save full features arrays here (they can be rebuilt from catalog)
            },
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.num_users = ckpt["num_users"]
        self.num_items = ckpt["num_items"]
        self.user_dim = ckpt.get("user_dim", self.user_dim)
        self.item_dim = ckpt.get("item_dim", self.item_dim)
        self.content_dim = ckpt.get("content_dim", self.content_dim)
        self.idmaps = IdMaps.from_dict(ckpt["idmaps"])
        spaces = ItemFeatureSpaces(
            n_manufacturer=ckpt["item_encoder"]["manufacturer2idx"].__len__(),
            n_part_name=ckpt["item_encoder"]["partname2idx"].__len__(),
        )
        # rebuild model and load weights
        self.model = HybridNCF(self.num_users, self.num_items, spaces,
                               user_dim=self.user_dim, item_dim=self.item_dim, content_dim=self.content_dim).to(self.device)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        # NOTE: caller must restore item_encoder and item_feats from the original catalog to keep indices consistent.
