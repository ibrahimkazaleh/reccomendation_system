# recommender.py
import math
import json
from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# -----------------------------
# Data prep
# -----------------------------

def build_id_maps(ratings_df: pd.DataFrame, orders_df: pd.DataFrame):
    user_ids = pd.Index(pd.concat([ratings_df["user_id"], orders_df["user_id"]]).unique())
    item_ids = pd.Index(pd.concat([ratings_df["item_id"], orders_df["item_id"]]).unique())
    user2idx = {u: i for i, u in enumerate(user_ids)}
    item2idx = {p: i for i, p in enumerate(item_ids)}
    idx2user = {i: u for u, i in user2idx.items()}
    idx2item = {i: p for p, i in item2idx.items()}
    return user2idx, item2idx, idx2user, idx2item

def normalize_rating(r, min_r=0.0, max_r=5.0):
    # map [0..5] -> [0..1]
    return (r - min_r) / (max_r - min_r + 1e-8)

def prepare_interactions(
    ratings_df: pd.DataFrame, 
    orders_df: pd.DataFrame,
    user2idx: Dict, item2idx: Dict
) -> pd.DataFrame:
    """
    Output schema:
      user_idx, item_idx, y_explicit (0..1 or NaN), y_implicit (0/1), seen (bool)
    """
    rows = []
    # Explicit ratings
    if not ratings_df.empty:
        for _, row in ratings_df.iterrows():
            if row["user_id"] in user2idx and row["item_id"] in item2idx:
                rows.append({
                    "user_idx": user2idx[row["user_id"]],
                    "item_idx": item2idx[row["item_id"]],
                    "y_explicit": normalize_rating(float(row["rating"])),
                    "y_implicit": np.nan,
                    "seen": True
                })
    # Implicit from orders (deduplicate to (user,item), option to use count as confidence later)
    if not orders_df.empty:
        agg = orders_df.groupby(["user_id","item_id"]).size().reset_index(name="count")
        for _, row in agg.iterrows():
            if row["user_id"] in user2idx and row["item_id"] in item2idx:
                rows.append({
                    "user_idx": user2idx[row["user_id"]],
                    "item_idx": item2idx[row["item_id"]],
                    "y_explicit": np.nan,
                    "y_implicit": 1.0,   # purchased -> positive implicit
                    "seen": True
                })
    df = pd.DataFrame(rows)
    # collapse duplicates, prefer explicit if both exist
    if not df.empty:
        df = (df
              .groupby(["user_idx","item_idx"], as_index=False)
              .agg({
                  "y_explicit":"max",  # if NaN + val => val ; if two ratings -> max
                  "y_implicit":"max",
                  "seen":"max"
              }))
    return df

def build_seen_sets(inter_df: pd.DataFrame) -> Dict[int, set]:
    seen = {}
    for _, r in inter_df.iterrows():
        seen.setdefault(int(r["user_idx"]), set()).add(int(r["item_idx"]))
    return seen

# -----------------------------
# Dataset
# -----------------------------

class CollabDataset(Dataset):
    def __init__(self, inter_df: pd.DataFrame, num_items: int, neg_ratio: float = 1.0):
        """
        Create mixed explicit/implicit dataset.
        For implicit, we also generate negative samples (y_implicit=0).
        """
        self.num_items = num_items
        self.pos = inter_df.copy()
        self.pos["has_explicit"] = (~self.pos["y_explicit"].isna()).astype(np.float32)
        self.pos["has_implicit"] = (~self.pos["y_implicit"].isna()).astype(np.float32)

        # Build negatives for implicit
        self.seen = build_seen_sets(self.pos)
        neg_rows = []
        total_pos_impl = int(self.pos["has_implicit"].sum())
        num_negs = int(neg_ratio * total_pos_impl)

        rng = np.random.default_rng(42)
        users = list(self.seen.keys())
        for _ in range(num_negs):
            u = int(rng.choice(users))
            # sample an item user hasn't seen
            while True:
                i = int(rng.integers(0, self.num_items))
                if i not in self.seen[u]:
                    break
            neg_rows.append({
                "user_idx": u, "item_idx": i,
                "y_explicit": np.nan, "y_implicit": 0.0,
                "has_explicit": 0.0, "has_implicit": 1.0, "seen": False
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

# -----------------------------
# Model
# -----------------------------

class NCF(nn.Module):
    def __init__(self, num_users: int, num_items: int, dim: int = 64):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, dim)
        self.item_emb = nn.Embedding(num_items, dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim*2, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        # heads: one for explicit rating (0..1), one for implicit logit
        self.head_explicit = nn.Linear(64, 1)
        self.head_implicit = nn.Linear(64, 1)

        # init
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)

    def forward(self, users, items):
        u = self.user_emb(users)
        i = self.item_emb(items)
        x = torch.cat([u, i], dim=-1)
        h = self.mlp(x)
        out_exp = self.head_explicit(h)      # sigmoid later for rating [0..1]
        out_impl_logit = self.head_implicit(h)  # BCEWithLogits
        return out_exp, out_impl_logit
   
    def predict_for_user(self, user_idx):
        """إرجاع توقع لكل عنصر لمستخدم معين"""
        u = self.user_emb(user_idx)
        all_items = self.item_emb.weight
        dot = torch.matmul(u, all_items.T) + self.item_bias.weight.T
        dot = dot + self.user_bias(user_idx)
        return dot.squeeze(0)

# -----------------------------
# Trainer / Recommender
# -----------------------------

class RecommenderSystem:
    def __init__(self, num_users: int, num_items: int, dim: int = 64, lr: float = 1e-3, device: Optional[str]=None):
        self.num_users = num_users
        self.num_items = num_items
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device='cpu'
        self.model = NCF(num_users, num_items, dim).to(self.device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.mse = nn.MSELoss(reduction="none")
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

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

                out_exp, out_impl_logit = self.model(users, items)
                out_exp = torch.sigmoid(out_exp).squeeze(1)  # map to [0..1]
                out_impl_logit = out_impl_logit.squeeze(1)

                # masked losses
                loss_exp = (self.mse(out_exp, y_exp) * m_exp).sum() / (m_exp.sum() + 1e-8)
                loss_impl = (self.bce(out_impl_logit, y_impl) * m_impl).sum() / (m_impl.sum() + 1e-8)

                # weighted sum: tune lambdas if needed
                loss = loss_exp + loss_impl

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                total_loss += loss.item()
            print(f"[Epoch {ep+1}/{epochs}] loss={total_loss/ max(1,len(loader)):.4f}")

    @torch.no_grad()

    def recommend(self, user_idx: int, top_n: int = 5, seen_items: Optional[set] = None) -> List[int]:
        self.model.eval()
        seen_items = seen_items or set()

        with torch.no_grad():
            # Repeat user_idx for all items
            user_tensor = torch.full((self.num_items,), user_idx, dtype=torch.long, device=self.device)
            item_tensor = torch.arange(self.num_items, dtype=torch.long, device=self.device)

            # Forward pass: get implicit logits
            _, logits = self.model(user_tensor, item_tensor)
            scores = torch.sigmoid(logits).cpu().numpy()  # convert to preference probabilities

        # Exclude seen items
        if seen_items:
            scores[list(seen_items)] = -1e9

        # Sort and pick top_n
        valid_items = np.where(scores > -1e8)[0]
        if len(valid_items) == 0:
            return []

        k = min(top_n, len(valid_items))
        top_idx = valid_items[np.argsort(-scores[valid_items])[:k]]

        return top_idx.tolist()


    # def recommend(self, user_idx: int, top_n: int = 10, seen_items: Optional[set]=None) -> List[int]:
    #     self.model.eval()
    #     seen_items = seen_items or set()
    #     # score all items for this user via implicit head
    #     user_vec = torch.full((self.num_items,), user_idx, dtype=torch.long, device=self.device)
    #     items = torch.arange(self.num_items, dtype=torch.long, device=self.device)
    #     _, logits = self.model(user_vec, items)
    #     scores = torch.sigmoid(logits).cpu().numpy()  # preference score
    #     # exclude seen
    #     scores[list(seen_items)] = -1e9
    #     top_idx = np.argpartition(-scores, kth=min(top_n, len(scores)-1))[:top_n]
    #     top_idx = top_idx[np.argsort(-scores[top_idx])]
    #     return top_idx.tolist()



       



    def save(self, path: str):
        torch.save({
            "state_dict": self.model.state_dict(),
            "num_users": self.num_users,
            "num_items": self.num_items
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.num_users = ckpt["num_users"]
        self.num_items = ckpt["num_items"]
        self.model = NCF(self.num_users, self.num_items).to(self.device)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()

# -----------------------------
# Example usage (replace with your real data)
# -----------------------------
if __name__ == "__main__":
    # مثال بيانات
    ratings = pd.DataFrame([
        {"user_id": 'a', "item_id": 100, "rating": 4.0},
        {"user_id": 'a', "item_id": 101, "rating": 5.0},
        {"user_id": 'b', "item_id": 100, "rating": 3.0},
        {"user_id": 'c', "item_id": 103, "rating": 4.5},
        {"user_id": 'd', "item_id": 100, "rating": 4.0},
        {"user_id": 'e', "item_id": 101, "rating": 5.0},

    ])
    orders = pd.DataFrame([
        {"user_id": 'a', "item_id": 101},
        {"user_id": 'a', "item_id": 101},
        {"user_id": 'b', "item_id": 104},
        {"user_id": 'c', "item_id": 101},
        {"user_id": 'c', "item_id": 105},
        {"user_id": 'd', "item_id": 102},
        {"user_id": 'b', "item_id": 101},
        {"user_id": 'b', "item_id": 104},
        {"user_id": 'c', "item_id": 105},
        {"user_id": 'c', "item_id": 105},
        
    ])

    user2idx, item2idx, idx2user, idx2item = build_id_maps(ratings, orders)
    inter = prepare_interactions(ratings, orders, user2idx, item2idx)

    num_users = len(user2idx)
    num_items = len(item2idx)
    ds = CollabDataset(inter, num_items=num_items, neg_ratio=1.0)
    ds.all
    ds.pos["has_implicit"]

    rec = RecommenderSystem(num_users=num_users, num_items=num_items, dim=64, lr=1e-3)
    rec.device
    rec.fit(ds, epochs=45, batch_size=256)

    seen_sets = build_seen_sets(inter)
    
    u = list(user2idx.values())[0]  # أول مستخدم كمثال
    recs_idx = rec.recommend(2, top_n=2, seen_items=seen_sets.get(u, set()))
    recs_idx = rec.recommend(0, top_n=2)

    print("Top-N item indices:", recs_idx)
    print("Mapped back to item_ids:", [idx2item[i] for i in recs_idx])

    # حفظ النموذج
    rec.save("recommender_ckpt.pt")


