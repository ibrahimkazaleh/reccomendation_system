# evaluate_hybrid.py
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from Hybrid import (
    build_id_maps,
    prepare_interactions,
    build_seen_sets,
    CollabDataset,
    ItemFeatureEncoder,
    RecommenderSystemHybrid,
)

# -----------------------------
# 1. توليد بيانات عشوائية للتجربة
# -----------------------------
np.random.seed(42)
n_users, n_items = 50, 80

ratings_df = pd.DataFrame({
    "user_id": np.random.choice(range(1, n_users + 1), size=300),
    "item_id": np.random.choice(range(1, n_items + 1), size=300),
    "rating": np.random.randint(1, 6, size=300)
})

orders_df = pd.DataFrame({
    "user_id": np.random.choice(range(1, n_users + 1), size=100),
    "item_id": np.random.choice(range(1, n_items + 1), size=100),
})

items_df = pd.DataFrame({
    "item_id": list(range(1, n_items + 1)),
    "year_of_make": np.random.randint(2000, 2023, size=n_items),
    "manufacturer": np.random.choice(["Toyota", "Ford", "BMW", "Honda"], size=n_items),
    "part_name": np.random.choice(["Engine", "Brake", "Door", "Light", "Wheel"], size=n_items)
})

# -----------------------------
# 2. بناء الخرائط و التفاعلات
# -----------------------------
idmaps = build_id_maps(ratings_df, orders_df)
inter_df = prepare_interactions(ratings_df, orders_df, idmaps)

# تقسيم train/test (80/20)
train_df, test_df = train_test_split(inter_df, test_size=0.2, random_state=42)

train_dataset = CollabDataset(train_df, num_items=len(idmaps.item2idx), neg_ratio=1.0)
item_encoder = ItemFeatureEncoder(items_df, idmaps)

# -----------------------------
# 3. تدريب النموذج
# -----------------------------
rec = RecommenderSystemHybrid(
    num_users=len(idmaps.user2idx),
    num_items=len(idmaps.item2idx),
    item_encoder=item_encoder,
    dim=64,
    lr=1e-3,
    device="cuda",
)
rec.device  
rec.fit(train_dataset, epochs=20, batch_size=3)

# -----------------------------
# 4. دوال التقييم
# -----------------------------
def hit_ratio_at_k(recommended, ground_truth, k=10):
    return 1.0 if ground_truth in recommended[:k] else 0.0

def ndcg_at_k(recommended, ground_truth, k=10):
    if ground_truth in recommended[:k]:
        idx = recommended.index(ground_truth)
        return 1.0 / np.log2(idx + 2)
    return 0.0

def precision_at_k(recommended, ground_truths, k=10):
    rec_k = recommended[:k]
    hits = len(set(rec_k) & set(ground_truths))
    return hits / k

def recall_at_k(recommended, ground_truths, k=10):
    rec_k = recommended[:k]
    hits = len(set(rec_k) & set(ground_truths))
    return hits / len(ground_truths) if ground_truths else 0

def average_precision(recommended, ground_truths, k=10):
    """AP@K: Average Precision"""
    score, hits = 0.0, 0
    for i, rec in enumerate(recommended[:k]):
        if rec in ground_truths:
            hits += 1
            score += hits / (i + 1.0)
    return score / min(len(ground_truths), k) if ground_truths else 0

# -----------------------------
# 5. التقييم على test
# -----------------------------
K = 10
hr_scores, ndcg_scores, prec_scores, rec_scores, map_scores = [], [], [], [], []

seen_sets_train = build_seen_sets(train_df)

for _, row in test_df.iterrows():
    user = row["user_idx"]
    true_item = row["item_idx"]

    recs = rec.recommend(user, top_n=K, seen_items=seen_sets_train.get(user, set()))

    hr_scores.append(hit_ratio_at_k(recs, true_item, k=K))
    ndcg_scores.append(ndcg_at_k(recs, true_item, k=K))
    prec_scores.append(precision_at_k(recs, [true_item], k=K))
    rec_scores.append(recall_at_k(recs, [true_item], k=K))
    map_scores.append(average_precision(recs, [true_item], k=K))

print(f"HR@{K}:   {np.mean(hr_scores):.4f}")
print(f"NDCG@{K}: {np.mean(ndcg_scores):.4f}")
print(f"Precision@{K}: {np.mean(prec_scores):.4f}")
print(f"Recall@{K}:    {np.mean(rec_scores):.4f}")
print(f"MAP@{K}:       {np.mean(map_scores):.4f}")
