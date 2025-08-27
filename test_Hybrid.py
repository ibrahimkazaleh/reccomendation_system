# test_hybrid_fixed.py
import numpy as np
import pandas as pd
import torch

# استورد الواجهات من ملف Hybrid.py الذي أعددناه سابقًا
from Hybrid import (
    build_id_maps,
    prepare_interactions,
    build_seen_sets,
    CollabDataset,
    ItemFeatureEncoder,
    RecommenderSystemHybrid,
)

# -----------------------------
# توليد بيانات عشوائية
# -----------------------------
np.random.seed(42)

n_users = 20
n_items = 50
rating_size =300
order_size = 200
ratings_df = pd.DataFrame({
    "user_id": np.random.choice(range(1, n_users + 1), size=rating_size),
    "item_id": np.random.choice(range(1, n_items + 1), size=rating_size),
    "rating": np.random.randint(1, 6, size=rating_size)
})

orders_df = pd.DataFrame({
    "user_id": np.random.choice(range(1, n_users + 1), size=order_size),
    "item_id": np.random.choice(range(1, n_items + 1), size=order_size),
})

items_df = pd.DataFrame({
    "item_id": list(range(1, n_items + 1)),
    "year_of_make": np.random.randint(2000, 2023, size=n_items),
    "manufacturer": np.random.choice(["Toyota", "Ford", "BMW", "Honda"], size=n_items),
    "part_name": np.random.choice(["Engine", "Brake", "Door", "Light", "Wheel"], size=n_items)
})

# -----------------------------
# بناء id maps (يعيد IdMaps dataclass)
# -----------------------------
idmaps = build_id_maps(ratings_df, orders_df)
# وصول للمخططات عبر idmaps.user2idx, idmaps.item2idx, idmaps.idx2user, idmaps.idx2item

# -----------------------------
# تجهيز التفاعلات (تتوقع idmaps كوسيط)
# -----------------------------
inter_df = prepare_interactions(ratings_df, orders_df, idmaps)
seen_sets = build_seen_sets(inter_df)

# -----------------------------
# dataset & encoder
# -----------------------------
dataset = CollabDataset(inter_df, num_items=len(idmaps.item2idx), neg_ratio=1.0)
item_encoder = ItemFeatureEncoder(items_df, idmaps)

# -----------------------------
# بناء و تدريب النظام الهجين
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
rec.fit(dataset, epochs=50, batch_size=4)

# -----------------------------
# تجربة توصية لمستخدم حقيقي (raw user id -> تحويل الى idx)
# -----------------------------
# اختَر مستخدما عشوائياً من القاموس
raw_user = next(iter(idmaps.user2idx.keys()))
uidx = idmaps.user2idx[raw_user]

for uidx in range(0,20):
    recs_idx = rec.recommend(uidx, top_n=5, seen_items=seen_sets.get(uidx, set()))
    # print("Recommended item_idx:", recs_idx)
    # print("Recommended raw item_id:", [idmaps.idx2item[i] for i in recs_idx])
    print( recs_idx)
