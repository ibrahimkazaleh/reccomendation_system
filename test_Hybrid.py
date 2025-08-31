# test_hybrid_fixed.py
import numpy as np
import pandas as pd
import torch
import random

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

def random_id(n=24):
    return ''.join(random.choices('0123456789abcdef', k=n))

users_df = pd.DataFrame({
    "user_id": [random_id() for _ in range(n_users)],
    "name": [f"user_{i}" for i in range(n_users)]
})

items_df = pd.DataFrame({
    "item_id": [random_id() for _ in range(n_items)],
    "name": np.random.choice(["فلتر", "فرامل", "باب", "ضوء", "عجلة"], size=n_items),
    "manufacturer": np.random.choice(["Toyota", "Ford", "BMW", "Honda"], size=n_items),
    "year": np.random.randint(2000, 2024, size=n_items)
})
ratings_df = pd.DataFrame({
    "user_id": np.random.choice(users_df["user_id"], size=rating_size),
    "item_id": np.random.choice(items_df["item_id"], size=rating_size),
    "rating": np.random.randint(1, 6, size=rating_size)
})
orders_df = pd.DataFrame({
    "user_id": np.random.choice(users_df["user_id"], size=order_size),
    "item_id": np.random.choice(items_df["item_id"], size=order_size)
})

# print("Users:\n", users_df.head(), "\n")
# print("Items:\n", items_df.head(), "\n")
# print("Ratings:\n", ratings_df.head(), "\n")
# print("Orders:\n", orders_df.head())


ratings_df = pd.DataFrame({
    "user_id": np.random.choice(users_df["user_id"], size=rating_size),
    "item_id": np.random.choice(items_df["item_id"], size=rating_size),
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
inter_df[inter_df['user_idx'] == 0 ]
# -----------------------------
# dataset & encoder
# -----------------------------
dataset = CollabDataset(inter_df, num_items=len(idmaps.item2idx), neg_ratio=1.0)
item_encoder = ItemFeatureEncoder(items_df, idmaps)
dataset.all[dataset.all['user_idx'] == 0 ]
dataset.all
dataset.__getitem__(0)
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
rec.fit(dataset, epochs=20, batch_size=4)

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

rec.save('model/Hybrid_model_v1.pt')



import pickle
import os
v = 2
os.makedirs("model", exist_ok=True)

# 1) حفظ النموذج
rec.save(f"model/Hybrid_model_v{v}.pt")

# 2) حفظ idmaps
with open(f"model/idmaps_v{v}.pkl", "wb") as f:
    pickle.dump(idmaps, f)

# 3) حفظ item_encoder
with open(f"model/item_encoder_v{v}.pkl", "wb") as f:
    pickle.dump(item_encoder, f)

# 4) (اختياري) حفظ seen_sets
with open(f"model/seen_sets_v{v}.pkl", "wb") as f:
    pickle.dump(seen_sets, f)

idmaps.user2idx.keys()

# item
# {

#             "user_id": "687ff530bf0de81878ed94ef",
#             "name": "فلتر",
#             "manufacturer": "Toyota",
#             "year": 2023,

#         },


# user {

#             "user_id": "34jfsdfjwef78ed94ef",
#             "name": "user",
#         }