# test_hybrid_fixed.py
import numpy as np
import pandas as pd
import torch
import random
from Hybrid import (
    IdMaps, prepare_interactions, ItemFeatureEncoder,
    CollabDataset, HybridNCF, RecommenderSystemHybrid,build_seen_sets
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
    "part_name": np.random.choice(["فلتر", "فرامل", "باب", "ضوء", "عجلة"], size=n_items),
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


# -----------------------------
# idmaps 
# -----------------------------
id_maps = IdMaps()
id_maps.fit(
    user_ids=list(set(ratings_df["user_id"]).union(orders_df["user_id"])),
    item_ids=list(items_df["item_id"])
)
number_of_item = len(id_maps.idx2item)
number_of_user = len(id_maps.idx2user)


# -----------------------------
# dataset & encoder
# -----------------------------
interactions = prepare_interactions(ratings_df, orders_df, id_maps)
item_encoder = ItemFeatureEncoder(items_df, id_maps)
len(id_maps.idx2item)
dataset = CollabDataset(interactions, number_of_user,neg_ratio=1)
seen_sets = build_seen_sets(interactions)
# -----------------------------
# بناء و تدريب النظام الهجين
# -----------------------------
system = RecommenderSystemHybrid(id_maps, item_encoder,device='cpu')
print(system.device)

print("🔹 Training...")
system.fit(dataset, epochs=30, batch_size=1)

# -----------------------------
# تجربة توصية لمستخدم حقيقي (raw user id -> تحويل الى idx)
# -----------------------------
# اختَر مستخدما عشوائياً من القاموس

# -------------------------------
print("\n🔹 Recommendations for existing user u1:")
print(system.recommend(4, top_n=6))
system.device

# -------------------------------
# إضافة مستخدم جديد
# -------------------------------
new_user_id = "u_n"
system.add_user(new_user_id)

print(f"\n🔹 Recommendations for NEW user {new_user_id}:")
print(system.recommend(21, top_n=3))


for n in range(0, number_of_user):
    print(n)
    print(system.recommend(n, top_n=6))






system.save('model/training_model/Hybrid_model_v4.pt')

id_maps.user2idx

import pickle
import os
v = 7
os.makedirs("model", exist_ok=True)

# # 1) حفظ النموذج
# rec.save(f"model/Hybrid_model_v{v}.pt")

# 2) حفظ idmaps
with open(f"model/file_saved/idmaps_v{v}.pkl", "wb") as f:
    pickle.dump(id_maps, f)

# 3) حفظ item_encoder
with open(f"model/file_saved/item_encoder_v{v}.pkl", "wb") as f:
    pickle.dump(item_encoder, f)

# 4) (اختياري) حفظ seen_sets
with open(f"model/file_saved/seen_sets_v{v}.pkl", "wb") as f:
    pickle.dump(seen_sets, f)


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