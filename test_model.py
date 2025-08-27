import pandas as pd
import random
import numpy as np
import Collaborative
from Collaborative_v1 import RecommenderSystem, CollabDataset, build_seen_sets,build_id_maps,prepare_interactions

# المستخدمين والعناصر
users = [chr(97+i) for i in range(10)]  # ['a','b',...,'j']
items = list(range(100, 110))  # [100,...,109]

# إنشاء بيانات التقييمات (Ratings)
ratings_data = []
for _ in range(50):  # 50 تقييم عشوائي
    ratings_data.append({
        "user_id": random.choice(users),
        "item_id": random.choice(items),
        "rating": round(random.uniform(1.0, 5.0), 1)
    })

ratings = pd.DataFrame(ratings_data)

# إنشاء بيانات الطلبات (Orders)
orders_data = []
for _ in range(20):  # 100 عملية شراء عشوائية
    orders_data.append({
        "user_id": random.choice(users),
        "item_id": random.choice(items)
    })

orders = pd.DataFrame(orders_data)

# عرض عينة من البيانات
print("Ratings Sample:")
print(ratings.head())
print("\nOrders Sample:")
print(orders.head())


# ================================================

user2idx, item2idx, idx2user, idx2item = build_id_maps(ratings, orders)
inter = prepare_interactions(ratings, orders, user2idx, item2idx)

num_users = len(user2idx)
num_items = len(item2idx)
ds = CollabDataset(inter, num_items=num_items, neg_ratio=1.0)
ds.all
from torch.utils.data import Dataset, DataLoader
data_loader = DataLoader(ds, batch_size=1, shuffle=True, num_workers=0)
for batch in data_loader:
                users = batch["user"]
                items = batch["item"]
                y_exp = batch["y_explicit"]
                y_impl = batch["y_implicit"]
                m_exp = batch["has_explicit"]
                m_impl = batch["has_implicit"]


rec = RecommenderSystem(num_users=num_users, num_items=num_items, dim=64, lr=1e-3)
rec.device
rec.fit(ds, epochs=30, batch_size=1)

seen_sets = build_seen_sets(inter)

for n in range(0, 8):
    print(n)
    recs_idx ,scores,l = rec.recommend(n, top_n=5, seen_items=seen_sets.get(n, set()))
    print(scores)
    print(f"Top-{n} recommendations:",recs_idx)

# -----------------------------------------------------
# ------------- visualization -------------------------
# -----------------------------------------------------


import matplotlib.pyplot as plt
user2idx
id = 0 
plot_rationg = ratings.copy()
plot_rationg['user_id'] = [a for a,_ in user2idx.items()]
ratings[['user_idx','item_id','rating']].iloc[:50].plot(kind='scatter',x='user_idx',y='item_id',c='rating',colormap='viridis')

ratings[ratings['user_idx']==0].plot(kind='scatter',x='item_id',y='rating',c='rating',colormap='viridis')

orders.plot(kind='scatter',x='item_id',y='user_id')

inter[inter['user_idx']==0].plot(kind='bar',x='item_idx',y='y_implicit')
ds.all
orders[orders['user_id']=='c'].plot(kind='scatter',x='item_id',y='user_id')
rows=[]





new_user_id = []
for i in range(len(ratings)):
    new_user_id.append({'user_idx':user2idx[ratings['user_id'][i]]})

ratings["user_idx"] = [row["user_idx"] for row in new_user_id]

new_item_id = []
for i in range(len(orders)):
    new_item_id.append({'item_idx':item2idx[ratings['item_id'][i]]})


