from fastapi import FastAPI
import pickle
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import torch
import pandas as pd
import numpy as np
# fromHybrid import RecommenderSystemHybrid, build_id_maps, ItemFeatureEncoder

v = 2
# ---------------------------
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Hybrid import RecommenderSystemHybrid, build_id_maps, ItemFeatureEncoder


# ----------------------------
# تعريف API
app = FastAPI(title="../Hybrid Recommender API")

# ----------------------------
# نموذج البيانات القادمة
# ----------------------------
class RequestData(BaseModel):
    user_id: str
    top_n: Optional[int] = 5

class TrainData(BaseModel):
    ratings: List[Dict[str, Any]]
    items: List[Dict[str, Any]]

# ----------------------------
# تحميل النموذج المدرب مسبقاً
# ----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # هذا يعطيك مسار api/
idmaps_path = os.path.join(BASE_DIR, "..", "model", "file_saved", "idmaps_v2.pkl")
item_encoder_path = os.path.join(BASE_DIR, "..", "model", "file_saved", "item_encoder_v2.pkl")
seen_sets_path   = os.path.join(BASE_DIR, "..", "model", "file_saved", "seen_sets_v2.pkl")
MODEL_PATH   = os.path.join(BASE_DIR, "..", "model", "training_model", "Hybrid_model_v2.pt")

# MODEL_PATH = f"../model/training_model/Hybrid_model_v2.pt"

with open(idmaps_path, "rb") as f:
    idmaps = pickle.load(f)

with open(item_encoder_path, "rb") as f:
    item_encoder = pickle.load(f)

with open(seen_sets_path, "rb") as f:
    seen_sets = pickle.load(f)

rec = RecommenderSystemHybrid(
    num_users=len(idmaps.user2idx),
    num_items=len(idmaps.item2idx),
    item_encoder=item_encoder,
    dim=64,
    lr=1e-3,
    device="cpu"
)
rec.load(MODEL_PATH)


# ----------------------------
# دالة مساعدة: تحويل numpy -> Python
# ----------------------------
def to_python_type(obj):
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: to_python_type(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_python_type(v) for v in obj]
    return obj


# ----------------------------
# Endpoint: توصية للمستخدم
# ----------------------------
@app.post("/recommend")
def recommend(data: RequestData):
    if data.user_id not in idmaps.user2idx:
        return {"error": f"User {data.user_id} not found"}

    user_idx = idmaps.user2idx[data.user_id]
    seen_items = seen_sets.get(data.user_id, set())

    top_items = rec.recommend(user_idx, top_n=data.top_n, seen_items=seen_items)
    # هنا top_items ممكن يكون numpy
    top_items = [to_python_type(i) for i in top_items]

    top_item_ids = [idmaps.idx2item[i] for i in top_items]

    return {
        "user_id": data.user_id,
        "recommendations": to_python_type(top_item_ids)
    }


# ----------------------------
# Endpoint: إعادة تدريب النموذج
# ----------------------------
@app.post("/retrain")
def retrain(data: TrainData):
    global rec, idmaps, item_encoder, seen_sets

    # 1) تحويل البيانات القادمة إلى DataFrame
    ratings_df = pd.DataFrame(data.ratings)  # user_id, item_id, rating
    items_df = pd.DataFrame(data.items)      # item_id, name, manufacturer, year

    # 2) بناء idmaps من جديد
    idmaps = build_id_maps(ratings_df, items_df)

    # 3) بناء item encoder
    item_encoder = ItemFeatureEncoder(items_df, idmaps)

    # 4) بناء نموذج جديد
    rec = RecommenderSystemHybrid(
        num_users=len(idmaps.user2idx),
        num_items=len(idmaps.item2idx),
        item_encoder=item_encoder,
        dim=64,
        lr=1e-3,
        device="cpu"
    )

    # 5) تدريب (مثال مبسط: 5 epochs)
    rec.fit(ratings_df, epochs=5, batch_size=128)

    # 6) بناء seen_sets
    seen_sets = {}
    for row in ratings_df.itertuples():
        seen_sets.setdefault(row.user_id, set()).add(row.item_id)

    # 7) حفظ كل شيء
    rec.save(MODEL_PATH)

    with open(f"model/file_saved/idmaps_v{v}.pkl", "wb") as f:
        pickle.dump(idmaps, f)

    with open(f"model/file_saved/item_encoder_v{v}.pkl", "wb") as f:
        pickle.dump(item_encoder, f)

    with open(f"model/file_saved/seen_sets_v{v}.pkl", "wb") as f:
        pickle.dump(seen_sets, f)

    return {"status": "Model retrained and saved successfully"}
