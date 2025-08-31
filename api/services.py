# api/services.py
import torch
import pandas as pd
import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score
from Hybrid import HybridRecommender, build_id_maps

class RecommenderService:
    def __init__(self, model_path="model\Hybrid_model_v1.pt", ratings_df=None, items_df=None, orders_df=None):
        # تحميل البيانات (ممكن تستبدلها بقاعدة بيانات)
        self.ratings_df = ratings_df
        self.items_df = items_df
        self.orders_df = orders_df

        # بناء الخرائط
        self.user2idx, self.item2idx, self.idx2user, self.idx2item = build_id_maps(ratings_df, orders_df)

        # تحميل النموذج
        n_users = len(self.user2idx)
        n_items = len(self.item2idx)
        n_features = 3  # (سنة التصنيع, الشركة, الاسم)
        self.model = HybridRecommender(n_users, n_items, n_features)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

    def recommend(self, user_id: int, top_k: int = 5):
        if user_id not in self.user2idx:
            return []

        user_idx = self.user2idx[user_id]
        item_indices = list(self.item2idx.values())

        # تجهيز features للقطع
        item_features = torch.tensor(self.items_df[["year", "company_id", "name_id"]].values, dtype=torch.float32)

        # حساب التوصيات
        scores = self.model(torch.tensor([user_idx]), torch.tensor(item_indices), item_features).detach().numpy()
        top_indices = scores.argsort()[::-1][:top_k]

        recs = []
        for idx in top_indices:
            item_id = self.idx2item[idx]
            recs.append({
                "item_id": int(item_id),
                "score": float(scores[idx]),
                "name": str(self.items_df.loc[self.items_df["item_id"] == item_id, "name"].values[0])
            })
        return recs

    
    def smart_search(self, query: str, top_k: int = 5):
        query = query.lower()
        matches = self.items_df[self.items_df["name"].str.lower().str.contains(query) |
                                self.items_df["company"].str.lower().str.contains(query) |
                                self.items_df["year"].astype(str).str.contains(query)]
        results = matches.head(top_k).to_dict(orient="records")
        return results

    def evaluate(self, k: int = 5):
        precisions, recalls, ndcgs = [], [], []

        for user in self.ratings_df["user_id"].unique():
            # العناصر الفعلية التي تفاعل معها المستخدم
            true_items = set(self.orders_df[self.orders_df["user_id"] == user]["item_id"].values)

            if not true_items:
                continue

            # التوصيات
            recs = self.recommend(user, top_k=k)
            rec_items = [r["item_id"] for r in recs]

            # Precision@K
            hits = len(set(rec_items) & true_items)
            precision = hits / k
            recall = hits / len(true_items)

            # NDCG@K
            y_true = np.isin(list(self.item2idx.keys()), list(true_items)).astype(int).reshape(1, -1)
            y_score = np.zeros((1, len(self.item2idx)))
            for i, r in enumerate(rec_items):
                if r in self.item2idx:
                    y_score[0, self.item2idx[r]] = 1.0 / (i + 1)

            ndcg = ndcg_score(y_true, y_score, k=k)

            precisions.append(precision)
            recalls.append(recall)
            ndcgs.append(ndcg)

        return {
            "precision_at_k": float(np.mean(precisions)),
            "recall_at_k": float(np.mean(recalls)),
            "ndcg_at_k": float(np.mean(ndcgs))
        }