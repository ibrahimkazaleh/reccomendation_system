# api/main.py
from fastapi import FastAPI, HTTPException
from api.schemas import RecommendationRequest, RecommendationResponse, SearchRequest, SearchResponse, EvaluationResponse
from api.services import RecommenderService
import pandas as pd

# 1. تجهيز البيانات (مؤقتاً توليد عشوائي/قراءة CSV)
# ratings_df = pd.read_csv("ratings.csv")
# orders_df = pd.read_csv("orders.csv")
# items_df = pd.read_csv("items.csv")

ratings_df = pd.DataFrame()
orders_df = pd.DataFrame()
items_df = pd.DataFrame()
# 2. تهيئة الخدمة
service = RecommenderService(
    model_path="model\Hybrid_model_v1.pt",
    ratings_df=ratings_df,
    orders_df=orders_df,
    items_df=items_df
)

# 3. تعريف التطبيق
app = FastAPI(title="Hybrid Recommender API")

@app.post("/recommendations", response_model=RecommendationResponse)
def get_recommendations(request: RecommendationRequest):
    try:
        recs = service.recommend(user_id=request.user_id, top_k=request.top_k)
        return RecommendationResponse(user_id=request.user_id, recommendations=recs)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search", response_model=SearchResponse)
def smart_search(request: SearchRequest):
    try:
        results = service.smart_search(query=request.query, top_k=request.top_k)
        return SearchResponse(results=results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/evaluate", response_model=EvaluationResponse)
def evaluate_model(k: int = 5):
    try:
        results = service.evaluate(k=k)
        return EvaluationResponse(**results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))