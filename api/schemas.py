# api/schemas.py
from pydantic import BaseModel
from typing import List, Optional

class RecommendationRequest(BaseModel):
    user_id: int
    top_k: Optional[int] = 5

class RecommendationResponse(BaseModel):
    user_id: int
    recommendations: List[dict]

class SearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5

class SearchResponse(BaseModel):
    results: List[dict]

class EvaluationResponse(BaseModel):
    precision_at_k: float
    recall_at_k: float
    ndcg_at_k: float