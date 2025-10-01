from fastapi import APIRouter
from app.api import items, feedback, portrait, embeddings

api_router = APIRouter()

api_router.include_router(items.router, prefix="/items", tags=["items"])
api_router.include_router(feedback.router, prefix="/feedback", tags=["feedback"])
api_router.include_router(portrait.router, prefix="/portrait", tags=["portrait"])
api_router.include_router(embeddings.router, prefix="/embeddings", tags=["embeddings"])
