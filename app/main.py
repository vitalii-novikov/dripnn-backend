from fastapi import FastAPI
from app.api import api_router
from app.core.database import Base, engine

# Create tables if not exist
Base.metadata.create_all(bind=engine)

app = FastAPI(title="Fashion Recommender API")

# Routes
app.include_router(api_router)
