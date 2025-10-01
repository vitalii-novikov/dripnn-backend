from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.models.items import Item
from app.schemas.items import ItemBase

router = APIRouter()

@router.get("/", response_model=list[ItemBase])
def get_items(db: Session = Depends(get_db), limit: int = 10):
    return db.query(Item).limit(limit).all()
