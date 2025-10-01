from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.models.feedback import Feedback
from app.models.items import Item
from app.schemas.portrait import Portrait

router = APIRouter()

@router.get("/{user_id}", response_model=Portrait)
def get_portrait(user_id: int, db: Session = Depends(get_db)):
    feedbacks = db.query(Feedback).filter(Feedback.user_id == user_id, Feedback.liked == True).all()
    style_counts = {}
    for fb in feedbacks:
        item = db.query(Item).filter(Item.id == fb.item_id).first()
        if not item:
            continue
        for style in [item.style_1, item.style_2, item.style_3]:
            if style:
                style_counts[style] = style_counts.get(style, 0) + 1

    total = sum(style_counts.values()) or 1
    styles = {s: round(c / total * 100, 2) for s, c in style_counts.items()}
    return {"user_id": user_id, "styles": styles}
