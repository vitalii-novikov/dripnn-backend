from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.models.feedback import Feedback
from app.schemas.feedback import FeedbackCreate

router = APIRouter()

@router.post("/")
def add_feedback(feedback: FeedbackCreate, db: Session = Depends(get_db)):
    fb = Feedback(user_id=feedback.user_id, item_id=feedback.item_id, liked=feedback.liked)
    db.add(fb)
    db.commit()
    return {"message": "Feedback saved"}
