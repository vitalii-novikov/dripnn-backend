from sqlalchemy import Column, Integer, Boolean, ForeignKey, TIMESTAMP, func
from app.core.database import Base

class Feedback(Base):
    __tablename__ = "feedback"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, index=True)
    item_id = Column(Integer, ForeignKey("items.id"))
    liked = Column(Boolean)
    created_at = Column(TIMESTAMP, server_default=func.now())
