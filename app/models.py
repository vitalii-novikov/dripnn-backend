from sqlalchemy import Column, Integer, String, Text, Float, TIMESTAMP, ForeignKey, Boolean
from sqlalchemy.orm import relationship
from app.database import Base

class ItemUnitary(Base):
    __tablename__ = "item_unitary"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255))
    url = Column(Text)
    gender = Column(String(50))
    category = Column(String(100))
    type = Column(String(100))
    baseColour = Column(String(50))
    season = Column(String(50))
    year = Column(Integer)
    original_style = Column(String(100))
    pred_style_1 = Column(String(100))
    pred_style_2 = Column(String(100))
    style1 = Column(String(100))
    style1_conf = Column(Float)
    style2 = Column(String(100))
    style2_conf = Column(Float)
    embedding = Column(Text)
    user_id = Column(Integer, ForeignKey("users.user_id"))
    create_dttm = Column(TIMESTAMP)

class Feedback(Base):
    __tablename__ = "feedback"

    id = Column(Integer, primary_key=True, index=True)
    item_id = Column(Integer, ForeignKey("item_unitary.id"))
    user_id = Column(Integer, ForeignKey("users.user_id"))
    feedback = Column(Boolean, nullable=False)
    create_dttm = Column(TIMESTAMP)
