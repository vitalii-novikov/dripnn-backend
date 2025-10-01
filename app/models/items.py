from sqlalchemy import Column, Integer, String
from app.core.database import Base

class Item(Base):
    __tablename__ = "items"

    id = Column(Integer, primary_key=True, index=True)
    articleType = Column(String)
    style_1 = Column(String)
    style_2 = Column(String)
    style_3 = Column(String)
    link = Column(String)
