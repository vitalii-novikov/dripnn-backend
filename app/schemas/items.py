from pydantic import BaseModel

class ItemBase(BaseModel):
    id: int
    articleType: str
    style_1: str
    style_2: str
    style_3: str
    link: str

    class Config:
        orm_mode = True
