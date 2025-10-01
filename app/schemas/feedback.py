from pydantic import BaseModel

class FeedbackCreate(BaseModel):
    user_id: int
    item_id: int
    liked: bool
