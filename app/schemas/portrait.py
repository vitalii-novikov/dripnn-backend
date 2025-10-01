from pydantic import BaseModel
from typing import Dict

class Portrait(BaseModel):
    user_id: int
    styles: Dict[str, float]  # { "Casual": 40.0, "Formal": 30.0, ... }
