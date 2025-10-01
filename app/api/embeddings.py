from fastapi import APIRouter, UploadFile
from app.services.clip_model import classify_style

router = APIRouter()

@router.post("/")
async def get_embedding(file: UploadFile):
    result = classify_style(file.file, ["Casual", "Formal", "Sports", "Party"])
    return result
