from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io
import requests

TORCHSERVE_URL = "http://model-server:8080/predictions/siglip"

app = FastAPI()

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    content = await image.read()
    pil_img = Image.open(io.BytesIO(content)).convert("RGB")

    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG")
    img_bytes = buf.getvalue()

    response = requests.post(
        TORCHSERVE_URL,
        files={"image": ("image.jpg", img_bytes, "image/jpeg")},
        data={"id": "uploaded_image"}
    )

    return response.json()
