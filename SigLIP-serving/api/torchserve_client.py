import requests
import base64
import json
from io import BytesIO
from PIL import Image

TORCHSERVE_URL = "http://model-server:8080/predictions/siglip"

def predict_siglip(image: Image.Image):
    buf = BytesIO()
    image.save(buf, format="JPEG")
    img_bytes = buf.getvalue()

    response = requests.post(
        TORCHSERVE_URL,
        files={"image": ("image.jpg", img_bytes, "image/jpeg")},
        data={"id": "uploaded_image"}
    )

    return response.json()
