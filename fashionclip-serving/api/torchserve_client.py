import requests
import os

TORCHSERVE_URL = os.environ.get("TORCHSERVE_URL", "http://modelserver:8080")
TOKEN = os.environ.get("TORCHSERVE_TOKEN", None)

def get_embedding(image_bytes):
    headers = {"Authorization": f"Bearer {TOKEN}"} if TOKEN else {}
    response = requests.post(
        f"{TORCHSERVE_URL}/predictions/fashionclip",
        data=image_bytes,
        headers=headers
    )
    response.raise_for_status()
    return response.json()[0]["embeddings"]
