from fastapi import FastAPI, UploadFile, File, HTTPException
import requests

app = FastAPI(title="FashionCLIP API")

TSSERVER_URL = "http://modelserver:8080/predictions/fashionclip"

@app.post("/embed")
async def embed(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        files = {"data": image_bytes}
        #files = {"data": (file.filename, image_bytes, file.content_type)}
        response = requests.post(TSSERVER_URL, files=files)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=str(e))
