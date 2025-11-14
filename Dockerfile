FROM python:3.10-slim

# --- System config ---
# ENV PYTHONDONTWRITEBYTECODE=1
# ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc libpq-dev wget curl git libgl1 libglib2.0-0 \
  && rm -rf /var/lib/apt/lists/*

# --- Python setup ---
# RUN python -m pip install --upgrade pip

# --- Install requirements ---
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# --- App setup ---
WORKDIR /app
COPY . /app

# --- Preload CLIP model to cache ---
RUN python -c "from transformers import CLIPModel, CLIPProcessor; \
CLIPModel.from_pretrained('openai/clip-vit-base-patch32'); \
CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32'); \
print('Model cached successfully ✅')"

# --- Port exposure and startup ---
EXPOSE 8080
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
