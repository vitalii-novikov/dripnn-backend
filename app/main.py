from fastapi import FastAPI, APIRouter, HTTPException, UploadFile, File, Form, Query
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
import os
import uuid
from google.cloud import storage
from PIL import Image
import io
import torch
from transformers import CLIPProcessor, CLIPModel
import wandb

# --- Load env variables ---
load_dotenv()

print("DEBUG: GOOGLE_APPLICATION_CREDENTIALS - ", os.getenv("GOOGLE_APPLICATION_CREDENTIALS"))
print("DEBUG: keys/gcs-key.json - ", os.path.exists("keys/gcs-key.json"))

# --- Init W&B ---
wandb_project = os.getenv("WANDB_PROJECT", "fashion-recommender")
wandb_entity = os.getenv("WANDB_ENTITY", None)
wandb_run_name = os.getenv("WANDB_RUN_NAME", None)
wandb_mode = os.getenv("WANDB_MODE", "offline")

wandb_run = wandb.init(project=wandb_project, entity=wandb_entity, name=wandb_run_name, mode=wandb_mode)

# --- Database config ---
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")
DB_NAME = os.getenv("DB_NAME", "fashiondb")
DB_HOST = os.getenv("DB_HOST", "127.0.0.1")
DB_PORT = os.getenv("DB_PORT", "5432")

INSTANCE_CONNECTION_NAME = os.getenv("INSTANCE_CONNECTION_NAME")
USE_CLOUD_SQL_AUTH_PROXY = os.getenv("USE_CLOUD_SQL_AUTH_PROXY", "false").lower() == "true"

if INSTANCE_CONNECTION_NAME and not USE_CLOUD_SQL_AUTH_PROXY:
    DATABASE_URL = (
        f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}"
        f"@/{DB_NAME}?host=/cloudsql/{INSTANCE_CONNECTION_NAME}"
    )
else:
    DATABASE_URL = (
        f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}"
        f"@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    )

# --- SQLAlchemy session ---
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# --- Google Cloud Storage ---
BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")
gcs_key = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
if gcs_key:
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.abspath(gcs_key)
storage_client = storage.Client()
bucket = storage_client.bucket(BUCKET_NAME)

# --- CLIP model (local, not API) ---
MODEL_NAME = os.getenv("MODEL_NAME", "openai/clip-vit-base-patch32")
print(f"Loading CLIP model: {MODEL_NAME}")
clip_model = CLIPModel.from_pretrained(MODEL_NAME)
clip_processor = CLIPProcessor.from_pretrained(MODEL_NAME)
clip_model.eval()

# --- Styles list ---
STYLES = [
    "Casual", "Business Casual", "Formal", "Sport/Activewear",
    "Streetwear", "Minimalist", "Home wear", "Trendy/Fashion-forward"
]

# --- FastAPI app setup ---
app = FastAPI(title="Fashion Recommender API")
router = APIRouter()

# --- 1) GET ITEMS ---
@router.get("/items")
def get_items():
    with SessionLocal() as db:
        query = text("""
            SELECT id, name, url, style1, style2, category, type, baseColour, season
            FROM item_unitary
            LIMIT 10;
        """)
        result = db.execute(query).mappings().all()
        return {"items": [dict(row) for row in result]}

# --- 1b) GET FILTERED ITEMS ---
@router.get("/filtered-items")
def get_filtered_items(
    gender: str = Query(None, description="Male or Female"),
    category: str = Query(None, description="Topwear or Bottomwear"),
    season: str = Query(None, description="Fall, Summer, Winter, or Spring"),
):
    filters = []
    params = {}

    if gender:
        filters.append("gender = :gender")
        params["gender"] = gender
    if category:
        filters.append("category = :category")
        params["category"] = category
    if season:
        filters.append("season = :season")
        params["season"] = season

    where_clause = "WHERE " + " AND ".join(filters) if filters else ""
    sql = text(f"""
        SELECT id, name, url, style1, style2, category, type, baseColour, season
        FROM item_unitary
        {where_clause}
        ORDER BY RANDOM()
        LIMIT 10;
    """)

    with SessionLocal() as db:
        result = db.execute(sql, params).mappings().all()
        return {"items": [dict(row) for row in result]}

# --- 2) POST FEEDBACK ---
@router.post("/feedback")
def post_feedback(item_id: int = Form(...), user_id: int = Form(...), feedback: str = Form(...)):
    if feedback not in ["like", "dislike"]:
        raise HTTPException(status_code=400, detail="Feedback must be 'like' or 'dislike'")
    
    with SessionLocal() as db:
        query = text("""
            INSERT INTO feedback (item_id, user_id, feedback)
            VALUES (:item_id, :user_id, :feedback)
        """)
        db.execute(query, {"item_id": item_id, "user_id": user_id, "feedback": feedback})
        db.commit()

    # Log feedback event in W&B
    wandb.log({
        "event": "feedback",
        "user_id": user_id,
        "item_id": item_id,
        "feedback": feedback
    })

    return {"status": "success", "item_id": item_id, "user_id": user_id, "feedback": feedback}

# --- 3) POST EMBEDDINGS (main CLIP inference endpoint) ---
@router.post("/embeddings")
async def post_embeddings(file: UploadFile = File(...), user_id: int = Form(...)):
    try:
        # --- Upload image to GCS ---
        file_bytes = await file.read()
        image = Image.open(io.BytesIO(file_bytes)).convert("RGB")

        filename = f"users_uploads/{uuid.uuid4()}.jpg"
        blob = bucket.blob(filename)
        blob.upload_from_string(file_bytes, content_type="image/jpeg")
        image_url = f"https://storage.googleapis.com/{BUCKET_NAME}/{filename}"

        # --- CLIP processing (local inference) ---
        inputs = clip_processor(text=STYLES, images=image, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = clip_model(**inputs)
            logits_per_image = outputs.logits_per_image.softmax(dim=1).cpu().numpy()[0]

        # --- Get top-2 styles ---
        sorted_idx = logits_per_image.argsort()[::-1]
        style1, style2 = STYLES[sorted_idx[0]], STYLES[sorted_idx[1]]
        style1_conf, style2_conf = float(logits_per_image[sorted_idx[0]]), float(logits_per_image[sorted_idx[1]])

        # --- Save results in DB ---
        with SessionLocal() as db:
            query = text("""
                INSERT INTO item_unitary (url, user_id, style1, style1_conf, style2, style2_conf)
                VALUES (:url, :user_id, :style1, :style1_conf, :style2, :style2_conf)
                RETURNING id;
            """)
            result = db.execute(query, {
                "url": image_url,
                "user_id": user_id,
                "style1": style1,
                "style1_conf": style1_conf,
                "style2": style2,
                "style2_conf": style2_conf,
            }).fetchone()
            db.commit()

        # --- Log prediction in W&B ---
        wandb.log({
            "event": "inference",
            "user_id": user_id,
            "image_url": image_url,
            "style1": style1,
            "style1_conf": style1_conf,
            "style2": style2,
            "style2_conf": style2_conf,
        })

        return {
            "status": "success",
            "id": result[0],
            "url": image_url,
            "style1": style1,
            "style1_conf": style1_conf,
            "style2": style2,
            "style2_conf": style2_conf
        }

    except Exception as e:
        wandb.log({"error": str(e)})
        raise HTTPException(status_code=500, detail=str(e))

# --- Register router ---
app.include_router(router)
