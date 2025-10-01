import os
from dotenv import load_dotenv

load_dotenv()

# Cloud SQL settings
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")
DB_NAME = os.getenv("DB_NAME", "fashiondb")
DB_HOST = os.getenv("DB_HOST", "127.0.0.1")  # локально можно localhost
DB_PORT = os.getenv("DB_PORT", "5432")

# For Cloud SQL (Unix socket)
INSTANCE_CONNECTION_NAME = os.getenv("INSTANCE_CONNECTION_NAME")  # project:region:instance
USE_CLOUD_SQL_AUTH_PROXY = os.getenv("USE_CLOUD_SQL_AUTH_PROXY", "false").lower() == "true"

if INSTANCE_CONNECTION_NAME and not USE_CLOUD_SQL_AUTH_PROXY:
    # Unix socket
    DATABASE_URL = (
        f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}"
        f"@/{DB_NAME}?host=/cloudsql/{INSTANCE_CONNECTION_NAME}"
    )
else:
    # TCP connection
    DATABASE_URL = (
        f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}"
        f"@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    )

MODEL_NAME = os.getenv("MODEL_NAME", "openai/clip-vit-base-patch32")
