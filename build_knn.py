import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import joblib
import os
import time
import wandb

DATA_DIR = "data"
CSV_PATH = os.path.join(DATA_DIR, "embeddings.csv")
KNN_PATH = os.path.join(DATA_DIR, "knn_model.pkl")
META_PATH = os.path.join(DATA_DIR, "item_meta.pkl")

# --- Init W&B run for training ---
run = wandb.init(project="fashion-recommender", job_type="build_knn")

print("🔹 Loading embeddings...")
df = pd.read_csv(CSV_PATH)
df["embedding"] = df["embedding"].apply(lambda x: np.array([float(v) for v in x.split(",")]))
X = np.vstack(df["embedding"].to_numpy())

wandb.log({"num_embeddings": len(X), "embedding_dim": X.shape[1]})

print(f"🔹 Fitting KNN model on {len(X)} embeddings...")
t0 = time.time()
knn = NearestNeighbors(n_neighbors=10, metric="cosine")
knn.fit(X)
wandb.log({"train_time_sec": time.time() - t0})

print("🔹 Saving model and metadata...")
joblib.dump(knn, KNN_PATH)
meta = df[["id", "name", "url"]].to_dict(orient="records")
joblib.dump(meta, META_PATH)

# --- Log artifacts to W&B ---
artifact = wandb.Artifact("knn_model", type="model")
artifact.add_file(KNN_PATH)
artifact.add_file(META_PATH)
wandb.log_artifact(artifact)

print("✅ Done! Saved:")
print(f" - {KNN_PATH}")
print(f" - {META_PATH}")

run.finish()
