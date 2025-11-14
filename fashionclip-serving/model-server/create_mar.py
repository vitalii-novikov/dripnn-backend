import os
import subprocess

MODEL_DIR = "model-server"
MODEL_NAME = "fashionclip"
MAR_VERSION = "1.0"
HANDLER_FILE = "fashionclip_handler.py"
CONFIG_FILE = "config.properties"


EXPORT_PATH = os.path.join(MODEL_DIR, "model-store")


os.makedirs(EXPORT_PATH, exist_ok=True)


cmd = [
    "torch-model-archiver",
    "--model-name", MODEL_NAME,
    "--version", MAR_VERSION,
    "--model-file", HANDLER_FILE,
    "--handler", HANDLER_FILE,
    "--extra-files", CONFIG_FILE,
    "--export-path", EXPORT_PATH,
    "--force"
]

print("Running:", " ".join(cmd))
subprocess.run(cmd, check=True)
print(f"Model archive created in {EXPORT_PATH}/{MODEL_NAME}.mar")
