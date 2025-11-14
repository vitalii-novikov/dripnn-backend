import torch
from transformers import AutoProcessor, AutoModel
from ts.torch_handler.base_handler import BaseHandler
from PIL import Image
import io
import os

class FashionCLIPHandler(BaseHandler):

    def initialize(self, ctx):
        self.manifest = ctx.manifest
        properties = ctx.system_properties
        model_dir = properties.get("model_dir")  # путь к папке с .mar, где лежат все файлы модели

        # Используем локальный путь вместо repo_id Hugging Face
        local_model_path = os.path.join(model_dir, "patrickjohnson-fashionclip-vit-base-patch32")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Загружаем модель и процессор из локальной папки
        self.model = AutoModel.from_pretrained(local_model_path).to(self.device)
        self.processor = AutoProcessor.from_pretrained(local_model_path)

        self.initialized = True

    def preprocess(self, data):
        image_bytes = data[0].get("data") or data[0].get("body")
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        return inputs

    def inference(self, inputs):
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
            return image_features.cpu().tolist()

    def postprocess(self, inference_output):
        return [{"embeddings": inference_output[0]}]
