import torch
import io
from PIL import Image
import numpy as np
from transformers import CLIPModel, CLIPProcessor
from ts.torch_handler.base_handler import BaseHandler

STYLES = [
    "Casual", "Business Casual", "Formal", "Sport/Activewear",
    "Streetwear", "Minimalist", "Home wear", "Trendy/Fashion-forward"
]

class ClipHandler(BaseHandler):
    def initialize(self, ctx):
        model_dir = ctx.system_properties.get("model_dir")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(model_dir).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_dir)
        self.model.eval()
        self.initialized = True

    def preprocess(self, data):
        # Берём файл из POST multipart
        image_bytes = None
        if "body" in data[0]:
            image_bytes = data[0].get("body")
        elif "data" in data[0]:
            image_bytes = data[0].get("data")
        else:
            raise ValueError("No file found in request")

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        inputs = self.processor(
            images=image,
            text=[f"This is {s} clothing" for s in STYLES],
            return_tensors="pt",
            padding=True
        )
        return {k: v.to(self.device) for k, v in inputs.items()}

    def inference(self, inputs):
        with torch.no_grad():
            outputs = self.model(**inputs)
            image_features = outputs.image_embeds
            text_features = outputs.text_embeds

            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            sims = (image_features @ text_features.T).squeeze(0).cpu().numpy()
            probs = np.exp(sims) / np.sum(np.exp(sims))

            top_idx = probs.argsort()[-2:][::-1]
            main_style = (STYLES[top_idx[0]], float(round(probs[top_idx[0]]*100,2)))
            secondary_style = (STYLES[top_idx[1]], float(round(probs[top_idx[1]]*100,2)))
            embedding = image_features.squeeze(0).cpu().numpy().tolist()

            return {
                "main_style": main_style[0],
                "main_confidence": main_style[1],
                "secondary_style": secondary_style[0],
                "secondary_confidence": secondary_style[1],
                "embedding_dim": len(embedding),
                "embedding": embedding
            }

    def postprocess(self, inference_output):
        return inference_output
