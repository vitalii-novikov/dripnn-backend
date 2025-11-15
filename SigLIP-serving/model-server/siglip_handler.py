import torch
import numpy as np
from ts.torch_handler.base_handler import BaseHandler
from transformers import AutoModel, AutoProcessor

STYLES = [
    "Casual", "Business Casual", "Formal", "Sport/Activewear", "Streetwear",
    "Minimalist", "Home wear", "Trendy/Fashion-forward"
]

class SigLipHandler:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModel.from_pretrained("model-store").to(self.device)
        self.processor = AutoProcessor.from_pretrained("model-store")
        self.model.eval()

    def predict(self, image, image_id="unknown"):
        inputs = self.processor(
            text=STYLES,
            images=image,
            return_tensors="pt",
            padding=True
        ).to(self.device)

        with torch.no_grad():
            image_embeds = self.model.get_image_features(pixel_values=inputs["pixel_values"])
            text_embeds = self.model.get_text_features(input_ids=inputs["input_ids"])

        similarities = (image_embeds @ text_embeds.T).squeeze(0).cpu().numpy()
        probs = np.exp(similarities) / np.sum(np.exp(similarities))

        top_idx = probs.argsort()[-2:][::-1]

        main_style = (STYLES[top_idx[0]], float(round(probs[top_idx[0]] * 100, 2)))
        secondary_style = (STYLES[top_idx[1]], float(round(probs[top_idx[1]] * 100, 2)))

        embedding = image_embeds.squeeze(0).cpu().numpy().tolist()

        return {
            "id": image_id,
            "main_style": main_style[0],
            "main_confidence": main_style[1],
            "secondary_style": secondary_style[0],
            "secondary_confidence": secondary_style[1],
            "embedding_dim": len(embedding),
            "embedding": embedding
        }