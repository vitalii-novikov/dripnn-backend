from transformers import CLIPProcessor, CLIPModel
import torch
from app.core.config import MODEL_NAME

model = CLIPModel.from_pretrained(MODEL_NAME)
processor = CLIPProcessor.from_pretrained(MODEL_NAME)

def classify_style(image_path: str, candidate_styles: list[str]):
    inputs = processor(text=candidate_styles, images=image_path, return_tensors="pt", padding=True)
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1).tolist()[0]

    main_idx = int(torch.argmax(logits_per_image))
    return {
        "main_style": candidate_styles[main_idx],
        "confidence": probs[main_idx],
        "all": dict(zip(candidate_styles, probs))
    }
