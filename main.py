from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image
import torch

app = FastAPI()

# Add CORS middleware to allow all origins (for testing)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change "*" to your frontend URL in production for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = ViTForImageClassification.from_pretrained("lthomas8/hair_type_classifier")
processor = ViTImageProcessor.from_pretrained("lthomas8/hair_type_classifier")
id2label = model.config.id2label

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = Image.open(file.file).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    probs = outputs.logits.softmax(dim=-1)
    predicted_class_idx = torch.argmax(probs).item()
    predicted_class_label = id2label[predicted_class_idx]
    confidence = round(probs[0][predicted_class_idx].item(), 4)
    return {"label": predicted_class_label, "confidence": confidence}
