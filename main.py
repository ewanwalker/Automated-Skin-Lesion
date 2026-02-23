from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
from PIL import Image
import io

app = FastAPI()
model = YOLO("runs/classify/train/weights/best.pt")

LABELS = {0: "benign", 1: "malignant"}  

@app.post("/images")
async def upload_image(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))  
    
    results = model(image)  
    probs = results[0].probs

    return {
        "prediction": LABELS[probs.top1],
        "confidence": float(probs.top1conf),  
    }