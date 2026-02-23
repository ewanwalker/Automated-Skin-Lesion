from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from datetime import datetime
from ultralytics import YOLO
from PIL import Image
import sqlite3
import io

app = FastAPI()
model = YOLO("runs/classify/train/weights/best.pt")

LABELS = {0: "Benign", 1: "Malignant"}  

app.mount("/static", StaticFiles(directory="Static"), name="static")

@app.get("/") # Serve the dashboard HTML page
def serve_dashboard():
    return FileResponse("Static/index.html")


@app.post("/images") # API endpoint for image upload and prediction
async def upload_image(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    results = model(image)
    probs = results[0].probs

    prediction = LABELS[probs.top1]
    confidence = float(probs.top1conf)

    log_prediction(file.filename, prediction, confidence)  

    return {"prediction": prediction, "confidence": confidence}


def log_prediction(filename, prediction, confidence): # Log predictions to SQLite database
    con = sqlite3.connect("predictions.db")
    cur = con.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            prediction TEXT,
            confidence REAL,
            timestamp TEXT
        )
    """)
    cur.execute(
        "INSERT INTO predictions VALUES (NULL, ?, ?, ?, ?)",
        (filename, prediction, confidence, datetime.now().isoformat())
    )
    con.commit()
    con.close()

@app.get("/dashboard/stats")
def get_stats():
    con = sqlite3.connect("predictions.db")
    cur = con.cursor()
    
    total = cur.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    benign = cur.execute("SELECT COUNT(*) FROM predictions WHERE prediction='benign'").fetchone()[0]
    malignant = cur.execute("SELECT COUNT(*) FROM predictions WHERE prediction='malignant'").fetchone()[0]
    avg_conf = cur.execute("SELECT AVG(confidence) FROM predictions").fetchone()[0]
    recent = cur.execute("SELECT * FROM predictions ORDER BY id DESC LIMIT 10").fetchall()
    
    con.close()
    return {
        "total": total,
        "benign": benign,
        "malignant": malignant,
        "avg_confidence": round(avg_conf, 3),
        "recent": recent
    }

if __name__ == "__main__": # Run the FastAPI app using Uvicorn
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)