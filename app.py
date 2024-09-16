from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from typing import List
import shutil
import os
import numpy as np
from PIL import Image
import time
import threading
from collections import defaultdict

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

class ImageResult(BaseModel):
    filename: str
    mean_score: float
    status: str
    url: str

RATE_LIMIT_SECONDS = 15
last_upload_time = defaultdict(lambda: 0)  # Store last upload time per IP 

# Serve static files like images
@app.get("/static/{filename}")
async def get_static_file(filename: str):
    file_path = os.path.join("static", filename)
    if os.path.exists(file_path):
        return HTMLResponse(content=open(file_path, "rb").read(), media_type="image/jpeg")
    return JSONResponse(content={"error": "File not found"}, status_code=404)

# Serve the index.html file
@app.get("/", response_class=HTMLResponse)
async def read_index():
    index_file_path = os.path.join("static", "index.html")
    if os.path.exists(index_file_path):
        with open(index_file_path, "r") as f:
            return HTMLResponse(content=f.read())
    return JSONResponse(content={"error": "Index file not found"}, status_code=404)

def delete_file_after_delay(file_path: str, delay: int):
    time.sleep(delay)
    if os.path.exists(file_path):
        os.remove(file_path)

@app.post("/curate/")
async def curate_images(request: Request, files: List[UploadFile] = File(...)):
    client_ip = request.client.host
    current_time = time.time()
    
    # Check rate limit
    if current_time - last_upload_time[client_ip] < RATE_LIMIT_SECONDS:
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Please wait before uploading again.")
    
    # Update last upload time
    last_upload_time[client_ip] = current_time
    
    results = []
    for file in files:
        file_location = os.path.join("static", file.filename)
        with open(file_location, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # Simulate image processing
        image = Image.open(file_location)
        score = np.random.uniform(0, 1)  # Example score
        status = "accepted" if score > 0.6 else "rejected"

        results.append({
            "filename": file.filename,
            "mean_score": score,
            "status": status,
            "url": f"/static/{file.filename}"
        })

        # Start a background thread to delete the file after a delay
        threading.Thread(target=delete_file_after_delay, args=(file_location, 30)).start()

    return results
