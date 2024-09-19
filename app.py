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
import tensorflow as tf
from tensorflow.keras.preprocessing import image as keras_image
from keras.applications.mobilenet import preprocess_input
from keras.models import Model
from keras.layers import Dense, Dropout
from utils.score_utils import mean_score, std_score

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

class ImageResult(BaseModel):
    filename: str
    mean_score: float
    status: str
    url: str

RATE_LIMIT_SECONDS = 15
last_upload_time = defaultdict(lambda: 0)  # last upload time per IP

def load_nima_model():
    base_model = tf.keras.applications.MobileNet(input_shape=(None, None, 3), alpha=1, include_top=False, pooling='avg', weights=None)
    x = Dropout(0.75)(base_model.output)
    x = Dense(10, activation='softmax')(x)
    model = Model(base_model.input, x)
    
    model.load_weights('weights/mobilenet_weights.h5')
    return model

nima_model = load_nima_model()

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

# delete a file after a delay
def delete_file_after_delay(file_path: str, delay: int):
    time.sleep(delay)
    if os.path.exists(file_path):
        os.remove(file_path)

# Helper function to preprocess images for the NIMA MobileNet model
def preprocess_image(img_path: str, target_size=(224, 224)):
    img = keras_image.load_img(img_path, target_size=target_size)  # Resize image to 224x224 for MobileNet
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)

# get NIMA score for a given image
def get_nima_score(img_path: str):
    img = preprocess_image(img_path)
    scores = nima_model.predict(img, batch_size=1, verbose=0)[0]
    
    mean = mean_score(scores)
    std = std_score(scores)
    
    return mean

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

        # Get score for the image using NIMA model
        score = get_nima_score(file_location)
        status = "accepted" if score > 5.01 else "rejected"  # Threshold for acceptance

        results.append({
            "filename": file.filename,
            "mean_score": score,
            "status": status,
            "url": f"/static/{file.filename}"
        })

        # Start a background thread to delete the file after a delay
        threading.Thread(target=delete_file_after_delay, args=(file_location, 30)).start()

    return results
