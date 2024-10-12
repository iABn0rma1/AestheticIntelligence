from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from typing import List
import os
import numpy as np
from PIL import Image
import time
import threading
import logging
from collections import defaultdict
import tensorflow as tf
from tensorflow.keras.preprocessing import image as keras_image
from keras.applications.mobilenet import preprocess_input
from keras.models import Model
from keras.layers import Dense, Dropout
from utils.score_utils import mean_score, std_score
import aiofiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

app = FastAPI()

print("Initializing FastAPI...")

# Middleware setup for CORS and GZIP compression
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)  # GZIP compression for large responses

# Mount the static directory for serving files
app.mount("/static", StaticFiles(directory="static"), name="static")

class ImageResult(BaseModel):
    filename: str
    mean_score: float
    status: str
    url: str

RATE_LIMIT_SECONDS = 15
UPLOAD_FOLDER = "static/uploads"
last_upload_time = defaultdict(lambda: 0)

nima_model = None  # Initialize NIMA model globally

def load_nima_model_lazy():
    """Lazy-load the NIMA model only when needed."""
    global nima_model
    if nima_model is None:
        print("Loading NIMA model...")
        base_model = tf.keras.applications.MobileNet(input_shape=(None, None, 3), alpha=1, include_top=False, pooling='avg', weights=None)
        x = Dropout(0.75)(base_model.output)
        x = Dense(10, activation='softmax')(x)
        model = Model(base_model.input, x)
        model.load_weights('weights/mobilenet_weights.h5')
        nima_model = model
    return nima_model

# Function to clear the uploads folder
def clear_upload_folder():
    print("Clearing upload folder")
    try:
        if os.path.exists(UPLOAD_FOLDER):
            for filename in os.listdir(UPLOAD_FOLDER):
                file_path = os.path.join(UPLOAD_FOLDER, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")
            print("Folder cleared")
    except Exception as e:
        print(f"Error clearing folder: {e}")

# Background task to periodically clear the folder
def folder_cleanup_periodically(interval: int):
    while True:
        time.sleep(interval)
        clear_upload_folder()

# Start background thread for periodic cleanup (every 1 hour)
cleanup_thread = threading.Thread(target=folder_cleanup_periodically, args=(60 * 60,))
cleanup_thread.daemon = True  # Ensures thread exits when main program exits
cleanup_thread.start()

# Middleware to log each request
@app.middleware("http")
async def log_requests(request: Request, call_next):
    clear_upload_folder()  # Clear the uploads folder on each request
    response = await call_next(request)
    return response

# Serve static files like images
@app.get("/static/{filename}")
async def get_static_file(filename: str):
    file_path = os.path.join("static", filename)
    if os.path.exists(file_path):
        headers = {"Cache-Control": "public, max-age=3600"}  # Enable caching
        return HTMLResponse(content=open(file_path, "rb").read(), media_type="image/jpeg", headers=headers)
    return JSONResponse(content={"error": "File not found"}, status_code=404)

# Serve the index.html file
@app.get("/", response_class=HTMLResponse)
async def read_index():
    index_file_path = os.path.join("static", "index.html")
    if os.path.exists(index_file_path):
        with open(index_file_path, "r") as f:
            return HTMLResponse(content=f.read())
    return JSONResponse(content={"error": "Index file not found"}, status_code=404)

# Helper function to preprocess images for the NIMA MobileNet model
def preprocess_image(img_path: str, target_size=(224, 224)):
    img = keras_image.load_img(img_path, target_size=target_size)
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)

# Get NIMA score for a given image
def get_nima_score(img_path: str):
    model = load_nima_model_lazy()  # Lazy-load the model
    img = preprocess_image(img_path)
    scores = model.predict(img, batch_size=1, verbose=0)[0]
    mean = mean_score(scores)
    std = std_score(scores)
    return mean

# Delete a file after a longer delay
def delete_file_after_delay(file_path: str, delay: int = 60):
    time.sleep(delay)  # Delay before deletion
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"Deleted after serving: {file_path}")

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
        file_location = os.path.join(UPLOAD_FOLDER, file.filename)
        
        # Asynchronously save the uploaded file
        async with aiofiles.open(file_location, "wb") as f:
            await f.write(await file.read())

        # Get score for the image using NIMA model
        score = get_nima_score(file_location)
        status = "accepted" if score > 5.01 else "rejected"  # Threshold for acceptance

        results.append({
            "filename": file.filename,
            "mean_score": score,
            "status": status,
            "url": f"/static/uploads/{file.filename}"
        })

    # Start a background thread to delete the files 60 seconds after response
    for file in files:
        file_location = os.path.join(UPLOAD_FOLDER, file.filename)
        threading.Thread(target=delete_file_after_delay, args=(file_location, 60)).start()

    return results
