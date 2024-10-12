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
from tensorflow.keras import applications as apps
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

logger.info("Initializing FastAPI...")

# Middleware setup for CORS and GZIP compression
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

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

nima_model = None

def load_nima_model_lazy():
    """Lazy-load the NIMA model only when needed."""
    global nima_model
    try:
        if nima_model is None:
            logger.info("Loading NIMA model...")
            base_model = apps.MobileNet(input_shape=(None, None, 3), alpha=1, include_top=False, pooling='avg', weights=None)
            x = Dropout(0.75)(base_model.output)
            x = Dense(10, activation='softmax')(x)
            model = Model(base_model.input, x)
            model.load_weights('weights/mobilenet_weights.h5')
            nima_model = model
        return nima_model
    except Exception as e:
        logger.error(f"Failed to load NIMA model: {e}")
        raise HTTPException(status_code=500, detail="Model loading failed.")

def clear_upload_folder():
    """Function to clear the uploads folder."""
    try:
        logger.info("Clearing upload folder")
        if os.path.exists(UPLOAD_FOLDER):
            for filename in os.listdir(UPLOAD_FOLDER):
                file_path = os.path.join(UPLOAD_FOLDER, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    logger.info(f"Deleted: {file_path}")
        logger.info("Folder cleared")
    except Exception as e:
        logger.error(f"Error clearing folder: {e}")

def folder_cleanup_periodically(interval: int):
    """Background task to periodically clear the folder."""
    while True:
        time.sleep(interval)
        clear_upload_folder()

# Start background thread for periodic cleanup
cleanup_thread = threading.Thread(target=folder_cleanup_periodically, args=(60 * 60,))
cleanup_thread.daemon = True
cleanup_thread.start()

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Middleware to log each request."""
    try:
        response = await call_next(request)
        return response
    except Exception as e:
        logger.error(f"Error during request processing: {e}")
        return JSONResponse(status_code=500, content={"error": "Internal server error"})

@app.get("/static/{filename}")
async def get_static_file(filename: str):
    """Serve static files like images."""
    try:
        file_path = os.path.join("static", filename)
        if os.path.exists(file_path):
            headers = {"Cache-Control": "public, max-age=3600"}
            return HTMLResponse(content=open(file_path, "rb").read(), media_type="image/jpeg", headers=headers)
        return JSONResponse(content={"error": "File not found"}, status_code=404)
    except Exception as e:
        logger.error(f"Error serving static file: {e}")
        return JSONResponse(content={"error": "Failed to retrieve file"}, status_code=500)

@app.get("/", response_class=HTMLResponse)
async def read_index():
    """Serve the index.html file."""
    try:
        index_file_path = os.path.join("static", "index.html")
        if os.path.exists(index_file_path):
            with open(index_file_path, "r") as f:
                return HTMLResponse(content=f.read())
        return JSONResponse(content={"error": "Index file not found"}, status_code=404)
    except Exception as e:
        logger.error(f"Error serving index file: {e}")
        return JSONResponse(content={"error": "Failed to load index page"}, status_code=500)

@app.get("/about", response_class=HTMLResponse)
async def read_index():
    """Serve the index.html file."""
    try:
        about_file_path = os.path.join("static", "about.html")
        if os.path.exists(about_file_path):
            with open(about_file_path, "r") as f:
                return HTMLResponse(content=f.read())
        return JSONResponse(content={"error": "About file not found"}, status_code=404)
    except Exception as e:
        logger.error(f"Error serving index file: {e}")
        return JSONResponse(content={"error": "Failed to load index page"}, status_code=500)

def preprocess_image(img_path: str, target_size=(224, 224)):
    """Helper function to preprocess images for the NIMA MobileNet model."""
    try:
        img = keras_image.load_img(img_path, target_size=target_size)
        img_array = keras_image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        return preprocess_input(img_array)
    except Exception as e:
        logger.error(f"Failed to preprocess image {img_path}: {e}")
        raise HTTPException(status_code=500, detail="Image preprocessing failed.")

def get_nima_score(img_path: str):
    """Get NIMA score for a given image."""
    try:
        model = load_nima_model_lazy()
        img = preprocess_image(img_path)
        scores = model.predict(img, batch_size=1, verbose=0)[0]
        mean = mean_score(scores)
        std = std_score(scores)
        return mean
    except Exception as e:
        logger.error(f"Error calculating NIMA score: {e}")
        raise HTTPException(status_code=500, detail="NIMA score calculation failed.")

def delete_file_after_delay(file_path: str, delay: int = 60):
    """Delete a file after a specified delay."""
    try:
        time.sleep(delay)
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Deleted after serving: {file_path}")
    except Exception as e:
        logger.error(f"Error deleting file {file_path}: {e}")

@app.post("/curate/")
async def curate_images(request: Request, files: List[UploadFile] = File(...)):
    client_ip = request.client.host
    current_time = time.time()

    # Check rate limit
    if current_time - last_upload_time[client_ip] < RATE_LIMIT_SECONDS:
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Please wait before uploading again.")
    
    last_upload_time[client_ip] = current_time
    results = []

    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)

    for file in files:
        try:
            file_location = os.path.join(UPLOAD_FOLDER, file.filename)
            
            # Asynchronously save the uploaded file
            async with aiofiles.open(file_location, "wb") as f:
                await f.write(await file.read())

            # Get score for the image using NIMA model
            score = get_nima_score(file_location)
            status = "accepted" if score > 5.01 else "rejected"

            results.append({
                "filename": file.filename,
                "mean_score": score,
                "status": status,
                "url": f"/static/uploads/{file.filename}"
            })

            # Start a background thread to delete the file
            threading.Thread(target=delete_file_after_delay, args=(file_location, 60)).start()

        except Exception as e:
            logger.error(f"Error processing file {file.filename}: {e}")
            raise HTTPException(status_code=500, detail=f"Error processing file {file.filename}")

    return results
