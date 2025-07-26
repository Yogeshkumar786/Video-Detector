from fastapi import FastAPI, UploadFile, File, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import shutil
import os
import logging
from inference import load_model, predict_video
from auth import verify_api_key, get_api_key_header

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Video Deepfake Detection API",
    description="API for detecting deepfake videos using cross-modal analysis",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
try:
    model = load_model()
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    model = None

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Video Deepfake Detection API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint for Render"""
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }

@app.post("/upload")
async def upload_video(
    file: UploadFile = File(...),
    authenticated: bool = Depends(verify_api_key)
):
    """Upload video for deepfake detection"""
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        raise HTTPException(status_code=400, detail="Invalid file format")
    
    try:
        file_path = os.path.join("uploads", file.filename)
        os.makedirs("uploads", exist_ok=True)
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info(f"Processing video: {file.filename}")
        prediction = predict_video(file_path, model)
        
        # Clean up uploaded file
        if os.path.exists(file_path):
            os.remove(file_path)
        
        return {
            "filename": file.filename,
            "prediction": prediction,
            "status": "success"
        }
    
    except Exception as e:
        logger.error(f"Error processing video: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api-key-info")
async def api_key_info():
    """Information about API key usage"""
    return {
        "message": "Include your API key in the Authorization header",
        "format": "Bearer YOUR_API_KEY",
        "example": "Authorization: Bearer your-api-key-here"
    }
