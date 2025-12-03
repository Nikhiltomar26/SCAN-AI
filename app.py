"""
app.py - FastAPI Application
Exposes REST API endpoint for medical report processing

Setup Instructions:
1. Install dependencies: pip install -r requirements.txt
2. Create a .env file in the project root with: GROQ_API_KEY=your_key_here
3. Run the server: python app.py
4. Open http://localhost:8000 in your browser
"""

import os
import tempfile
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from model import MedicalReportProcessor
import traceback
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


# Initialize FastAPI app
app = FastAPI(title="Medical Report Analyzer", version="1.0.0")

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy-initialized ML processor (created on first request)
processor = None


@app.get("/")
async def serve_frontend():
    """Serve the frontend HTML page"""
    return FileResponse("index.html")


@app.get("/style.css")
async def serve_css():
    """Serve the CSS file"""
    return FileResponse("style.css")


@app.get("/script.js")
async def serve_js():
    """Serve the JavaScript file"""
    return FileResponse("script.js")


@app.post("/api/analyze")
async def analyze_report(file: UploadFile = File(...)):
    """
    Endpoint to analyze uploaded medical report image
    
    Args:
        file: Uploaded image file (JPEG, PNG, etc.)
        
    Returns:
        JSON response with:
        - raw_text: Text extracted via OCR
        - explanation: Plain-language explanation
        - highlights: Key findings
        - success: Boolean status
        
    Raises:
        HTTPException: If processing fails
    """
    # Validate file type
    allowed_types = ["image/jpeg", "image/jpg", "image/png", "image/bmp", "image/tiff"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed types: {', '.join(allowed_types)}"
        )
    
    # Create temporary file to store uploaded image
    temp_file = None
    try:
        # Read uploaded file
        contents = await file.read()
        
        # Validate file size (max 10MB)
        if len(contents) > 10 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File too large. Maximum size is 10MB")
        
        # Save to temporary file
        suffix = os.path.splitext(file.filename)[1] or ".jpg"
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        temp_file.write(contents)
        temp_file.close()
        
        # Ensure processor is initialized (lazy init to allow uvicorn import/test)
        global processor
        if processor is None:
            try:
                processor = MedicalReportProcessor()
            except ValueError as e:
                # missing API key or client init failure
                raise HTTPException(status_code=500, detail=str(e))

        # Process the medical report
        result = processor.process_medical_report(temp_file.name)
        
        return JSONResponse(content={
            "success": True,
            "raw_text": result["raw_text"],
            "explanation": result["explanation"],
            "highlights": result["highlights"]
        })
    
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    
    except Exception as e:
        # Log the full error for debugging
        print(f"Error processing report: {str(e)}")
        print(traceback.format_exc())
        
        # Return user-friendly error
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process medical report: {str(e)}"
        )
    
    finally:
        # Clean up temporary file
        if temp_file and os.path.exists(temp_file.name):
            try:
                os.unlink(temp_file.name)
            except:
                pass


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "Medical Report Analyzer"}


if __name__ == "__main__":
    import uvicorn
    # Run the server (uvicorn) - keep this simple for testing
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)