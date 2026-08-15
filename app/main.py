import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from processor import run

app = FastAPI(
    title="License Plate Recognition API",
    description="AI-based vehicle license plate detection and recognition API",
    version="1.0.0"
)

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "license_plate_best.pt"
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "output"

UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


@app.get("/")
def root():
    return {
        "message": "License Plate Recognition API is running",
        "version": "1.0.0"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_exists": MODEL_PATH.exists()
    }


@app.post("/process-video")
async def process_video(file: UploadFile = File(...)):

    allowed_extensions = {".mp4", ".avi", ".mov", ".mkv"}
    file_extension = Path(file.filename).suffix.lower()

    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail="Unsupported video format. Use MP4, AVI, MOV, or MKV."
        )

    if not MODEL_PATH.exists():
        raise HTTPException(
            status_code=500,
            detail="License plate model not found."
        )

    job_id = uuid.uuid4().hex
    input_path = UPLOAD_DIR / f"{job_id}{file_extension}"
    output_path = OUTPUT_DIR / f"{job_id}_output.mp4"

    try:
        with open(input_path, "wb") as buffer:
            while chunk := await file.read(1024 * 1024):
                buffer.write(chunk)

        # show_preview=False is required here: there's no display attached
        # to the API process, so cv2.imshow/waitKey would error out or hang.
        run(
            str(input_path),
            str(output_path),
            str(MODEL_PATH),
            show_preview=False
        )

        if not output_path.exists():
            raise HTTPException(
                status_code=500,
                detail="Video processing failed. Output file was not created."
            )

        return FileResponse(
            path=str(output_path),
            media_type="video/mp4",
            filename="processed_video.mp4"
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Video processing failed: {str(e)}"
        )
    finally:
        if input_path.exists():
            try:
                input_path.unlink()
            except Exception:
                pass