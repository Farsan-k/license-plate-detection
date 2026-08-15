FROM python:3.11-slim

# System libraries OpenCV needs at import time (even with show_preview=False,
# cv2 itself still links against these on Debian-based images).
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Install Python deps first so this layer is cached unless requirements change
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code + model weights
COPY app/ ./app/
COPY models/ ./models/

# Runtime folders (uploads/output are written to at request time)
RUN mkdir -p uploads output videos

WORKDIR /workspace/app

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]