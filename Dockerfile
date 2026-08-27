FROM python:3.11-slim

# System libraries OpenCV needs at import time (even with show_preview=False,
# cv2 itself still links against these on Debian-based images).
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Install Python deps first so this layer is cached unless requirements change.
# Split into separate RUN steps so each produces a smaller layer -- this makes
# pushes to registries far more resilient on unstable connections, since a
# dropped upload only costs you one smaller layer instead of one giant blob.
COPY requirements.txt .
RUN pip install --no-cache-dir --default-timeout=200 --retries 10 \
    torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir --default-timeout=200 --retries 10 \
    ultralytics
RUN pip install --no-cache-dir --default-timeout=200 --retries 10 \
    -r requirements.txt

# App code + model weights
COPY app/ ./app/
COPY models/ ./models/

# Runtime folders (uploads/output are written to at request time)
RUN mkdir -p uploads output videos

WORKDIR /workspace/app

EXPOSE 8000

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]