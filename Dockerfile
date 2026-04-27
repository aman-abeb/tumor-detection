FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ── Step 1: Install CPU-only PyTorch first (saves ~2 GB vs CUDA build) ────────
RUN pip install --upgrade pip && \
    pip install --no-cache-dir \
        torch==2.3.1+cpu \
        torchvision==0.18.1+cpu \
        --index-url https://download.pytorch.org/whl/cpu && \
    rm -rf /root/.cache/pip

# ── Step 2: Install remaining dependencies ────────────────────────────────────
COPY webapp/requirements.txt /app/webapp/requirements.txt
RUN pip install --no-cache-dir \
        flask \
        numpy \
        opencv-python-headless \
        matplotlib \
        pillow \
        ultralytics \
        gunicorn \
    && rm -rf /root/.cache/pip

# ── Step 3: Copy application code ─────────────────────────────────────────────
COPY . /app

EXPOSE 5001

CMD ["sh", "-c", "gunicorn --chdir /app/webapp --workers=1 --threads=4 --timeout=300 --bind 0.0.0.0:${PORT:-5001} app:app"]
