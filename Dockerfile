FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY webapp/requirements.txt /app/webapp/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/webapp/requirements.txt gunicorn

COPY . /app

EXPOSE 5001

CMD ["sh", "-c", "gunicorn --chdir /app/webapp --workers=1 --threads=4 --timeout=300 --bind 0.0.0.0:${PORT:-5001} app:app"]
