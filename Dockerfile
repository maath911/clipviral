FROM python:3.11-slim

# Outils système (ffprobe est inclus dans le paquet ffmpeg, pas séparé)
RUN apt-get update && \
    apt-get install -y ffmpeg git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# setuptools + wheel dans l'env principal
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# openai-whisper utilise setup.py ancien style → pkg_resources manque dans l'env isolé de build
# --no-build-isolation force pip à utiliser l'env principal où setuptools est disponible
COPY requirements.txt .
RUN pip install --no-cache-dir --no-build-isolation -r requirements.txt

COPY . .
RUN mkdir -p uploads outputs static

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
