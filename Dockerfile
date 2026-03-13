FROM python:3.11-slim

# Outils système (FFmpeg requis pour le traitement vidéo)
RUN apt-get update && \
    apt-get install -y ffmpeg ffprobe git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ⚠️ Étape critique : setuptools + wheel AVANT requirements.txt
# openai-whisper utilise setup.py qui a besoin de pkg_resources (fourni par setuptools)
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Code de l'application
COPY . .

# Dossiers requis
RUN mkdir -p uploads outputs static

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
