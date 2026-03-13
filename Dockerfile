FROM python:3.11-slim

# Outils système (ffprobe est inclus dans ffmpeg, pas un paquet séparé)
RUN apt-get update && \
    apt-get install -y ffmpeg git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ⚠️ setuptools + wheel AVANT requirements.txt
# openai-whisper utilise setup.py qui a besoin de pkg_resources
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
