FROM python:3.11-slim

RUN apt-get update && \
    apt-get install -y ffmpeg git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Installer setuptools ANCIEN (avant qu'ils retirent pkg_resources de l'env isolé)
# + l'installer dans l'env isolé via PIP_NO_BUILD_ISOLATION
RUN pip install --no-cache-dir "setuptools==69.5.1" wheel pip

# Installer openai-whisper séparément avec --no-build-isolation
# pour qu'il utilise le setuptools du système, pas l'env isolé vide
RUN pip install --no-cache-dir --no-build-isolation openai-whisper==20231117

# Reste des dépendances normalement
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN mkdir -p uploads outputs static

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
