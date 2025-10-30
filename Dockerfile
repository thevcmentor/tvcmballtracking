FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgl1 ffmpeg libjpeg62-turbo libpng16-16 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt /app/requirements.txt
COPY app.py /app/app.py

RUN pip install --upgrade pip wheel setuptools

# Torch + torchvision CPU wheels for Py3.10
RUN pip install --no-cache-dir \
  --index-url https://download.pytorch.org/whl/cpu \
  torch==2.4.0+cpu torchvision==0.19.0+cpu

RUN pip install --no-cache-dir -r requirements.txt

ENV PORT=10000
CMD ["sh","-c","uvicorn app:app --host 0.0.0.0 --port ${PORT:-10000}"]
