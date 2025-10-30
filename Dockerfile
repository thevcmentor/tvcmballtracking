FROM python:3.11-slim

# --- system deps for OpenCV + torchvision image backends ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgl1 ffmpeg \
    libjpeg62-turbo libpng16-16 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt /app/requirements.txt
COPY app.py /app/app.py

# --- Python deps ---
# 1) upgrade pip tooling
RUN pip install --upgrade pip wheel setuptools

# install torch/vision first
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu torch==2.5.0 torchvision==0.22.1

# then the rest
RUN pip install -r requirements.txt

ENV PORT=10000
CMD ["sh","-c","uvicorn app:app --host 0.0.0.0 --port ${PORT:-10000}"]
