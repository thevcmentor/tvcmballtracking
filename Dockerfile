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

# 2) install PyTorch + torchvision CPU wheels from the official index
#    This pair is compatible with Python 3.11 CPU wheels.
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu \
    torch==2.5.1 torchvision==0.20.1

# 3) install the rest (do NOT include torch/vision again in requirements)
RUN pip install -r requirements.txt

EXPOSE 10000
CMD ["uvicorn","app:app","--host","0.0.0.0","--port","10000"]
