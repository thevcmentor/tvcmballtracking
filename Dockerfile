FROM python:3.11-slim

# --- system deps for opencv ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgl1 ffmpeg && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt /app/requirements.txt

# --- Install Python deps ---
RUN pip install --upgrade pip wheel setuptools && \
    pip install --extra-index-url https://download.pytorch.org/whl/cpu torch==2.5.0 torchvision==0.21.0 && \
    pip install -r requirements.txt

COPY app.py /app/app.py

EXPOSE 10000
CMD ["uvicorn","app:app","--host","0.0.0.0","--port","10000"]
