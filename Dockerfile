# RunPod Serverless Dockerfile for YOLO Nutrition OCR
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

WORKDIR /app

# 安裝系統依賴
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-chi-tra \
    tesseract-ocr-chi-sim \
    tesseract-ocr-eng \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

# 先升級 pip
RUN pip install --upgrade pip

# 安裝 Python 依賴（分開安裝避免衝突）
RUN pip install --no-cache-dir flask flask-cors Pillow
RUN pip install --no-cache-dir ultralytics
RUN pip install --no-cache-dir opencv-python-headless
RUN pip install --no-cache-dir pytesseract
RUN pip install --no-cache-dir runpod

# 複製應用程式檔案
COPY . .

# 設定環境變數
ENV PYTHONUNBUFFERED=1

# RunPod Serverless 入口點
CMD ["python", "-u", "handler.py"]
