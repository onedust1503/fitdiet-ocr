# RunPod Serverless Dockerfile for YOLO Nutrition OCR
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

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

# 複製 requirements 並安裝
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir runpod

# 複製應用程式檔案
COPY . .

# 設定環境變數
ENV PYTHONUNBUFFERED=1

# RunPod Serverless 入口點
CMD ["python", "-u", "handler.py"]
