# Dockerfile for YOLO + EasyOCR (GPU Accelerated)
# v5.1 - 移除預下載模型避免 build 超時
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

WORKDIR /app

# 設定時區和環境
ENV TZ=Asia/Taipei
ENV DEBIAN_FRONTEND=noninteractive
ENV EASYOCR_MODULE_PATH=/app/.EasyOCR

# 安裝系統依賴
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    tzdata \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 安裝 Python 套件
RUN pip install --upgrade pip && \
    pip install --no-cache-dir ultralytics && \
    pip install --no-cache-dir easyocr && \
    pip install --no-cache-dir runpod && \
    pip install --no-cache-dir --force-reinstall "numpy<2.0" "opencv-python-headless<4.10"

# 複製應用程式
COPY . .

# 設定入口點
CMD ["python", "-u", "handler.py"]
