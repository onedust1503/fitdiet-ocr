# Dockerfile for YOLO + EasyOCR (GPU Accelerated)
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

WORKDIR /app

# 設定時區
ENV TZ=Asia/Taipei
ENV DEBIAN_FRONTEND=noninteractive

# 安裝系統依賴
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    tzdata \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

# 升級 pip
RUN pip install --upgrade pip

# 安裝 Python 套件
RUN pip install --no-cache-dir ultralytics
RUN pip install --no-cache-dir easyocr
RUN pip install --no-cache-dir runpod
RUN pip install --no-cache-dir --force-reinstall "numpy<2.0" "opencv-python-headless<4.10"

# 預先下載 EasyOCR 模型
RUN python -c "import easyocr; easyocr.Reader(['ch_tra', 'ch_sim', 'en'], gpu=False)"

# 複製應用程式
COPY . .

# 設定入口點
CMD ["python", "-u", "handler.py"]
