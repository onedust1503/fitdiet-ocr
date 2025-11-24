# 使用官方 Python 3.10 映像
FROM python:3.10-slim

# 安裝系統依賴：Tesseract OCR + 中文語言包 + OpenCV 需要的圖形庫
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-chi-tra \
    tesseract-ocr-eng \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 設定工作目錄
WORKDIR /app

# 複製 requirements.txt 並安裝 Python 套件
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 複製所有應用程式檔案（包含 best.pt）
COPY . .

# 暴露 8000 port
EXPOSE 8000

# 啟動 Flask 應用
CMD ["python", "app.py"]
