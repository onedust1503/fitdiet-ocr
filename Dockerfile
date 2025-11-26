FROM ultralytics/ultralytics:latest

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-chi-tra \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir runpod pytesseract
RUN pip install --no-cache-dir --force-reinstall "numpy<2.0" "opencv-python-headless<4.10"

COPY . .

CMD ["python", "-u", "handler.py"]
