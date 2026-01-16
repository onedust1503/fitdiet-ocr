# FitDiet OCR API

基於 YOLOv8 + Tesseract 的營養標籤辨識 API，用於食尚健身 App。

## 功能

- 使用 YOLO 模型偵測營養標籤上的 10 個欄位
- 多策略 OCR 前處理與投票機制
- 自動能量驗算（蛋白質、脂肪、碳水化合物）
- RESTful API 介面

## 影像辨識環境建置
conda create -n yolo310 python=3.10 -y
conda activate yolo310

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install ultralytics opencv-python pillow pytesseract

conda activate yolo310

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

pip install ultralytics opencv-python pillow pytesseract

python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"

python -c "from ultralytics import YOLO; model = YOLO('yolov8n.pt'); results = model('https://ultralytics.com/images/bus.jpg'); print(results[0].boxes.xyxy)"

python -c "import pytesseract; print(pytesseract.get_tesseract_version())"
