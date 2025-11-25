"""
RunPod Serverless Handler for YOLO Nutrition OCR
優化版：減少 OCR 執行次數，提升速度
"""

# 先檢查 numpy 是否可用
try:
    import numpy as np
    print(f"✅ Numpy version: {np.__version__}")
except ImportError as e:
    print(f"❌ Numpy import failed: {e}")
    import sys
    sys.exit(1)

import runpod
import base64
import tempfile
import os
import json
from ultralytics import YOLO
import cv2
import pytesseract
import re

# 全域變數 - 模型只載入一次
model = None

def load_model():
    """載入 YOLO 模型（只執行一次）"""
    global model
    if model is None:
        print("Loading YOLO model...")
        model = YOLO("best.pt")
        print("Model loaded successfully!")
    return model

def extract_number(text):
    """從文字中提取數字"""
    if not text:
        return None
    text = text.replace(',', '.').replace('，', '.')
    matches = re.findall(r'[\d]+\.?[\d]*', text)
    if matches:
        try:
            return float(matches[0])
        except:
            return None
    return None

def ocr_with_single_strategy(image, bbox):
    """優化後的 OCR：只使用單一策略以提升速度"""
    x1, y1, x2, y2 = map(int, bbox)
    
    # 擴展邊界框
    pad = 5
    h, w = image.shape[:2]
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)
    
    roi = image[y1:y2, x1:x2]
    if roi.size == 0:
        return None, None
    
    # 轉灰階
    if len(roi.shape) == 3:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    else:
        gray = roi
    
    # 放大圖片有助於辨識小字
    scale = 2
    gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    
    # 二值化 - 只用 OTSU，最穩定
    _, processed = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    try:
        # 只執行一次 OCR
        text = pytesseract.image_to_string(
            processed,
            lang='chi_tra+eng',
            config='--psm 7'  # PSM 7 假設是單行文字
        ).strip()
        
        value = extract_number(text)
        if value is not None:
            return text, value
            
    except Exception as e:
        print(f"OCR Error: {e}")
        return None, None
    
    return None, None

def process_image(image_path):
    """處理圖片並返回 OCR 結果"""
    # 載入模型
    yolo = load_model()
    
    # 讀取圖片
    image = cv2.imread(image_path)
    if image is None:
        return {"error": "無法讀取圖片"}
    
    print(f"圖片大小: {image.shape}")
    
    # YOLO 推論 - 降低信心度閾值，加快 NMS
    results = yolo(
        image,
        verbose=False,
        conf=0.25,      # 提高信心度閾值，減少候選框
        iou=0.7,        # 提高 IOU 閾值，加快 NMS
        max_det=20      # 最多只偵測 20 個物件
    )
    
    # 類別名稱對應
    class_names = {
        0: 'serving_size',
        1: 'servings_per_package',
        2: 'calories',
        3: 'protein',
        4: 'fat',
        5: 'saturated',
        6: 'trans_fat',
        7: 'carbs',
        8: 'sugar',
        9: 'sodium'
    }
    
    items = []
    
    # 處理每個偵測結果
    for result in results:
        if result.boxes is None:
            continue
            
        print(f"偵測到 {len(result.boxes)} 個框框")
        
        for box in result.boxes:
            cls_id = int(box.cls[0])
            bbox = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            
            if conf < 0.3:
                continue
            
            class_name = class_names.get(cls_id, f'unknown_{cls_id}')
            
            # OCR - 只執行一次
            raw_text, value = ocr_with_single_strategy(image, bbox)
            
            if value is not None:
                items.append({
                    "label": class_name,
                    "value": value,
                    "raw_text": raw_text or "",
                    "confidence": conf
                })
                print(f"  {class_name}: {value} (conf: {conf:.2f})")
    
    print(f"成功辨識 {len(items)} 個欄位")
    
    return {
        "success": True,
        "items": items
    }

def handler(event):
    """RunPod Serverless handler"""
    try:
        # 取得輸入
        input_data = event.get("input", {})
        image_base64 = input_data.get("image")
        
        if not image_base64:
            return {"error": "No image provided"}
        
        # 解碼 base64 圖片
        image_bytes = base64.b64decode(image_base64)
        
        # 儲存為暫存檔
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(image_bytes)
            temp_path = f.name
        
        try:
            # 處理圖片
            result = process_image(temp_path)
            return result
        finally:
            # 清理暫存檔
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
    except Exception as e:
        print(f"Handler error: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

# 預先載入模型
print("=" * 50)
print("Initializing handler...")
print(f"Python version: {sys.version}")
print(f"Numpy available: {np.__version__}")
print("=" * 50)

load_model()
print("Handler ready!")

# RunPod 入口點
runpod.serverless.start({"handler": handler})