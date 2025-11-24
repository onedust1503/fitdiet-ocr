"""
RunPod Serverless Handler for YOLO Nutrition OCR
直接載入模型，不使用 subprocess
"""
import runpod
import base64
import tempfile
import os
import json
from ultralytics import YOLO
import cv2
import numpy as np
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
    # 清理文字
    text = text.replace(',', '.').replace('，', '.')
    # 尋找數字（包含小數）
    matches = re.findall(r'[\d]+\.?[\d]*', text)
    if matches:
        try:
            return float(matches[0])
        except:
            return None
    return None

def ocr_with_strategies(image, bbox):
    """使用多種前處理策略進行 OCR"""
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
    
    # 多種前處理策略
    strategies = {
        'gray': gray,
        'otsu': cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
        'adaptive': cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2),
    }
    
    # 嘗試每種策略
    results = []
    for name, processed in strategies.items():
        try:
            text = pytesseract.image_to_string(
                processed, 
                lang='chi_tra+eng',
                config='--psm 7'
            ).strip()
            value = extract_number(text)
            if value is not None:
                results.append((text, value))
        except:
            continue
    
    # 投票選出最常見的數值
    if results:
        values = [r[1] for r in results]
        # 簡單取中位數
        values.sort()
        best_value = values[len(values)//2]
        best_text = results[0][0]
        return best_text, best_value
    
    return None, None

def process_image(image_path):
    """處理圖片並返回 OCR 結果"""
    # 載入模型
    yolo = load_model()
    
    # 讀取圖片
    image = cv2.imread(image_path)
    if image is None:
        return {"error": "無法讀取圖片"}
    
    # YOLO 推論
    results = yolo(image, verbose=False)
    
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
            
        for box in result.boxes:
            cls_id = int(box.cls[0])
            bbox = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            
            if conf < 0.3:  # 信心度過低則跳過
                continue
            
            class_name = class_names.get(cls_id, f'unknown_{cls_id}')
            
            # OCR
            raw_text, value = ocr_with_strategies(image, bbox)
            
            if value is not None:
                items.append({
                    "label": class_name,
                    "value": value,
                    "raw_text": raw_text or "",
                    "confidence": conf
                })
    
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
        return {"error": str(e)}

# 預先載入模型（冷啟動時）
print("Initializing handler...")
load_model()
print("Handler ready!")

# RunPod 入口點
runpod.serverless.start({"handler": handler})