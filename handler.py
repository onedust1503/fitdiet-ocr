"""
RunPod Serverless Handler for YOLO Nutrition OCR
穩定版
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

print("=" * 50)
print("Starting handler initialization...")
print(f"Numpy version: {np.__version__}")
print("=" * 50)

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

def ocr_with_multiple_strategies(image, bbox):
    """優化後的 OCR：嘗試多種策略找最佳結果"""
    try:
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
        
        # 放大圖片 - 提高到 3 倍
        scale = 3
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        
        # 嘗試 3 種策略
        strategies = []
        
        # 策略 1: OTSU 二值化
        _, binary1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        strategies.append(('otsu', binary1))
        
        # 策略 2: 反轉 OTSU（處理白底黑字）
        _, binary2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        strategies.append(('otsu_inv', binary2))
        
        # 策略 3: 自適應閾值
        adaptive = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        strategies.append(('adaptive', adaptive))
        
        # 對每種策略執行 OCR
        results = []
        for name, processed in strategies:
            # 去噪
            denoised = cv2.medianBlur(processed, 3)
            
            # 執行 OCR
            text = pytesseract.image_to_string(
                denoised,
                lang='eng',  # 只用英文，數字辨識更準確
                config='--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789.'
            ).strip()
            
            value = extract_number(text)
            if value is not None and value > 0:
                # 計算信心分數（數字長度越長，越可能正確）
                confidence = len(text.replace('.', '').replace(',', ''))
                results.append((value, confidence, text, name))
        
        if results:
            # 選擇信心分數最高的結果
            results.sort(key=lambda x: x[1], reverse=True)
            best_value, best_conf, best_text, best_strategy = results[0]
            print(f"  OCR Strategy: {best_strategy}, Text: '{best_text}', Value: {best_value}")
            return best_text, best_value
                
    except Exception as e:
        print(f"OCR Error: {e}")
        return None, None
    
    return None, None

def process_image(image_path):
    """處理圖片並返回 OCR 結果"""
    try:
        # 載入模型
        yolo = load_model()
        
        # 讀取圖片
        image = cv2.imread(image_path)
        if image is None:
            return {"error": "無法讀取圖片"}
        
        print(f"Image size: {image.shape}")
        
        # YOLO 推論
        results = yolo(
            image,
            verbose=False,
            conf=0.25,
            iou=0.7,
            max_det=20
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
                
            print(f"Detected {len(result.boxes)} boxes")
            
            for box in result.boxes:
                cls_id = int(box.cls[0])
                bbox = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                
                if conf < 0.3:
                    continue
                
                class_name = class_names.get(cls_id, f'unknown_{cls_id}')
                
                # OCR - 嘗試多種策略
                raw_text, value = ocr_with_multiple_strategies(image, bbox)
                
                if value is not None:
                    items.append({
                        "label": class_name,
                        "value": value,
                        "raw_text": raw_text or "",
                        "confidence": conf
                    })
                    print(f"  {class_name}: {value} (conf: {conf:.2f})")
        
        print(f"Successfully recognized {len(items)} fields")
        
        return {
            "success": True,
            "items": items
        }
    except Exception as e:
        print(f"Process error: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

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
print("Initializing handler...")
load_model()
print("Handler ready!")

# RunPod 入口點
runpod.serverless.start({"handler": handler})