"""
RunPod Serverless Handler for YOLO Nutrition OCR
優化版 - 快速且準確
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
    """載入 YOLO 模型(只執行一次)"""
    global model
    if model is None:
        print("Loading YOLO model...")
        model = YOLO("best.pt")
        print("Model loaded successfully!")
    return model

def extract_number(text):
    """從文字中提取數字,智能處理小數點"""
    if not text:
        return None
    
    # 移除常見的干擾字符
    text = text.replace(',', '.').replace('，', '.').replace(' ', '').strip()
    
    # 找出所有數字和小數點
    matches = re.findall(r'\d+\.?\d*', text)
    
    if not matches:
        return None
    
    try:
        # 取第一個匹配的數字
        num_str = matches[0]
        value = float(num_str)
        
        # 智能修正:如果數字太大(可能是小數點錯誤)
        if value > 1000 and '.' not in num_str:
            str_value = str(int(value))
            
            # 如果是 5 位數,嘗試 XX.XXX 或 XXX.XX
            if len(str_value) == 5:
                option1 = float(str_value[:2] + '.' + str_value[2:])
                if option1 < 100:
                    return option1
                option2 = float(str_value[:3] + '.' + str_value[3:])
                if option2 < 1000:
                    return option2
            
            # 如果是 4 位數,嘗試 XX.XX
            elif len(str_value) == 4:
                option1 = float(str_value[:2] + '.' + str_value[2:])
                if option1 < 100:
                    return option1
            
            # 如果是 3 位數,嘗試 X.XX
            elif len(str_value) == 3:
                option1 = float(str_value[0] + '.' + str_value[1:])
                if option1 < 10:
                    return option1
        
        return value
        
    except:
        return None

def ocr_single_fast_strategy(image, bbox, timeout=3):
    """快速 OCR:單一策略"""
    import time
    ocr_start = time.time()
    
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
        
        # 檢查超時
        if time.time() - ocr_start > timeout:
            return None, None
        
        # 轉灰階
        if len(roi.shape) == 3:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi
        
        # 放大圖片
        scale = 1.5
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        
        # OTSU 二值化
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # OCR - 只允許數字
        text = pytesseract.image_to_string(
            binary,
            lang='eng',
            config='--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789.'
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
    import time
    start_time = time.time()
    MAX_PROCESSING_TIME = 60  # 改為 60 秒(避免超時)
    
    try:
        print("Started.")
        
        # 載入模型
        yolo = load_model()
        
        # 讀取圖片
        image = cv2.imread(image_path)
        if image is None:
            return {"error": "無法讀取圖片"}
        
        print(f"Image size: {image.shape}")
        
        # ⭐ YOLO 推論 - 關鍵參數優化
        results = yolo(
            image,
            verbose=False,
            conf=0.25,      # ⭐ 改回 0.25 (平衡準確度和速度)
            iou=0.5,        # ⭐ 降到 0.5 (減少重複框)
            max_det=15      # ⭐ 降到 15 (避免太多框)
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
        processed_labels = set()  # ⭐ 追蹤已處理的標籤(避免重複)
        
        # 處理每個偵測結果
        for result in results:
            if result.boxes is None:
                continue
                
            boxes_with_conf = []
            for box in result.boxes:
                cls_id = int(box.cls[0])
                bbox = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                boxes_with_conf.append((cls_id, bbox, conf))
            
            # ⭐ 按信心度排序(優先處理高信心度的框)
            boxes_with_conf.sort(key=lambda x: x[2], reverse=True)
            
            print(f"Detected {len(boxes_with_conf)} boxes (sorted by confidence)")
            
            for cls_id, bbox, conf in boxes_with_conf:
                # 檢查超時
                if time.time() - start_time > MAX_PROCESSING_TIME:
                    print(f"⚠️ Timeout after {MAX_PROCESSING_TIME}s, returning partial results")
                    break
                
                class_name = class_names.get(cls_id, f'unknown_{cls_id}')
                
                # ⭐ 如果這個標籤已經處理過,跳過(取最高信心度的那個)
                if class_name in processed_labels:
                    print(f"  Skipped {class_name} (already processed)")
                    continue
                
                print(f"  Box: {class_name} (conf: {conf:.2f})")
                
                # OCR - 快速單一策略
                raw_text, value = ocr_single_fast_strategy(image, bbox, timeout=3)
                
                if value is not None:
                    items.append({
                        "label": class_name,
                        "value": value,
                        "raw_text": raw_text or "",
                        "confidence": conf
                    })
                    processed_labels.add(class_name)  # ⭐ 標記為已處理
                    print(f"    ✓ {class_name}: {value} (conf: {conf:.2f})")
                else:
                    print(f"    ✗ {class_name}: OCR failed (raw_text: '{raw_text}')")
        
        elapsed = time.time() - start_time
        print(f"Successfully recognized {len(items)} fields in {elapsed:.2f}s")
        print("Finished.")
        
        return {
            "success": True,
            "items": items,
            "processing_time": round(elapsed, 2)
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