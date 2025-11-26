"""
RunPod Serverless Handler for YOLO Nutrition OCR
v3 - 改善 OCR 成功率
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
    """從文字中提取數字"""
    if not text:
        return None
    
    # 清理文字
    text = text.replace(',', '.').replace('，', '.').replace(' ', '')
    text = text.replace('O', '0').replace('o', '0')  # O 誤認為 0
    text = text.replace('l', '1').replace('I', '1')  # l/I 誤認為 1
    text = text.strip()
    
    # 找數字
    matches = re.findall(r'\d+\.?\d*', text)
    
    if not matches:
        return None
    
    try:
        num_str = matches[0]
        value = float(num_str)
        return value
    except:
        return None

def correct_value(label, value):
    """修正異常值"""
    if value is None:
        return None
    
    # 合理範圍
    max_values = {
        'calories': 2000,
        'protein': 100,
        'fat': 100,
        'saturated': 50,
        'trans_fat': 10,
        'carbs': 200,
        'sugar': 100,
        'sodium': 5000,
        'serving_size': 1000,
        'servings_per_package': 50,
    }
    
    max_val = max_values.get(label, 10000)
    
    # 如果值太大，嘗試插入小數點
    if value > max_val:
        str_val = str(int(value))
        
        if len(str_val) >= 5:
            opt1 = float(f"{str_val[:2]}.{str_val[2:]}")
            if opt1 <= max_val:
                return round(opt1, 1)
            opt2 = float(f"{str_val[:3]}.{str_val[3:]}")
            if opt2 <= max_val:
                return round(opt2, 1)
        elif len(str_val) == 4:
            opt1 = float(f"{str_val[:2]}.{str_val[2:]}")
            if opt1 <= max_val:
                return round(opt1, 1)
        elif len(str_val) == 3:
            opt1 = float(f"{str_val[:1]}.{str_val[1:]}")
            if opt1 <= max_val:
                return round(opt1, 1)
    
    return round(value, 1) if value < 1000 else value

def ocr_with_multiple_strategies(image, bbox):
    """使用多種 OCR 策略，提高成功率"""
    
    x1, y1, x2, y2 = map(int, bbox)
    
    # 擴展邊界框 - 擴大一點
    h, w = image.shape[:2]
    pad = 10  # 增加 padding
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
        gray = roi.copy()
    
    # 放大圖片 - 增加到 2 倍
    scale = 2.0
    gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    
    # 多種預處理策略
    strategies = []
    
    # 策略 1: OTSU 二值化
    _, binary1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    strategies.append(('otsu', binary1))
    
    # 策略 2: OTSU 反轉
    _, binary2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    strategies.append(('otsu_inv', binary2))
    
    # 策略 3: 自適應閾值
    binary3 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 11, 2)
    strategies.append(('adaptive', binary3))
    
    # 策略 4: 固定閾值
    _, binary4 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    strategies.append(('fixed', binary4))
    
    # 策略 5: 原圖直接 OCR
    strategies.append(('gray', gray))
    
    # 多種 PSM 模式
    psm_modes = [7, 8, 6, 13]  # 7=單行, 8=單詞, 6=統一塊, 13=原始行
    
    # 嘗試所有組合
    for strategy_name, processed in strategies:
        for psm in psm_modes:
            try:
                # OCR 配置 - 允許更多字符
                config = f'--psm {psm} --oem 3 -c tessedit_char_whitelist=0123456789.'
                
                text = pytesseract.image_to_string(processed, lang='eng', config=config).strip()
                
                if text:
                    value = extract_number(text)
                    if value is not None and value > 0:
                        print(f"    OCR Strategy: {strategy_name}, PSM: {psm}, Text: '{text}', Value: {value}")
                        return text, value
                        
            except Exception as e:
                continue
    
    # 最後嘗試：不限制字符
    for strategy_name, processed in strategies[:2]:  # 只用前兩種策略
        try:
            config = '--psm 7 --oem 3'
            text = pytesseract.image_to_string(processed, lang='eng', config=config).strip()
            
            if text:
                value = extract_number(text)
                if value is not None and value > 0:
                    print(f"    OCR Strategy: {strategy_name} (no whitelist), Text: '{text}', Value: {value}")
                    return text, value
        except:
            continue
    
    return None, None

def process_image(image_path):
    """處理圖片並返回 OCR 結果"""
    import time
    start_time = time.time()
    MAX_PROCESSING_TIME = 60
    
    try:
        print("Started.")
        
        # 載入模型
        yolo = load_model()
        
        # 讀取圖片
        image = cv2.imread(image_path)
        if image is None:
            return {"error": "無法讀取圖片", "success": False, "items": []}
        
        print(f"Image size: {image.shape}")
        
        # YOLO 推論 - 降低 conf 閾值以偵測更多
        results = yolo(
            image,
            verbose=False,
            conf=0.20,      # 降低到 0.20
            iou=0.5,
            max_det=15
        )
        
        # 類別名稱
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
        processed_labels = set()
        
        for result in results:
            if result.boxes is None:
                continue
            
            # 收集所有框並排序
            boxes_with_conf = []
            for box in result.boxes:
                cls_id = int(box.cls[0])
                bbox = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                boxes_with_conf.append((cls_id, bbox, conf))
            
            # 按信心度排序
            boxes_with_conf.sort(key=lambda x: x[2], reverse=True)
            
            print(f"Detected {len(boxes_with_conf)} boxes (sorted by confidence)")
            
            for cls_id, bbox, conf in boxes_with_conf:
                # 超時檢查
                if time.time() - start_time > MAX_PROCESSING_TIME:
                    print(f"⚠️ Timeout after {MAX_PROCESSING_TIME}s")
                    break
                
                class_name = class_names.get(cls_id, f'unknown_{cls_id}')
                
                # 跳過已處理的標籤
                if class_name in processed_labels:
                    continue
                
                print(f"  Box: {class_name} (conf: {conf:.2f})")
                
                # OCR - 使用多種策略
                raw_text, value = ocr_with_multiple_strategies(image, bbox)
                
                if value is not None:
                    # 修正異常值
                    corrected_value = correct_value(class_name, value)
                    
                    if corrected_value is not None:
                        items.append({
                            "label": class_name,
                            "value": corrected_value,
                            "raw_text": raw_text or "",
                            "confidence": round(conf, 2)
                        })
                        processed_labels.add(class_name)
                        print(f"    ✓ {class_name}: {corrected_value} (conf: {conf:.2f})")
                    else:
                        print(f"    ✗ {class_name}: Value correction failed")
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
        return {"error": str(e), "success": False, "items": []}

def handler(event):
    """RunPod Serverless handler"""
    try:
        input_data = event.get("input", {})
        image_base64 = input_data.get("image")
        
        if not image_base64:
            return {"error": "No image provided", "success": False, "items": []}
        
        # 解碼 base64
        image_bytes = base64.b64decode(image_base64)
        
        # 暫存檔
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(image_bytes)
            temp_path = f.name
        
        try:
            result = process_image(temp_path)
            return result
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
    except Exception as e:
        print(f"Handler error: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e), "success": False, "items": []}

# 預先載入模型
print("Initializing handler...")
load_model()
print("Handler ready!")

# RunPod 入口點
runpod.serverless.start({"handler": handler})