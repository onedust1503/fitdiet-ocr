"""
RunPod Serverless Handler for YOLO Nutrition OCR
v5 - 使用 EasyOCR (GPU 加速) - 又快又準
"""
import runpod
import base64
import tempfile
import os
import cv2
import numpy as np
import re
from ultralytics import YOLO
import easyocr

print("=" * 50)
print("Handler v5 - EasyOCR GPU Accelerated")
print("=" * 50)

# 全域變數
yolo_model = None
ocr_reader = None

def load_models():
    """載入 YOLO 和 EasyOCR 模型"""
    global yolo_model, ocr_reader
    
    if yolo_model is None:
        print("Loading YOLO model...")
        yolo_model = YOLO("best.pt")
        print("YOLO loaded!")
    
    if ocr_reader is None:
        print("Loading EasyOCR model (GPU)...")
        # 支援中文繁體、簡體、英文
        ocr_reader = easyocr.Reader(['ch_tra', 'ch_sim', 'en'], gpu=True)
        print("EasyOCR loaded!")
    
    return yolo_model, ocr_reader

def extract_number(text):
    """從文字中提取數字"""
    if not text or text.strip() == '':
        return None
    
    text = text.strip()
    text = text.replace(',', '.').replace('，', '.')
    text = text.replace('O', '0').replace('o', '0')
    text = text.replace('l', '1').replace('I', '1').replace('|', '1')
    text = text.replace(' ', '')
    
    # 找數字
    matches = re.findall(r'(\d+\.?\d*)', text)
    if matches:
        try:
            return float(matches[0])
        except:
            pass
    return None

def correct_value(label, value):
    """修正異常值"""
    if value is None:
        return None
    
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
    }
    
    max_val = max_values.get(label, 10000)
    
    if value > max_val:
        str_val = str(int(value))
        for i in range(1, min(3, len(str_val))):
            try:
                new_val = float(f"{str_val[:i]}.{str_val[i:]}")
                if new_val <= max_val:
                    return round(new_val, 1)
            except:
                continue
    
    return round(value, 1) if value < 1000 else value

def ocr_with_easyocr(image, bbox, reader):
    """使用 EasyOCR 進行文字辨識 - GPU 加速"""
    x1, y1, x2, y2 = map(int, bbox)
    
    # 擴展邊界框
    h, w = image.shape[:2]
    pad = 15
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)
    
    roi = image[y1:y2, x1:x2]
    if roi.size == 0:
        return None, None
    
    # 放大圖片
    scale = 2.0
    roi = cv2.resize(roi, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    
    # EasyOCR 辨識
    try:
        results = reader.readtext(roi, detail=1)
        
        # 收集所有辨識結果
        all_text = []
        for (box, text, conf) in results:
            if conf > 0.3:  # 信心度閾值
                all_text.append(text)
        
        # 合併文字
        combined_text = ' '.join(all_text)
        
        if combined_text:
            value = extract_number(combined_text)
            if value is not None and value > 0:
                return combined_text, value
                
    except Exception as e:
        print(f"    EasyOCR error: {e}")
    
    return None, None

def process_image(image_path):
    """處理圖片"""
    import time
    start_time = time.time()
    
    try:
        print("Started.")
        yolo, reader = load_models()
        
        # 讀取圖片
        image = cv2.imread(image_path)
        if image is None:
            return {"error": "無法讀取圖片", "success": False, "items": []}
        
        print(f"Image size: {image.shape}")
        
        # YOLO 推論
        results = yolo(
            image,
            verbose=False,
            conf=0.20,
            iou=0.5,
            max_det=15
        )
        
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
            
            boxes_data = []
            for box in result.boxes:
                cls_id = int(box.cls[0])
                bbox = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                boxes_data.append((cls_id, bbox, conf))
            
            boxes_data.sort(key=lambda x: x[2], reverse=True)
            print(f"Detected {len(boxes_data)} boxes")
            
            for cls_id, bbox, conf in boxes_data:
                class_name = class_names.get(cls_id, f'unknown_{cls_id}')
                
                if class_name in processed_labels:
                    continue
                
                print(f"  Processing: {class_name} (conf: {conf:.2f})")
                
                # EasyOCR
                raw_text, value = ocr_with_easyocr(image, bbox, reader)
                
                if value is not None:
                    corrected = correct_value(class_name, value)
                    if corrected is not None:
                        items.append({
                            "label": class_name,
                            "value": corrected,
                            "confidence": round(conf, 2)
                        })
                        processed_labels.add(class_name)
                        print(f"    ✓ {class_name}: {corrected} (text: '{raw_text}')")
                else:
                    print(f"    ✗ {class_name}: OCR failed")
        
        elapsed = time.time() - start_time
        print(f"Finished. Recognized {len(items)} fields in {elapsed:.1f}s")
        
        return {
            "success": True,
            "items": items,
            "processing_time": round(elapsed, 2)
        }
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e), "success": False, "items": []}

def handler(event):
    """RunPod handler"""
    try:
        input_data = event.get("input", {})
        image_base64 = input_data.get("image")
        
        if not image_base64:
            return {"error": "No image", "success": False, "items": []}
        
        image_bytes = base64.b64decode(image_base64)
        
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(image_bytes)
            temp_path = f.name
        
        try:
            return process_image(temp_path)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
    except Exception as e:
        print(f"Handler error: {e}")
        return {"error": str(e), "success": False, "items": []}

print("Loading models on startup...")
load_models()
print("Handler v5 ready!")

runpod.serverless.start({"handler": handler})