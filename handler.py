"""
RunPod Serverless Handler for YOLO Nutrition OCR
v6 - 極度精簡版 (快速穩定)
"""
import runpod
import base64
import tempfile
import os
import cv2
import numpy as np
import pytesseract
import re
from ultralytics import YOLO

print("Handler v6 - Fast & Stable")

model = None

def load_model():
    global model
    if model is None:
        model = YOLO("best.pt")
    return model

def extract_number(text):
    if not text:
        return None
    text = text.strip().replace(',', '.').replace('O', '0').replace('o', '0')
    text = text.replace('l', '1').replace('I', '1').replace(' ', '')
    matches = re.findall(r'(\d+\.?\d*)', text)
    if matches:
        try:
            return float(matches[0])
        except:
            pass
    return None

def correct_value(label, value):
    if value is None:
        return None
    max_vals = {'calories': 2000, 'protein': 100, 'fat': 100, 'saturated': 50, 'trans_fat': 10, 'carbs': 200, 'sugar': 100, 'sodium': 5000}
    max_val = max_vals.get(label, 10000)
    if value > max_val:
        s = str(int(value))
        for i in range(1, min(3, len(s))):
            try:
                new_val = float(f"{s[:i]}.{s[i:]}")
                if new_val <= max_val:
                    return round(new_val, 1)
            except:
                continue
    return round(value, 1) if value < 1000 else value

def ocr_fast(image, bbox):
    x1, y1, x2, y2 = map(int, bbox)
    h, w = image.shape[:2]
    pad = 15
    x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
    x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
    roi = image[y1:y2, x1:x2]
    if roi.size == 0:
        return None, None
    if len(roi.shape) == 3:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    else:
        gray = roi
    gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    config = '--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789.'
    try:
        text = pytesseract.image_to_string(binary, lang='eng', config=config).strip()
        if text:
            value = extract_number(text)
            if value and value > 0:
                return text, value
    except:
        pass
    _, binary_inv = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    try:
        text = pytesseract.image_to_string(binary_inv, lang='eng', config=config).strip()
        if text:
            value = extract_number(text)
            if value and value > 0:
                return text, value
    except:
        pass
    return None, None

def process_image(image_path):
    import time
    start = time.time()
    try:
        yolo = load_model()
        image = cv2.imread(image_path)
        if image is None:
            return {"error": "Cannot read image", "success": False, "items": []}
        results = yolo(image, verbose=False, conf=0.20, iou=0.5, max_det=12)
        names = {0: 'serving_size', 1: 'servings_per_package', 2: 'calories', 3: 'protein', 4: 'fat', 5: 'saturated', 6: 'trans_fat', 7: 'carbs', 8: 'sugar', 9: 'sodium'}
        items = []
        done = set()
        for result in results:
            if result.boxes is None:
                continue
            boxes = [(int(b.cls[0]), b.xyxy[0].tolist(), float(b.conf[0])) for b in result.boxes]
            boxes.sort(key=lambda x: x[2], reverse=True)
            print(f"Detected {len(boxes)} boxes")
            for cls_id, bbox, conf in boxes:
                name = names.get(cls_id, f'unknown_{cls_id}')
                if name in done:
                    continue
                raw, value = ocr_fast(image, bbox)
                if value:
                    corrected = correct_value(name, value)
                    if corrected:
                        items.append({"label": name, "value": corrected, "confidence": round(conf, 2)})
                        done.add(name)
                        print(f"  ✓ {name}: {corrected}")
                else:
                    print(f"  ✗ {name}: failed")
        elapsed = time.time() - start
        print(f"Done. {len(items)} fields in {elapsed:.1f}s")
        return {"success": True, "items": items, "processing_time": round(elapsed, 2)}
    except Exception as e:
        print(f"Error: {e}")
        return {"error": str(e), "success": False, "items": []}

def handler(event):
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
        return {"error": str(e), "success": False, "items": []}

load_model()
runpod.serverless.start({"handler": handler})
