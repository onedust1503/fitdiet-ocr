"""
RunPod Serverless Handler for YOLO Nutrition OCR
"""
import runpod
import base64
import tempfile
import os
import subprocess
import json

def handler(event):
    """
    RunPod Serverless handler function
    
    Input format:
    {
        "input": {
            "image": "base64_encoded_image_string"
        }
    }
    """
    try:
        # 1. 取得輸入
        input_data = event.get("input", {})
        image_base64 = input_data.get("image")
        
        if not image_base64:
            return {"error": "No image provided"}
        
        # 2. 解碼 base64 圖片並儲存為暫存檔
        image_bytes = base64.b64decode(image_base64)
        
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(image_bytes)
            temp_image_path = f.name
        
        # 3. 建立輸出檔案路徑
        temp_output_path = tempfile.mktemp(suffix=".json")
        
        try:
            # 4. 呼叫 infer_ocr_tesseract.py
            result = subprocess.run(
                [
                    "python", "infer_ocr_tesseract.py",
                    "--weights", "best.pt",
                    "--source", temp_image_path,
                    "--out", temp_output_path
                ],
                capture_output=True,
                text=True,
                timeout=120  # 2 分鐘超時
            )
            
            # 5. 讀取結果
            if os.path.exists(temp_output_path):
                with open(temp_output_path, "r", encoding="utf-8") as f:
                    ocr_result = json.load(f)
                
                # 6. 格式化輸出
                items = []
                fields = ocr_result.get("fields", {})
                
                for field_name, field_data in fields.items():
                    if field_data.get("ok") and field_data.get("value") is not None:
                        items.append({
                            "label": field_name,
                            "value": field_data["value"],
                            "raw_text": field_data.get("raw", "")
                        })
                
                return {
                    "success": True,
                    "items": items,
                    "energy_check": ocr_result.get("energy_check", {})
                }
            else:
                return {
                    "error": "OCR processing failed",
                    "stderr": result.stderr
                }
                
        finally:
            # 清理暫存檔
            if os.path.exists(temp_image_path):
                os.remove(temp_image_path)
            if os.path.exists(temp_output_path):
                os.remove(temp_output_path)
                
    except Exception as e:
        return {"error": str(e)}


# RunPod Serverless 入口點
runpod.serverless.start({"handler": handler})
