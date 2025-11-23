import os, json, subprocess, tempfile
from typing import List, Dict, Any, Tuple
from PIL import Image

class InferenceError(Exception):
    """推論相關錯誤"""
    pass

class CliAdapter:
    """
    專門用命令列呼叫 infer_ocr_tesseract.py
    完全不修改原本的 infer_ocr_tesseract.py
    """
    def __init__(self):
        self.prog = os.getenv("CLI_PROG", "python")
        self.script = os.getenv("CLI_SCRIPT", "infer_ocr_tesseract.py")

    def infer(self, pil_img: Image.Image, engine: str = "tesseract",
              return_boxes: bool = False, **kwargs) -> Tuple[List[Dict[str,Any]], List[Dict[str,Any]]]:
        # 1. 先把 PIL 圖存成暫存 jpg
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=True) as tf:
            pil_img.save(tf.name, format="JPEG", quality=90)

            # 2. 組 CLI 指令（跟你平常在 cmd 跑的很像）
            weights = os.getenv("YOLO_WEIGHTS", "best.pt")
            out_path = "result.json"  # infer_ocr_tesseract 會寫這個 JSON
            cmd = [
                self.prog,
                self.script,
                "--weights", weights,
                "--source", tf.name,
                "--out", out_path,
            ]

            try:
                subprocess.run(cmd, check=True)
                with open(out_path, encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                raise InferenceError(f"CLI 執行失敗：{e}")

            # 3. 從 result.json 讀出 fields / boxes，轉成 API 要的格式
            fields = data.get("fields", {})
            items: List[Dict[str, Any]] = [
                {
                    "label": k,
                    "value": v.get("value"),
                    "raw_text": v.get("raw"),
                    "conf": None,  # 原本 JSON 裡沒有信心度，就先給 None
                }
                for k, v in fields.items()
            ]
            boxes = data.get("boxes", [])
            return items, boxes if return_boxes else []

def get_infer_adapter():
    # 直接回傳 CLI adapter（我們就是走 CLI 模式）
    return CliAdapter()
