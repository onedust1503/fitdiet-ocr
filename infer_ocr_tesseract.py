import cv2
import pytesseract
from ultralytics import YOLO
import json, argparse, os, re, shutil
from collections import Counter

# --- 多策略前處理 ---
def preprocess_strategies(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    strategies = {}

    strategies["gray"] = gray

    # OTSU
    _, th1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    strategies["otsu"] = th1

    # Adaptive
    th2 = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 2
    )
    strategies["adaptive"] = th2

    # Gaussian + OTSU
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    strategies["blur_otsu"] = th3

    # Morphology Closing
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    th4 = cv2.morphologyEx(th1, cv2.MORPH_CLOSE, kernel, iterations=1)
    strategies["closing"] = th4

    # Resize (放大2x)
    th5 = cv2.resize(th1, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    strategies["resize"] = th5

    # Median Blur
    median = cv2.medianBlur(gray, 3)
    _, th6 = cv2.threshold(median, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    strategies["median"] = th6

    # Morphology Opening
    th7 = cv2.morphologyEx(th1, cv2.MORPH_OPEN, kernel, iterations=1)
    strategies["opening"] = th7

    # CLAHE + OTSU
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    _, th8 = cv2.threshold(clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    strategies["clahe_otsu"] = th8

    # Tophat + OTSU
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    _, th9 = cv2.threshold(tophat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    strategies["tophat_otsu"] = th9

    # Blackhat + OTSU
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, th10 = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    strategies["blackhat_otsu"] = th10

    return strategies

# --- 數字正規化 ---
def normalize_text(text: str):
    text = text.strip().replace(" ", "")
    text = text.replace("大卡", "kcal").replace("卡路里", "kcal")
    text = text.replace("公克", "g").replace("克", "g")
    text = text.replace("毫克", "mg").replace("公升", "L")
    text = re.sub(r"[^0-9a-zA-Z\u4e00-\u9fff\.\-]", "", text)
    return text

# --- sodium 專屬清理 ---
def clean_sodium_text(text: str):
    match = re.search(r"(\d+(\.\d+)?)", text)
    if match:
        return match.group(1)  # 只取第一個數字，後面一律丟掉
    return text

# --- 數值提取 ---
def extract_value(text: str, cls_name: str = None):
    match = re.search(r"(\d+(\.\d+)?)", text)
    if match:
        val = float(match.group(1))
        return float(f"{val:.1f}")

    # 特別規則：trans_fat 出現但沒數字 → 預設 0.0
    if cls_name == "trans_fat" and "反式脂肪" in text:
        return 0.0

    return None

# --- 投票機制 ---
def select_best_vote(votes):
    vals = [v["value"] for v in votes if v["value"] is not None]
    if not vals:
        return None, []
    counter = Counter(vals)
    best_val, _ = counter.most_common(1)[0]
    selected = [v["strategy"] for v in votes if v["value"] == best_val]
    return best_val, selected

# --- 能量檢查 ---
def check_energy(fields):
    try:
        cal = fields["calories"]["value"]
        p = fields["protein"]["value"]
        f = fields["fat"]["value"]
        c = fields["carbs"]["value"]

        if None in [cal, p, f, c]:
            return {"ok": False, "expected": None, "error": "缺少必要欄位"}

        expected = round(p * 4 + c * 4 + f * 9, 1)
        ok = abs(cal - expected) / expected < 0.25
        if ok:
            return {"ok": True, "expected": expected}
        else:
            return {"ok": False, "expected": expected, "error": f"熱量不符 (cal={cal}, expected≈{expected})"}
    except Exception as e:
        return {"ok": False, "expected": None, "error": str(e)}

# --- sodium 範圍檢查 ---
def check_sodium(val):
    if val is None:
        return False
    return 0 <= val <= 2000

# --- 主流程 ---
def run(weights, source, out, pad=3, debug=False):
    if os.path.exists("debug"):
        shutil.rmtree("debug")
    os.makedirs("debug", exist_ok=True)

    model = YOLO(weights)
    img = cv2.imread(source)
    results = model(img)[0]

    output = {
        "image": source,
        "yolo_device": str(model.device),
        "ocr_engine": "tesseract",
        "fields": {},
        "boxes": []
    }

    classes = {
        0: "serving_size",
        1: "servings_per_package",
        2: "calories",
        3: "protein",
        4: "fat",
        5: "saturated",
        6: "trans_fat",
        7: "carbs",
        8: "sugar",
        9: "sodium"
    }

    fields = {k: {"raw": None,"cleaned": None,"value": None,"ok": False} for k in classes.values()}

    for i, box in enumerate(results.boxes):
        x1,y1,x2,y2 = map(int, box.xyxy[0])
        crop = img[max(0,y1-pad):y2+pad, max(0,x1-pad):x2+pad]
        cv2.imwrite(f"debug/crop_{i}.png", crop)

        strategies = preprocess_strategies(crop)
        votes = []

        cls_id = int(box.cls.item())
        cls_name = classes.get(cls_id, f"cls_{cls_id}")

        for name, pre in strategies.items():
            text = pytesseract.image_to_string(pre, lang="chi_tra+eng", config="--oem 3 --psm 6")
            text = normalize_text(text)
            val = extract_value(text, cls_name)

            if debug:
                cv2.imwrite(f"debug/crop_{i}_{name}.png", pre)

            votes.append({"strategy": name, "text": text, "value": val})

        best_val, selected_strategies = select_best_vote(votes)

        box_info = {
            "index": i,
            "bbox": [x1,y1,x2,y2],
            "class_id": cls_id,
            "class_name": cls_name,
            "score": float(box.conf.item()),
            "ocr": votes[0]["text"] if votes else None,
            "value": best_val,
            "ocr_votes": votes,
            "selected_strategy": selected_strategies
        }

        output["boxes"].append(box_info)

        if fields[cls_name]["raw"] is None:
            fields[cls_name]["raw"] = votes[0]["text"] if votes else None
            if cls_name == "sodium" and votes:
                fields[cls_name]["cleaned"] = clean_sodium_text(votes[0]["text"])
            else:
                fields[cls_name]["cleaned"] = re.sub(r"[^\d\.]", "", votes[0]["text"]) if votes else ""
            fields[cls_name]["value"] = best_val
            if cls_name == "sodium":
                fields[cls_name]["ok"] = check_sodium(best_val)
            else:
                fields[cls_name]["ok"] = best_val is not None

    output["fields"] = fields
    output["energy_check"] = check_energy(fields)

    with open(out, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"✅ OCR 完成，結果已輸出到 {out}")
    print(f"📂 最新切割與前處理圖片已存放於 debug/ 資料夾")

# --- CLI ---
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", default="result.json")
    ap.add_argument("--pad", type=int, default=3)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()
    run(args.weights, args.source, args.out, args.pad, args.debug)








