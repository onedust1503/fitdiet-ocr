# FitDiet OCR API

基於 YOLOv8 + Tesseract 的營養標籤辨識 API，用於食尚健身 App。

## 功能

- 使用 YOLO 模型偵測營養標籤上的 10 個欄位
- 多策略 OCR 前處理與投票機制
- 自動能量驗算（蛋白質、脂肪、碳水化合物）
- RESTful API 介面

## 部署到 Render

### 1. Fork 或 Clone 此儲存庫

```bash
git clone https://github.com/你的帳號/fitdiet-ocr.git
cd fitdiet-ocr
```

### 2. 確保檔案都在

- `app.py` - Flask API 主程式
- `adapters.py` - CLI 橋接器
- `infer_ocr_tesseract.py` - YOLO + OCR 推論腳本
- `best.pt` - 訓練好的 YOLO 權重
- `requirements.txt` - Python 依賴
- `Dockerfile` - Docker 容器設定
- `render.yaml` - Render 部署設定

### 3. 在 Render 建立服務

1. 登入 https://render.com
2. 點選 "New +" → "Blueprint"
3. 連結你的 GitHub 儲存庫
4. Render 會自動讀取 `render.yaml`
5. 點選 "Apply" 開始部署

### 4. 等待部署完成（約 5-10 分鐘）

部署成功後會得到一個網址：
```
https://fitdiet-ocr.onrender.com
```

## API 使用方式

### 健康檢查

```bash
curl https://fitdiet-ocr.onrender.com/health
```

### OCR 辨識

```bash
curl -X POST https://fitdiet-ocr.onrender.com/ocr \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -F "image=@nutrition_label.jpg"
```

回應範例：
```json
{
  "request_id": "a1b2c3d4",
  "model": "yolo-nutrition-ocr@1.0.0",
  "latency_ms": 1234,
  "items": [
    {"label": "calories", "value": 250.0, "raw_text": "250大卡"},
    {"label": "protein", "value": 8.0, "raw_text": "8g"},
    {"label": "fat", "value": 10.0, "raw_text": "10g"},
    {"label": "carbs", "value": 35.0, "raw_text": "35g"}
  ]
}
```

## 環境變數

在 Render Dashboard 可設定：

- `API_KEY` - API 驗證金鑰（自動產生）
- `YOLO_WEIGHTS` - 模型路徑（預設 `best.pt`）
- `LOG_LEVEL` - 日誌等級（預設 `INFO`）

## 技術架構

```
Flutter App → Render API → Flask → CliAdapter → infer_ocr_tesseract.py → YOLO + Tesseract
```

## 注意事項

- 免費方案閒置 15 分鐘後會休眠
- 首次啟動需要 30-50 秒
- 免費額度：每月 750 小時運行時間

## 授權

本專案為國立臺南大學數位學習科技學系畢業專題。
