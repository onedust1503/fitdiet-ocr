import io, os, time, uuid, json, logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
from werkzeug.middleware.proxy_fix import ProxyFix
from adapters import get_infer_adapter, InferenceError

APP_NAME = os.getenv("APP_NAME", "ocr-tesseract-api")
MODEL_NAME = os.getenv("MODEL_NAME", "ocr-tesseract@1.0.0")
API_KEY = os.getenv("API_KEY", "")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(APP_NAME)

app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1)
CORS(app, resources={r"/*": {"origins": "*"}})

adapter = get_infer_adapter()  # 依 .env 選用 func 或 cli

def _auth_ok(req):
    if not API_KEY:
        return True
    auth = req.headers.get("Authorization", "")
    return auth == f"Bearer {API_KEY}"

@app.route("/health")
def health():
    return jsonify({"ok": True, "model": MODEL_NAME})

@app.route("/ocr", methods=["POST"])
def ocr():
    t0 = time.time()

    if not _auth_ok(request):
        return jsonify({"error": "unauthorized"}), 401

    if "image" not in request.files:
        return jsonify({"error": "no image"}), 400

    f = request.files["image"]
    try:
        pil = Image.open(io.BytesIO(f.read())).convert("RGB")
    except Exception as e:
        return jsonify({"error": "invalid image", "detail": str(e)}), 400

    return_boxes = request.form.get("return_boxes", "false").lower() == "true"

    try:
        items, boxes = adapter.infer(
            pil_img=pil,
            engine="tesseract",
            return_boxes=return_boxes
        )
    except InferenceError as e:
        log.exception("inference error")
        return jsonify({"error": "inference_error", "detail": str(e)}), 500
    except Exception as e:
        log.exception("unknown error")
        return jsonify({"error": "unknown_error", "detail": str(e)}), 500

    resp = {
        "request_id": uuid.uuid4().hex[:8],
        "model": MODEL_NAME,
        "latency_ms": int((time.time() - t0) * 1000),
        "items": items
    }
    if return_boxes:
        resp["boxes"] = boxes

    return jsonify(resp), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
