import os
import uuid
import base64
import datetime as dt

import numpy as np
import cv2
from flask import Flask, request, jsonify
from flask_cors import CORS
from supabase import create_client

# ========= Config =========
ALLOWED_ORIGIN        = os.getenv("ALLOWED_ORIGIN", "*")
SUPABASE_URL          = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY  = os.getenv("SUPABASE_SERVICE_KEY")   # use SEMPRE no backend
SUPABASE_BUCKET       = os.getenv("SUPABASE_BUCKET", "photos")

if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
    raise RuntimeError("Defina SUPABASE_URL e SUPABASE_SERVICE_KEY nas variáveis de ambiente.")

supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": ALLOWED_ORIGIN}})

def _unique_path() -> str:
    """Gera caminho único por data/UUID para o arquivo no bucket."""
    today = dt.datetime.utcnow().strftime("%Y/%m/%d")
    return f"{today}/{uuid.uuid4().hex}.png"

# ======= Rotas de saúde/diagnóstico =======
@app.get("/")
def root():
    return "✅ Backend online. Use POST /api/processar-foto", 200

@app.get("/health")
def health():
    return {"status": "ok"}, 200

# ============== API principal ==============
@app.post("/api/processar-foto")
def processar_foto():
    try:
        data = request.get_json()
        if not data or "imagem" not in data or "tamanho" not in data:
            return jsonify({"erro": "Payload inválido"}), 400

        # imagem vem como "data:image/jpeg;base64,...."
        try:
            b64 = data["imagem"].split(",")[1]
        except Exception:
            return jsonify({"erro": "Formato de imagem inválido (esperado dataURL)"}), 400

        arr = np.frombuffer(base64.b64decode(b64), np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return jsonify({"erro": "Falha ao decodificar imagem"}), 400

        # Redimensiona
        h, w = map(int, data["tamanho"])  # [height, width]
        proc = cv2.resize(img, (w, h))

        vmin, vmax = float(proc.min()), float(proc.max())

        # Codifica PNG em memória (bytes)
        ok, buf = cv2.imencode(".png", proc)
        if not ok:
            return jsonify({"erro": "Falha ao codificar PNG"}), 500
        png_bytes: bytes = buf.tobytes()

        # Para exibir a imagem processada no front
        b64_proc = base64.b64encode(png_bytes).decode("utf-8")

        # ===== Upload no Supabase Storage (USAR BYTES, não BytesIO) =====
        path = _unique_path()
        try:
            # supabase-py v2 espera bytes (ou caminho de arquivo). Nada de BytesIO aqui.
            resp = supabase.storage.from_(SUPABASE_BUCKET).upload(
                path=path,
                file=png_bytes,
                file_options={"contentType": "image/png", "upsert": "true"}
            )
            print("UPLOAD RESP:", resp, flush=True)
        except Exception as up_err:
            print("UPLOAD ERROR:", up_err, flush=True)
            return jsonify({"erro": f"Falha no upload Supabase: {up_err}"}), 500

        # Bucket público: URL pública
        public_url = supabase.storage.from_(SUPABASE_BUCKET).get_public_url(path)

        # Se o bucket for PRIVADO, use signed URL (descomente):
        # signed = supabase.storage.from_(SUPABASE_BUCKET).create_signed_url(path, 3600)  # 1h
        # public_url = signed.get("signedURL") if isinstance(signed, dict) else public_url

        return jsonify({
            "dimensao_original": [int(img.shape[0]), int(img.shape[1])],
            "dimensao_processada": [int(proc.shape[0]), int(proc.shape[1])],
            "valor_min": vmin,
            "valor_max": vmax,
            "arquivo_salvo": path,
            "arquivo_salvo_url": public_url,
            "imagem_processada_base64": b64_proc
        })

    except Exception as e:
        print("API ERROR:", e, flush=True)
        return jsonify({"erro": str(e)}), 500


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
