import os, uuid, base64, io, datetime as dt
import numpy as np
import cv2
from flask import Flask, request, jsonify
from flask_cors import CORS
from supabase import create_client

ALLOWED_ORIGIN        = os.getenv("ALLOWED_ORIGIN", "*")
SUPABASE_URL          = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY  = os.getenv("SUPABASE_SERVICE_KEY")  # <- use service key no backend
SUPABASE_ANONKEY      = os.getenv("SUPABASE_ANON_KEY")     # opcional (não usar para upload)
SUPABASE_BUCKET       = os.getenv("SUPABASE_BUCKET", "photos")

if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
    raise RuntimeError("Defina SUPABASE_URL e SUPABASE_SERVICE_KEY nas variáveis de ambiente.")

supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": ALLOWED_ORIGIN}})

def _unique_path():
    today = dt.datetime.utcnow().strftime("%Y/%m/%d")
    return f"{today}/{uuid.uuid4().hex}.png"

@app.route("/api/processar-foto", methods=["POST"])
def processar_foto():
    try:
        data = request.get_json()
        if not data or "imagem" not in data or "tamanho" not in data:
            return jsonify({"erro": "Payload inválido"}), 400

        # decode base64
        b64 = data["imagem"].split(",")[1]
        arr = np.frombuffer(base64.b64decode(b64), np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return jsonify({"erro": "Falha ao decodificar imagem"}), 400

        h, w = map(int, data["tamanho"])           # [height, width]
        proc = cv2.resize(img, (w, h))
        vmin, vmax = float(proc.min()), float(proc.max())

        ok, buf = cv2.imencode(".png", proc)
        if not ok:
            return jsonify({"erro": "Falha ao codificar PNG"}), 500
        png_bytes = buf.tobytes()
        b64_proc = base64.b64encode(png_bytes).decode("utf-8")

        # --- Upload Supabase (com Service Key) ---
        path = _unique_path()
        try:
            supabase.storage.from_(SUPABASE_BUCKET).upload(
                path=path,
                file=io.BytesIO(png_bytes),
                file_options={"contentType": "image/png", "upsert": True}  # <- contentType correto
            )
        except Exception as up_err:
            # log útil para os logs do Render
            print("UPLOAD ERROR:", up_err, flush=True)
            return jsonify({"erro": f"Falha no upload Supabase: {up_err}"}), 500

        # Se bucket for público:
        public_url = supabase.storage.from_(SUPABASE_BUCKET).get_public_url(path)

        # Se seu bucket for PRIVADO, use signed URL:
        # signed = supabase.storage.from_(SUPABASE_BUCKET).create_signed_url(path, 3600)
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
