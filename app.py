import os
import uuid
import base64
import datetime as dt

import numpy as np
import cv2
from flask import Flask, request, jsonify
from flask_cors import CORS
from supabase import create_client

# skimage para resize + normalização como no seu exemplo
from skimage import transform as sktf, color as skcolor, util as skut

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

        # ---- 1) Decodificação (dataURL -> np.uint8) ----
        try:
            b64 = data["imagem"].split(",")[1]
        except Exception:
            return jsonify({"erro": "Formato de imagem inválido (esperado dataURL)"}), 400

        arr = np.frombuffer(base64.b64decode(b64), np.uint8)
        # sempre carrega colorida para unificar o fluxo
        img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            return jsonify({"erro": "Falha ao decodificar imagem"}), 400

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # ---- 2) Converte para grayscale float [0,1] ----
        if img_rgb.ndim == 3:
            img_gray_f = skcolor.rgb2gray(img_rgb)        # float64 [0,1]
        else:
            # já é 1 canal uint8 -> normaliza para [0,1]
            img_gray_f = skut.img_as_float(img_rgb)

        # ---- 3) Redimensiona com anti_aliasing (como no seu exemplo) ----
        h, w = map(int, data["tamanho"])   # [height, width], ex.: 64, 64
        proc_f = sktf.resize(
            img_gray_f,
            (h, w),
            anti_aliasing=True,
            preserve_range=False  # saída já em [0,1]
        )

        # ---- 4) Normaliza explicitamente para [0,1] ----
        maxv = float(proc_f.max())
        if maxv > 0:
            proc_f = proc_f / maxv

        # metadados em float [0,1]
        vmin, vmax = float(proc_f.min()), float(proc_f.max())

        # ---- 5) Converte para uint8 p/ salvar PNG ----
        proc_u8 = (np.clip(proc_f, 0.0, 1.0) * 255.0).round().astype(np.uint8)

        # ---- 6) Codifica PNG (bytes) ----
        ok, buf = cv2.imencode(".png", proc_u8, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        if not ok:
            return jsonify({"erro": "Falha ao codificar PNG"}), 500
        png_bytes: bytes = buf.tobytes()

        # também devolve base64 para exibir direto no front
        b64_proc = base64.b64encode(png_bytes).decode("utf-8")

        # ---- 7) Upload no Supabase Storage (bytes) ----
        path = _unique_path()
        try:
            resp = supabase.storage.from_(SUPABASE_BUCKET).upload(
                path=path,
                file=png_bytes,  # envia bytes (não BytesIO)
                file_options={"contentType": "image/png", "upsert": "true"}
            )
            print("UPLOAD RESP:", resp, flush=True)
        except Exception as up_err:
            print("UPLOAD ERROR:", up_err, flush=True)
            return jsonify({"erro": f"Falha no upload Supabase: {up_err}"}), 500

        # Bucket público -> URL pública
        public_url = supabase.storage.from_(SUPABASE_BUCKET).get_public_url(path)

        # Se o bucket for PRIVADO, troque por signed URL:
        # signed = supabase.storage.from_(SUPABASE_BUCKET).create_signed_url(path, 3600)  # 1h
        # public_url = signed.get("signedURL") if isinstance(signed, dict) else public_url

        return jsonify({
            "dimensao_original": [int(img_rgb.shape[0]), int(img_rgb.shape[1])],
            "dimensao_processada": [h, w],
            "valor_min": vmin,
            "valor_max": vmax,          # deve ser ~1.0 após normalização
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
