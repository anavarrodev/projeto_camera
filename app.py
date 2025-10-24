import os
import uuid
import base64
import datetime as dt

import numpy as np
import cv2
from flask import Flask, request, jsonify
from flask_cors import CORS
from supabase import create_client

from skimage import transform as sktf, color as skcolor, util as skut

# ========= Config =========
ALLOWED_ORIGIN        = os.getenv("ALLOWED_ORIGIN", "*")
SUPABASE_URL          = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY  = os.getenv("SUPABASE_SERVICE_KEY")
SUPABASE_BUCKET       = os.getenv("SUPABASE_BUCKET", "photos")

if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
    raise RuntimeError("Defina SUPABASE_URL e SUPABASE_SERVICE_KEY nas variáveis de ambiente.")

supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": ALLOWED_ORIGIN}})

# Adiciona logging de requisições
@app.before_request
def log_request():
    print(f"\n{'='*50}", flush=True)
    print(f"📥 {request.method} {request.path}", flush=True)
    print(f"Origin: {request.headers.get('Origin', 'N/A')}", flush=True)
    print(f"Content-Type: {request.headers.get('Content-Type', 'N/A')}", flush=True)

@app.after_request
def log_response(response):
    print(f"📤 Status: {response.status_code}", flush=True)
    print(f"{'='*50}\n", flush=True)
    return response

def _unique_path() -> str:
    """Gera caminho único por data/UUID para o arquivo no bucket."""
    today = dt.datetime.utcnow().strftime("%Y/%m/%d")
    return f"{today}/{uuid.uuid4().hex}.png"

# ======= Rotas de saúde/diagnóstico =======
@app.get("/")
def root():
    return jsonify({"message": "✅ Backend online", "endpoints": ["/health", "/api/processar-foto"]}), 200

@app.get("/health")
def health():
    return jsonify({"status": "ok", "timestamp": dt.datetime.utcnow().isoformat()}), 200

# ============== API principal ==============
@app.post("/api/processar-foto")
def processar_foto():
    print("🔄 Iniciando processamento...", flush=True)
    
    try:
        # Verifica Content-Type
        if not request.is_json:
            print(f"❌ Content-Type inválido: {request.content_type}", flush=True)
            return jsonify({"erro": "Content-Type deve ser application/json"}), 400
        
        data = request.get_json()
        if not data:
            print("❌ Body vazio", flush=True)
            return jsonify({"erro": "Body vazio"}), 400
            
        if "imagem" not in data or "tamanho" not in data:
            print(f"❌ Campos faltando. Recebido: {list(data.keys())}", flush=True)
            return jsonify({"erro": "Payload inválido - faltam 'imagem' e/ou 'tamanho'"}), 400

        print(f"✅ Payload válido. Tamanho solicitado: {data['tamanho']}", flush=True)

        # ---- 1) Decodificação (dataURL -> np.uint8) ----
        try:
            b64 = data["imagem"].split(",")[1]
        except Exception as e:
            print(f"❌ Erro ao extrair base64: {e}", flush=True)
            return jsonify({"erro": "Formato de imagem inválido (esperado dataURL)"}), 400

        arr = np.frombuffer(base64.b64decode(b64), np.uint8)
        img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            print("❌ cv2.imdecode retornou None", flush=True)
            return jsonify({"erro": "Falha ao decodificar imagem"}), 400

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        print(f"✅ Imagem decodificada: {img_rgb.shape}", flush=True)

        # ---- 2) Converte para grayscale float [0,1] ----
        if img_rgb.ndim == 3:
            img_gray_f = skcolor.rgb2gray(img_rgb)
        else:
            img_gray_f = skut.img_as_float(img_rgb)

        # ---- 3) Redimensiona com anti_aliasing ----
        h, w = map(int, data["tamanho"])
        proc_f = sktf.resize(
            img_gray_f,
            (h, w),
            anti_aliasing=True,
            preserve_range=False
        )
        print(f"✅ Imagem redimensionada: {proc_f.shape}", flush=True)

        # ---- 4) Normaliza explicitamente para [0,1] ----
        maxv = float(proc_f.max())
        if maxv > 0:
            proc_f = proc_f / maxv

        vmin, vmax = float(proc_f.min()), float(proc_f.max())
        print(f"✅ Normalizada - min: {vmin:.4f}, max: {vmax:.4f}", flush=True)

        # ---- 5) Converte para uint8 p/ salvar PNG ----
        proc_u8 = (np.clip(proc_f, 0.0, 1.0) * 255.0).round().astype(np.uint8)

        # ---- 6) Codifica PNG (bytes) ----
        ok, buf = cv2.imencode(".png", proc_u8, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        if not ok:
            print("❌ Falha ao codificar PNG", flush=True)
            return jsonify({"erro": "Falha ao codificar PNG"}), 500
        png_bytes: bytes = buf.tobytes()
        b64_proc = base64.b64encode(png_bytes).decode("utf-8")
        print(f"✅ PNG codificado: {len(png_bytes)} bytes", flush=True)

        # ---- 7) Upload no Supabase Storage ----
        path = _unique_path()
        print(f"🔄 Fazendo upload para: {path}", flush=True)
        
        try:
            resp = supabase.storage.from_(SUPABASE_BUCKET).upload(
                path=path,
                file=png_bytes,
                file_options={"contentType": "image/png", "upsert": "true"}
            )
            print(f"✅ Upload concluído: {resp}", flush=True)
        except Exception as up_err:
            print(f"❌ Erro no upload: {up_err}", flush=True)
            return jsonify({"erro": f"Falha no upload Supabase: {up_err}"}), 500

        public_url = supabase.storage.from_(SUPABASE_BUCKET).get_public_url(path)
        print(f"✅ URL pública: {public_url}", flush=True)

        resultado = {
            "dimensao_original": [int(img_rgb.shape[0]), int(img_rgb.shape[1])],
            "dimensao_processada": [h, w],
            "valor_min": vmin,
            "valor_max": vmax,
            "arquivo_salvo": path,
            "arquivo_salvo_url": public_url,
            "imagem_processada_base64": b64_proc
        }
        
        print("✅ Processamento concluído com sucesso!", flush=True)
        return jsonify(resultado)

    except Exception as e:
        print(f"❌ ERRO GERAL: {type(e).__name__}: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return jsonify({"erro": str(e)}), 500


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    print(f"\n{'='*50}")
    print(f"🚀 Iniciando servidor na porta {port}")
    print(f"📍 Endpoints disponíveis:")
    print(f"   GET  http://localhost:{port}/")
    print(f"   GET  http://localhost:{port}/health")
    print(f"   POST http://localhost:{port}/api/processar-foto")
    print(f"{'='*50}\n")
    app.run(host="0.0.0.0", port=port, debug=True)
