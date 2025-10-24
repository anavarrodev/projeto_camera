import os
import uuid
import base64
import datetime as dt

import numpy as np
import cv2
from flask import Flask, request, jsonify
from flask_cors import CORS
from supabase import create_client
from skimage import transform, color

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

        # ===== PROCESSAMENTO SEGUINDO O CÓDIGO ORIGINAL =====
        
        # 1. Decodificar imagem
        arr = np.frombuffer(base64.b64decode(b64), np.uint8)
        img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            return jsonify({"erro": "Falha ao decodificar imagem"}), 400
        
        # 2. Converter BGR -> RGB
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # 3. Converter para escala de cinza (imagem_original do código)
        if len(img_rgb.shape) == 3:
            imagem_original = color.rgb2gray(img_rgb)
        else:
            imagem_original = img_rgb
        
        print(f"Dimensão Original: {imagem_original.shape}")
        
        # 4. Definição do tamanho alvo
        h, w = map(int, data["tamanho"])  # ex: [64, 64]
        novo_tamanho = (h, w)
        
        # 5. Redimensionamento da imagem
        # Usamos 'anti_aliasing=True' para uma suavização melhor
        imagem_redimensionada = transform.resize(
            imagem_original, 
            novo_tamanho, 
            anti_aliasing=True
        )
        
        # 6. Normalização
        # O 'transform.resize' já normaliza automaticamente para [0,1] se a entrada for uint8/uint16.
        # Para garantir, podemos fazer explicitamente:
        imagem_normalizada = imagem_redimensionada / imagem_redimensionada.max()
        
        print(f"Dimensão Processada: {imagem_normalizada.shape}")
        print(f"Valor Máximo (Processada): {imagem_normalizada.max():.2f}")
        
        # ===== FIM DO PROCESSAMENTO =====
        
        # 7. Metadados
        vmin, vmax = float(imagem_normalizada.min()), float(imagem_normalizada.max())
        
        # 8. Converter para uint8 para salvar PNG (0..255)
        proc_u8 = (np.clip(imagem_normalizada, 0.0, 1.0) * 255.0).astype(np.uint8)
        
        # 9. Codifica PNG em memória (bytes)
        ok, buf = cv2.imencode(".png", proc_u8, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        if not ok:
            return jsonify({"erro": "Falha ao codificar PNG"}), 500
        png_bytes: bytes = buf.tobytes()

        # Para exibir a imagem processada no front
        b64_proc = base64.b64encode(png_bytes).decode("utf-8")

        # ===== Upload no Supabase Storage =====
        path = _unique_path()
        try:
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

        return jsonify({
            "dimensao_original": [int(imagem_original.shape[0]), int(imagem_original.shape[1])],
            "dimensao_processada": [int(imagem_normalizada.shape[0]), int(imagem_normalizada.shape[1])],
            "valor_min": vmin,
            "valor_max": vmax,
            "arquivo_salvo": path,
            "arquivo_salvo_url": public_url,
            "imagem_processada_base64": b64_proc
        })

    except Exception as e:
        print("API ERROR:", e, flush=True)
        import traceback
        traceback.print_exc()
        return jsonify({"erro": str(e)}), 500


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
