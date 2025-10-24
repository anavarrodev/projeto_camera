from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
from PIL import Image
from skimage import transform, color
import io
import base64
import os

app = Flask(__name__)
CORS(app)  # Permitir requisições do frontend

# Diretório para salvar as fotos
UPLOAD_FOLDER = 'fotos_capturadas'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def processar_imagem(imagem_array, novo_tamanho=(64, 64)):
    """
    Processa a imagem: converte para escala de cinza, redimensiona e normaliza
    
    Args:
        imagem_array: array numpy da imagem
        novo_tamanho: tupla com as dimensões desejadas (altura, largura)
    
    Returns:
        dicionário com informações da imagem processada
    """
    
    # 1. Converter para escala de cinza se for colorida
    if len(imagem_array.shape) == 3:
        imagem_cinza = color.rgb2gray(imagem_array)
    else:
        imagem_cinza = imagem_array
    
    # 2. Redimensionamento da imagem
    imagem_redimensionada = transform.resize(
        imagem_cinza, 
        novo_tamanho, 
        anti_aliasing=True
    )
    
    # 3. Normalização
    imagem_normalizada = imagem_redimensionada / imagem_redimensionada.max()
    
    # Converter para base64 para enviar ao frontend
    imagem_pil = Image.fromarray((imagem_normalizada * 255).astype(np.uint8))
    buffer = io.BytesIO()
    imagem_pil.save(buffer, format='PNG')
    img_base64 = base64.b64encode(buffer.getvalue()).decode()
    
    return {
        'dimensao_original': imagem_cinza.shape,
        'dimensao_processada': imagem_normalizada.shape,
        'valor_max': float(imagem_normalizada.max()),
        'valor_min': float(imagem_normalizada.min()),
        'imagem_processada_base64': img_base64
    }


@app.route('/')
def index():
    """Serve o arquivo HTML"""
    return send_from_directory('.', 'index.html')


@app.route('/api/processar-foto', methods=['POST'])
def processar_foto():
    """
    Endpoint para receber a foto do frontend, processar e retornar resultados
    """
    try:
        data = request.json
        
        # Receber imagem em base64
        imagem_base64 = data.get('imagem')
        novo_tamanho = data.get('tamanho', [64, 64])
        
        if not imagem_base64:
            return jsonify({'erro': 'Nenhuma imagem fornecida'}), 400
        
        # Decodificar imagem
        imagem_data = base64.b64decode(imagem_base64.split(',')[1])
        imagem = Image.open(io.BytesIO(imagem_data))
        imagem_array = np.array(imagem)
        
        # Salvar imagem original
        timestamp = len(os.listdir(app.config['UPLOAD_FOLDER'])) + 1
        filename = f'foto_{timestamp}.jpg'
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        imagem.save(filepath)
        
        # Processar imagem
        resultado = processar_imagem(imagem_array, tuple(novo_tamanho))
        resultado['arquivo_salvo'] = filename
        
        print(f"✅ Foto processada: {filename}")
        print(f"📐 Original: {resultado['dimensao_original']}")
        print(f"📐 Processada: {resultado['dimensao_processada']}")
        
        return jsonify(resultado), 200
        
    except Exception as e:
        print(f"❌ Erro ao processar foto: {str(e)}")
        return jsonify({'erro': str(e)}), 500


@app.route('/api/fotos', methods=['GET'])
def listar_fotos():
    """
    Lista todas as fotos capturadas
    """
    try:
        fotos = os.listdir(app.config['UPLOAD_FOLDER'])
        return jsonify({'fotos': fotos, 'total': len(fotos)}), 200
    except Exception as e:
        return jsonify({'erro': str(e)}), 500


@app.route('/api/foto/<filename>')
def obter_foto(filename):
    """
    Retorna uma foto específica
    """
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == '__main__':
    print("=" * 70)
    print("🚀 SERVIDOR BACKEND - CAPTURA E PROCESSAMENTO DE IMAGENS")
    print("=" * 70)
    print("\n📍 Servidor rodando em: http://localhost:5000")
    print("📂 Fotos salvas em:", UPLOAD_FOLDER)
    print("\n⚠️  ENDPOINTS DISPONÍVEIS:")
    print("  • GET  /                    - Frontend (index.html)")
    print("  • POST /api/processar-foto  - Processar imagem")
    print("  • GET  /api/fotos           - Listar fotos capturadas")
    print("  • GET  /api/foto/<filename> - Obter foto específica")
    print("=" * 70)
    print()
    
    app.run(debug=True, host='0.0.0.0', port=8080)
