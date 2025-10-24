# 📸 Projeto de Captura e Processamento de Imagens

Sistema web para **capturar imagens da câmera**, enviar para um **backend Flask** e exibir o resultado do **processamento em tempo real**, incluindo metadados e imagem processada.

---

## 🚀 Funcionalidades

✅ Captura de imagem diretamente da câmera (desktop ou mobile)  
✅ Contagem regressiva com animação antes da captura  
✅ Envio da imagem em base64 para o backend Flask  
✅ Processamento e retorno da imagem tratada  
✅ Exibição da imagem original e processada lado a lado  
✅ Visualização de metadados (dimensões, valores, nome do arquivo)  
✅ Botão 🔄 **Reiniciar** para novo ciclo de captura sem recarregar a página  

---

## 🧩 Estrutura do Projeto

Meu_projeto_Camera/
│
├── app.py # Backend Flask (API de processamento)
├── templates/
│ └── index.html # Interface principal (frontend)
├── static/
│ └── imagens/ # (Opcional) Pasta para salvar imagens processadas
└── README.md # Este arquivo
