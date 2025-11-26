
# 💤 Detector de Olhos Fechados

Projeto simples que detecta olhos **abertos/fechados** usando a **webcam**.
Se os olhos ficarem fechados por **4 segundos**, um **som de alerta** é tocado.

---

## 🚀 Funcionalidades

* Detecção em tempo real (MediaPipe + OpenCV)
* Identifica olhos abertos/fechados
* Alarme após 4s de olhos fechados
* Leve e sem necessidade de treinar modelos

---

## 📁 Estrutura

```
detector-olhos/
│
├── src/
│   └── detector_olhos.py
│
├── sounds/
│   └── alert.mp3
│
├── requirements.txt
└── README.md
```

---

## 🧩 Instalação

```bash
pip install -r requirements.txt
```

---

## ▶️ Como rodar

```bash
python src/detector_olhos.py
```

Pressione **Q** para sair.

---

## 🔊 Observação

Coloque seu som de alerta em:
---
sounds/alert.mp3
---