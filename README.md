# 🧠 Shap-E 3D Asset Generator

Generate high-quality 3D assets (GLB, OBJ, PLY, GIF) from text prompts using OpenAI's [Shap-E](https://github.com/openai/shap-e) model. This repo supports:

- ✅ CLI-based 3D generation using `diffusers` and `shap-e`
- ✅ FastAPI server to expose 3D generation via `/generate3d` API
- ✅ Ngrok tunnel support for public access

---

## 🚀 Features

- **Text-to-3D** using `Shap-E` (via Hugging Face or OpenAI's repo)
- Outputs: `.glb`, `.obj`, `.ply`, `.gif`
- Lightweight REST API built with FastAPI
- Auto GPU/CPU support
- Ngrok tunneling for Colab/public access

---

## 📦 Folder Structure

```bash
.
├── api_server.py        # FastAPI server for Shap-E
├── ngrok_server.py      # Starts ngrok tunnel
├── shap_e_import.py     # Standalone script for batch 3D generation
├── generated_assets/    # Output folder for 3D assets
├── requirements.txt
└── README.md

## Installation
git clone https://github.com/your-username/3d-shap-e-generator.git
cd 3d-shap-e-generator

# Install requirements
pip install -r requirements.txt

# For local development
pip install -e git+https://github.com/openai/shap-e.git#egg=shap_e

