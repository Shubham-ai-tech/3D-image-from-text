#  Shap-E 3D Asset Generator

Generate high-quality 3D assets (GLB, OBJ, PLY, GIF) from text prompts using OpenAI's [Shap-E](https://github.com/openai/shap-e) model. This repo supports:

- CLI-based 3D generation using `diffusers` and `shap-e`
- FastAPI server to expose 3D generation via `/generate3d` API
- Ngrok tunnel support for public access

---

##  Features

- **Text-to-3D** using `Shap-E` (via Hugging Face or OpenAI's repo)
- Outputs: `.glb`, `.obj`, `.ply`, `.gif`
- Lightweight REST API built with FastAPI
- Auto GPU/CPU support
- Ngrok tunneling for Colab/public access

---

##  Folder Structure

```bash
.
├── api_server.py        # FastAPI server for Shap-E
├── ngrok_server.py      # Starts ngrok tunnel
├── shap_e_import.py     # Standalone script for batch 3D generation
├── generated_assets/    # Output folder for 3D assets
├── requirements.txt
└── README.md
```

---

##  Installation

```bash
git clone https://github.com/your-username/3d-shap-e-generator.git
cd 3d-shap-e-generator

# Install requirements
pip install -r requirements.txt

# For local development
pip install -e git+https://github.com/openai/shap-e.git#egg=shap_e
```

---

##  Usage

###  Script-Based Generation

```bash
python shap_e_import.py
```

3D models (GLB & OBJ) will be saved in `generated_assets/`.

---

###  Run API Server

```bash
python api_server.py
```

Test with:
```
http://localhost:8000/generate3d?prompt=a wooden boat&output_type=glb
```

---

###  Optional: Run ngrok tunnel (for Colab/remote)

```bash
python ngrok_server.py
```

It will expose the local FastAPI server to the internet.

---

##  API Endpoints

### `GET /generate3d`

Generate a 3D asset from text.

**Query Parameters:**

| Param         | Type   | Description                              |
|---------------|--------|------------------------------------------|
| `prompt`      | string | Input text for 3D generation             |
| `output_type` | string | `gif`, `obj`, `ply`, `glb`               |
| `batch_size`  | int    | 1–4 objects                              |
| `karras_steps`| int    | Number of inference steps (default: 64)  |
| `guidance_scale` | float | How strictly prompt is followed        |
| `render_size` | int    | For GIF output only                      |

---

### `GET /status`

Check server uptime and GPU availability.

---

##  Example Prompts

- `"a cute cartoon dog wearing a hat"`
- `"a futuristic spaceship"`
- `"a medieval knight helmet"`
- `"a lava sword with glowing runes"`

---

##  Credits

Built with:
- OpenAI's [Shap-E](https://github.com/openai/shap-e)
- Hugging Face [Diffusers](https://huggingface.co/docs/diffusers/index)
- FastAPI + Uvicorn
- Trimesh + Pyngrok

---

## 📜 License

MIT License


