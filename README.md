 Shap-E 3D Asset Generator API
This project provides a powerful API for generating 3D assets from text prompts using OpenAI's Shap-E model. It's designed to be easily deployable on cloud GPU platforms like Lightning AI, with an optional ngrok tunnel for public access, making it ideal for rapid prototyping and integrating generative 3D capabilities into game development or other applications.

✨ Features
Text-to-3D Generation: Convert descriptive text prompts (e.g., "a red sports car", "a medieval sword") into 3D models.

Multiple Output Formats: Generate 3D assets in GIF (preview), OBJ, PLY, or GLB formats.

Configurable Generation: Adjust parameters like batch_size, karras_steps (quality), guidance_scale (adherence to prompt), and render_size (for GIF output).

RESTful API: A FastAPI-based API provides programmatic access to the 3D generation and status monitoring.

GPU Accelerated: Leverages CUDA-enabled GPUs for efficient and fast model inference.

Ngrok Integration: Includes a utility script (ngrok_server.py) to expose the local API server to the internet via an ngrok tunnel, useful for testing and sharing.

📦 Project Structure
.
├── api_server.py          # The main FastAPI application for 3D generation
├── ngrok_server.py        # Utility script to create a public ngrok tunnel
├── generated_assets/      # Directory where generated 3D models are temporarily saved
└── README.md              # This project's README file

🚀 Setup and Usage
This project is designed to run efficiently in a cloud environment with GPU access, such as a Lightning AI Studio.

Prerequisites
Python 3.8+

CUDA-enabled GPU (highly recommended for performance)

git installed

An ngrok authentication token (optional, for public access)

1. On Lightning AI (Recommended)
Sign Up/Log In: If you don't have one, create a free account on lightning.ai.

Create a New Studio:

From your Lightning AI dashboard, click "New Studio."

Choose a Python template (e.g., "Blank Python").

Crucially, select a GPU instance type (e.g., T4, L4, A100) from the "Compute" dropdown for practical 3D generation speeds.

Access Terminal: Once your Studio is running, open the integrated terminal.

Clone the Repository:

git clone https://github.com/your-username/your-repo-name.git .
# Replace with your actual GitHub repository URL

Install Dependencies:

Install PyTorch (matching CUDA): First, check the CUDA version available on your Lightning AI instance using nvcc --version. Then, go to pytorch.org/get-started/locally/ and copy the pip install command that matches your CUDA version.
Example for CUDA 11.8:

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

Install Shap-E and trimesh:

pip install -e git+https://github.com/openai/shap-e.git#egg=shap_e
pip install trimesh

Install other Python packages:

pip install fastapi uvicorn python-multipart Pillow

(Note: pyngrok is needed for ngrok_server.py but not directly by api_server.py)

Run the API Server:

In the Studio terminal, run:

python api_server.py

The FastAPI server will start, typically on port 8000. You'll see messages like Uvicorn running on http://0.0.0.0:8000.

(Optional) Expose API with Ngrok:

Open a new terminal in your Lightning AI Studio (keep api_server.py running in the first terminal).

Install pyngrok: pip install pyngrok

Set your ngrok auth token: The ngrok_server.py file expects NGROK_AUTH_TOKEN to be set. You can either:

Paste your token directly into ngrok_server.py (less secure for public repos): ngrok.set_auth_token("YOUR_AUTH_TOKEN_HERE")

Or, preferably, set it as an environment variable in your Studio.

Run the ngrok tunnel script:

python ngrok_server.py

This script will print a public URL (e.g., https://xxxx-xxxx-xxxx-xxxx.ngrok-free.app). You can use this URL to access your API from anywhere.
