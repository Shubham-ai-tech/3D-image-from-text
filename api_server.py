import torch
import os
import io
import time
from fastapi import FastAPI, Query, HTTPException, Response
from fastapi.responses import StreamingResponse, FileResponse
from PIL import Image
import uvicorn
import datetime
import tempfile
import shutil

# Import shap-e components
# Ensure shap-e is installed: pip install -e git+https://github.com/openai/shap-e.git#egg=shap_e
# Also ensure trimesh is installed: pip install trimesh
try:
    from shap_e.diffusion.sample import sample_latents
    from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
    from shap_e.models.download import load_model, load_config
    from shap_e.util.notebooks import create_pan_cameras, decode_latent_images, gif_widget, decode_latent_mesh
    import trimesh # Import trimesh for potential GLB conversion or robust OBJ/PLY handling
except ImportError as e:
    print(f"Error importing shap-e or trimesh: {e}")
    print("Please ensure you have shap-e installed from its GitHub repo and trimesh: ")
    print("pip install -e git+https://github.com/openai/shap-e.git#egg=shap_e")
    print("pip install trimesh")
    exit(1) # Exit if essential libraries are not found

# --- FastAPI App Setup ---
app = FastAPI(
    title="Shap-E 3D Asset Generator API",
    description="API for generating 3D assets from text prompts using OpenAI's Shap-E model.",
    version="0.1.0",
)

# --- Global Model Loading ---
# Load models only once when the application starts
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device} for Shap-E model loading.")

# Define global variables for models
xm = None
model = None
diffusion = None
app_start_time = datetime.datetime.now()

@app.on_event("startup")
async def load_models_on_startup():
    global xm, model, diffusion
    print("Loading Shap-E models... This may take a few minutes.")
    try:
        xm = load_model('transmitter', device=device)
        model = load_model('text300M', device=device) # 'text300M' for text-to-3D
        diffusion = diffusion_from_config(load_config('diffusion'))
        print("Shap-E models loaded successfully!")
    except Exception as e:
        print(f"Failed to load Shap-E models: {e}")
        # You might want to raise an exception here to prevent the app from starting
        # if model loading is critical. For now, we'll let it start but generation will fail.
        raise RuntimeError(f"Failed to load Shap-E models: {e}")

# --- Utility Functions ---

def generate_shap_e_latents(prompt: str, batch_size: int = 1, guidance_scale: float = 15.0, karras_steps: int = 64):
    """Generates latents using Shap-E's diffusion model."""
    if model is None or diffusion is None:
        raise RuntimeError("Shap-E models are not loaded. Please wait for startup or check logs.")

    print(f"Sampling latents for prompt: '{prompt}' (batch_size={batch_size})")
    latents = sample_latents(
        batch_size=batch_size,
        model=model,
        diffusion=diffusion,
        guidance_scale=guidance_scale,
        model_kwargs=dict(texts=[prompt] * batch_size),
        progress=True,
        clip_denoised=True,
        use_fp16=(device.type == 'cuda'), # Use FP16 only if CUDA is available
        use_karras=True,
        karras_steps=karras_steps,
        sigma_min=1e-3,
        sigma_max=160,
        s_churn=0,
    )
    return latents

# --- API Endpoints ---

@app.get("/generate3d")
async def generate_3d_asset(
    prompt: str = Query(..., min_length=3, max_length=100, description="Text prompt for 3D generation"),
    output_type: str = Query("gif", pattern="^(gif|obj|ply|glb)$", description="Desired output format (gif, obj, ply, glb)"),
    batch_size: int = Query(1, ge=1, le=4, description="Number of assets to generate (max 4 for efficiency)"),
    karras_steps: int = Query(64, ge=32, le=128, description="Number of sampling steps (higher = better quality, slower)"),
    guidance_scale: float = Query(15.0, ge=5.0, le=30.0, description="Guidance scale (higher = more adherence to prompt)"),
    render_size: int = Query(64, ge=32, le=128, description="Resolution of GIF frames (only for gif output)")
):
    """
    Generates a 3D asset or its preview from a text prompt.
    """
    try:
        latents = generate_shap_e_latents(prompt, batch_size, guidance_scale, karras_steps)

        if output_type == "gif":
            if xm is None:
                raise RuntimeError("Transmitter model (xm) not loaded for GIF rendering.")
            
            # Create camera trajectory for GIF
            cameras = create_pan_cameras(render_size, device)
            
            # Decode latents to images for GIF
            all_images = []
            for latent in latents:
                images = decode_latent_images(xm, latent, cameras, rendering_mode='nerf') # 'nerf' or 'stf'
                all_images.extend(images) # Collect all images from all generated latents

            if not all_images:
                raise HTTPException(status_code=500, detail="Failed to generate any images for GIF.")

            # Create a BytesIO object to store the GIF
            gif_byte_stream = io.BytesIO()
            
            # gif_widget creates and saves the GIF, but it's designed for notebooks.
            # We need to manually create the GIF using PIL.
            # Convert images to PIL.Image objects if they aren't already
            pil_images = [img for img in all_images if isinstance(img, Image.Image)]

            if not pil_images:
                raise HTTPException(status_code=500, detail="Generated images are not in PIL format for GIF creation.")

            # Save the list of PIL Images as a GIF
            pil_images[0].save(
                gif_byte_stream,
                format="GIF",
                save_all=True,
                append_images=pil_images[1:],
                duration=100, # milliseconds per frame
                loop=0 # loop forever
            )
            gif_byte_stream.seek(0) # Rewind to the beginning of the stream

            return StreamingResponse(gif_byte_stream, media_type="image/gif")

        elif output_type in ["obj", "ply", "glb"]:
            if xm is None:
                raise RuntimeError("Transmitter model (xm) not loaded for mesh decoding.")

            # Create a temporary directory to store the mesh files
            # This is important because decode_latent_mesh might generate multiple files (obj, mtl, textures)
            with tempfile.TemporaryDirectory() as tmpdir:
                generated_files = []
                for i, latent in enumerate(latents):
                    t = decode_latent_mesh(xm, latent).tri_mesh()
                    
                    base_filename = f"{prompt.replace(' ', '_')}_{i}"
                    
                    if output_type == "obj" or output_type == "glb":
                        obj_path = os.path.join(tmpdir, f"{base_filename}.obj")
                        with open(obj_path, 'w') as f:
                            t.write_obj(f)
                        generated_files.append(obj_path)

                        if output_type == "glb":
                            try:
                                # trimesh load can be finicky, force='mesh' might help
                                loaded_mesh = trimesh.load(obj_path, force='mesh')
                                glb_path = os.path.join(tmpdir, f"{base_filename}.glb")
                                loaded_mesh.export(glb_path, file_type="glb")
                                generated_files[-1] = glb_path # Replace obj path with glb path
                                print(f"Converted {obj_path} to {glb_path}")
                            except Exception as e:
                                print(f"Failed to convert OBJ to GLB for {obj_path}: {e}")
                                # If GLB conversion fails, we might still want to return the OBJ
                                if output_type == "glb": # If GLB was explicitly requested, raise error
                                    raise HTTPException(status_code=500, detail=f"Failed to convert to GLB: {e}. Returning OBJ instead if available.")
                                else:
                                    # Continue, returning OBJ if GLB conversion was secondary
                                    pass


                    elif output_type == "ply":
                        ply_path = os.path.join(tmpdir, f"{base_filename}.ply")
                        with open(ply_path, 'wb') as f: # PLY is often binary
                            t.write_ply(f)
                        generated_files.append(ply_path)
                
                if not generated_files:
                    raise HTTPException(status_code=500, detail="No mesh files were generated.")

                # If multiple files are generated, you might want to zip them.
                # For simplicity, let's return the first generated file.
                # For multiple assets, a more complex response (e.g., zip archive) would be better.
                if len(generated_files) > 1:
                    print(f"Warning: {len(generated_files)} assets generated. Returning only the first one.")
                
                final_file_path = generated_files[0]
                
                # Determine media type based on extension
                media_type = {
                    "obj": "model/obj",
                    "ply": "model/ply",
                    "glb": "model/gltf-binary"
                }.get(os.path.splitext(final_file_path)[1].lstrip('.').lower(), "application/octet-stream")

                # Serve the file
                # FileResponse handles cleaning up the temporary file (if the tempdir context manager is still active)
                # or you'd need to manually manage cleanup.
                # Since we use TemporaryDirectory, it's cleaned up after the 'with' block exits.
                # So we need to copy the file out or use a different temporary file strategy.

                # Let's copy the file to a known location or stream it directly
                # Streaming the file from tempfile is more robust.
                with open(final_file_path, "rb") as f:
                    content = f.read()
                
                # Create a response that includes headers for file download
                response = Response(content=content, media_type=media_type)
                response.headers["Content-Disposition"] = f"attachment; filename={os.path.basename(final_file_path)}"
                return response

        else:
            raise HTTPException(status_code=400, detail="Invalid output_type specified.")

    except HTTPException as he:
        raise he # Re-raise FastAPI HTTP exceptions directly
    except Exception as e:
        print(f"An error occurred during generation: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

@app.get("/status")
async def get_status():
    """
    Returns the current status of the model and server.
    """
    uptime = datetime.datetime.now() - app_start_time
    # Format uptime nicely
    days = uptime.days
    hours, remainder = divmod(uptime.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    uptime_str = f"{days} days, {hours:02}:{minutes:02}:{seconds:02}"

    gpu_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if gpu_available else "N/A"

    return {
        "status": "online",
        "model_loaded": xm is not None and model is not None and diffusion is not None,
        "model_name": "Shap-E (openai/text300M)",
        "device": device.type,
        "gpu_name": gpu_name,
        "uptime": uptime_str,
        "api_version": app.version,
    }

# --- Run the application (for local testing) ---
if __name__ == "__main__":
    # Ensure a directory for generated assets exists
    os.makedirs("generated_assets", exist_ok=True)
    
    # You might want to adjust the host/port for deployment
    uvicorn.run(app, host="0.0.0.0", port=8000)