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

try:
    from shap_e.diffusion.sample import sample_latents
    from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
    from shap_e.models.download import load_model, load_config
    from shap_e.util.notebooks import create_pan_cameras, decode_latent_images, gif_widget, decode_latent_mesh
    import trimesh
except ImportError as e:
    print(f"Error importing shap-e or trimesh: {e}")
    print("Please ensure you have shap-e installed from its GitHub repo and trimesh: ")
    print("pip install -e git+https://github.com/openai/shap-e.git#egg=shap_e")
    print("pip install trimesh")
    exit(1)

# --- FastAPI App Setup ---
app = FastAPI(
    title="Shap-E 3D Asset Generator API",
    description="API for generating 3D assets from text prompts using OpenAI's Shap-E model.",
    version="0.1.0",
)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device} for Shap-E model loading.")


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
        model = load_model('text300M', device=device) 
        diffusion = diffusion_from_config(load_config('diffusion'))
        print("Shap-E models loaded successfully!")
    except Exception as e:
        print(f"Failed to load Shap-E models: {e}")
       
        raise RuntimeError(f"Failed to load Shap-E models: {e}")



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
        use_fp16=(device.type == 'cuda'),
        use_karras=True,
        karras_steps=karras_steps,
        sigma_min=1e-3,
        sigma_max=160,
        s_churn=0,
    )
    return latents



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
            
            
            cameras = create_pan_cameras(render_size, device)
            
            
            all_images = []
            for latent in latents:
                images = decode_latent_images(xm, latent, cameras, rendering_mode='nerf') 
                all_images.extend(images)

            if not all_images:
                raise HTTPException(status_code=500, detail="Failed to generate any images for GIF.")

            
            gif_byte_stream = io.BytesIO()
            
            
            pil_images = [img for img in all_images if isinstance(img, Image.Image)]

            if not pil_images:
                raise HTTPException(status_code=500, detail="Generated images are not in PIL format for GIF creation.")

           
            pil_images[0].save(
                gif_byte_stream,
                format="GIF",
                save_all=True,
                append_images=pil_images[1:],
                duration=100,
                loop=0
            )
            gif_byte_stream.seek(0) 

            return StreamingResponse(gif_byte_stream, media_type="image/gif")

        elif output_type in ["obj", "ply", "glb"]:
            if xm is None:
                raise RuntimeError("Transmitter model (xm) not loaded for mesh decoding.")

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
                               
                                loaded_mesh = trimesh.load(obj_path, force='mesh')
                                glb_path = os.path.join(tmpdir, f"{base_filename}.glb")
                                loaded_mesh.export(glb_path, file_type="glb")
                                generated_files[-1] = glb_path
                                print(f"Converted {obj_path} to {glb_path}")
                            except Exception as e:
                                print(f"Failed to convert OBJ to GLB for {obj_path}: {e}")
                                
                                if output_type == "glb": 
                                    raise HTTPException(status_code=500, detail=f"Failed to convert to GLB: {e}. Returning OBJ instead if available.")
                                else:
                                    pass


                    elif output_type == "ply":
                        ply_path = os.path.join(tmpdir, f"{base_filename}.ply")
                        with open(ply_path, 'wb') as f: # PLY is often binary
                            t.write_ply(f)
                        generated_files.append(ply_path)
                
                if not generated_files:
                    raise HTTPException(status_code=500, detail="No mesh files were generated.")


                if len(generated_files) > 1:
                    print(f"Warning: {len(generated_files)} assets generated. Returning only the first one.")
                
                final_file_path = generated_files[0]
                

                media_type = {
                    "obj": "model/obj",
                    "ply": "model/ply",
                    "glb": "model/gltf-binary"
                }.get(os.path.splitext(final_file_path)[1].lstrip('.').lower(), "application/octet-stream")


                with open(final_file_path, "rb") as f:
                    content = f.read()
                
                
                response = Response(content=content, media_type=media_type)
                response.headers["Content-Disposition"] = f"attachment; filename={os.path.basename(final_file_path)}"
                return response

        else:
            raise HTTPException(status_code=400, detail="Invalid output_type specified.")

    except HTTPException as he:
        raise he 
    except Exception as e:
        print(f"An error occurred during generation: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

@app.get("/status")
async def get_status():
    """
    Returns the current status of the model and server.
    """
    uptime = datetime.datetime.now() - app_start_time

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


if __name__ == "__main__":

    os.makedirs("generated_assets", exist_ok=True)
    

    uvicorn.run(app, host="0.0.0.0", port=8000)