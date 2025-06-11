# main.py
from fastapi import FastAPI, UploadFile, File, Form
from pathlib import Path
from PIL import Image
from diffusers import FluxFillPipeline
from diffusers.utils import load_image
import shutil, torch, uuid, os

app = FastAPI()
model = FluxFillPipeline.from_pretrained("black-forest-labs/FLUX.1-Fill-dev", torch_dtype=torch.bfloat16)
model.enable_model_cpu_offload()

@app.post("/inpaint/")
async def inpaint_image(
    prompt: str = Form("a clean product image with no watermark"),
    image: UploadFile = File(...),
    mask: UploadFile = File(...)
):
    uid = str(uuid.uuid4())
    input_path = Path(f"/tmp/{uid}_input.png")
    mask_path = Path(f"/tmp/{uid}_mask.png")

    # Save uploaded files
    with open(input_path, "wb") as f:
        shutil.copyfileobj(image.file, f)
    with open(mask_path, "wb") as f:
        shutil.copyfileobj(mask.file, f)

    # Load and preprocess
    img = load_image(str(input_path)).convert("RGB")
    msk = load_image(str(mask_path)).convert("L").resize(img.size)

    # Inpaint
    result = model(
        prompt=prompt,
        image=img.resize((512, 512)),
        mask_image=msk.resize((512, 512)),
        height=512, width=512,
        guidance_scale=30, num_inference_steps=50,
        generator=torch.Generator("cuda").manual_seed(0),
    ).images[0]

    result = result.resize(img.size)
    result_path = Path(f"/tmp/{uid}_out.png")
    result.save(result_path)

    return {
        "result_url": f"/static/{result_path.name}"
    }
