import torch
from diffusers import FluxFillPipeline
from diffusers.utils import load_image
from PIL import Image
import time
import logging
from pathlib import Path
import argparse


# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# Time tracking helper
def checkpoint(msg, start_time):
    elapsed = time.time() - start_time
    log.info(f"{msg} - {elapsed:.2f}s")
    return time.time()

# Function to process a single image
def process_image(image_path, mask_path, output_dir, pipe, prompt):
    """
    Loads an image and a mask, performs inpainting, and saves the result.

    Args:
        image_path (Path): Path to the input image file.
        mask_path (Path): Path to the corresponding mask file.
        output_dir (Path): Directory where the processed image will be saved.
        pipe (FluxFillPipeline): The loaded FluxFill pipeline.
        prompt (str): The prompt for the inpainting process.
    """
    image_name = image_path.stem
    log.info(f"Processing image: {image_name}")

    try:
        # Load input image
        image = load_image(str(image_path)).convert("RGB") # Ensure RGB format
        w, h = image.size
        log.info(f"Original image size: {w}x{h}")

        # Load mask
        mask_image = load_image(mask_path).convert("L") # Ensure grayscale mask
        # Resize mask to match image size if they don't match
        if mask_image.size != (w, h):
            log.warning(f"Mask size {mask_image.size} does not match image size {image.size}. Resizing mask.")
            mask_image = mask_image.resize((w, h), resample=Image.NEAREST)
        log.info(f"Mask size: {mask_image.size}")

        # Inpainting
        log.info(f"Type of image: {type(image)}, mode: {image.mode}, size: {image.size}")
        log.info(f"Type of mask_image: {type(mask_image)}, mode: {mask_image.mode}, size: {mask_image.size}")

        log.info("Starting inpainting...")
        result = pipe(
            prompt=prompt,
            image=image,
            mask_image=mask_image,
            height=512,
            width=512,
            guidance_scale=30,
            num_inference_steps=50,
            max_sequence_length=512,
            generator=torch.Generator("cuda").manual_seed(0),
        ).images[0]
        log.info("Inpainting complete.")

        # Resize result back to original size
        result = result.resize((w, h), resample=Image.LANCZOS)
        log.info(f"Resized output image to: {w}x{h}")

        # Save output
        output_path = output_dir / f"{image_name}_processed.png"
        result.save(output_path)
        log.info(f"Saved output image to: {output_path}")

    except Exception as e:
        log.error(f"Error processing image {image_path}: {e}")

def main(args):
    """
    Main function to handle arguments, load model, and process images.
    """
    start = time.time()
    log.info("Starting watermark removal script")

    # Create output directory if it doesn't exist
    args.output_path.mkdir(parents=True, exist_ok=True)

    # Load FLUX pipeline
    t1 = checkpoint("Loading FLUX pipeline", start)
    try:
        pipe = FluxFillPipeline.from_pretrained("black-forest-labs/FLUX.1-Fill-dev", torch_dtype=torch.bfloat16)
        pipe.enable_model_cpu_offload()
        t2 = checkpoint("Pipeline loaded", t1)
    except Exception as e:
        log.error(f"Error loading FLUX pipeline: {e}")
        return # Exit if pipeline loading fails

    # Get list of image files in the data directory
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')
    image_files = [f for f in args.data_path.iterdir() if f.is_file() and f.suffix.lower() in image_extensions]

    if not image_files:
        log.warning(f"No image files found in {args.data_path}")
        return

    log.info(f"Found {len(image_files)} images to process.")

    # Process each image file
    for image_path in image_files:
        image_name = image_path.stem
        
        # Construct expected mask filename (assuming mask has the same base name)
        mask_filename = "mask_modified.png" # Adjust if mask naming convention is different
        mask_path = args.mask_path / mask_filename

        # Check if the corresponding mask exists
        if mask_path.is_file():
            process_image(image_path, mask_path, args.output_path, pipe, args.prompt)
        else:
            log.warning(f"No corresponding mask found for {image_name} at {mask_path}. Skipping.")

    checkpoint("Script completed", t2) # Use t2 as the last successful checkpoint

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog='Watermark Remover',
                    description='Removes Watermark from images in a directory using FluxFill based on masks.'
                    )
    parser.add_argument("--data_path", 
                        type=Path, 
                        default="/home/alessandro/workspace/diffusers/304779.008510160004.104/",
                        help="Root Path to the directory containing input images.")
    parser.add_argument("--mask_path", 
                        type=Path, 
                        default="/home/alessandro/workspace/diffusers/inferece_scripts/watermark_remover/data/masks",
                        help="Root Path to the directory containing corresponding mask images.")
    parser.add_argument("--output_path", 
                        type=Path, 
                        default="/home/alessandro/workspace/diffusers/inferece_scripts/watermark_remover/data/processed_images",
                        help="Root Path where processed images will be saved.")
    parser.add_argument("--prompt",
                        type=str,
                        default="a clean product image with no watermark",
                        help="Prompt to guide the inpainting process.")
    args = parser.parse_args()
    main(args)