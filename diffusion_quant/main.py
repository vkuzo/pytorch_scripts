import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import lpips
import os

# -----------------------------
# Config
# -----------------------------
MODEL_ID = "OFA-Sys/small-stable-diffusion-v0"
PROMPT = "a small cozy cabin in the snowy mountains at sunset, high detail"
IMAGE_SIZE = (512, 512)  # (width, height)
OUTPUT_DIR = "diffusion_quant/outputs"
BASELINE_SEED = 42
MODIFIED_SEED = 43  # using different seed to simulate a changed model

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# 1. Load model from HuggingFace
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
)
pipe = pipe.to(device)

# Optional: enable memory optimizations
pipe.enable_attention_slicing()


def generate_image(prompt: str, seed: int, out_path: str) -> Image.Image:
    """
    Generate a single image from a prompt and seed, save to disk, and return it.
    """
    generator = torch.manual_seed(seed)
    image = pipe(
        prompt,
        num_inference_steps=25,   # can tweak for speed vs quality
        guidance_scale=7.5,
        generator=generator,
    ).images[0]

    # Resize (if needed) to a fixed size so LPIPS sees consistent shapes
    if IMAGE_SIZE is not None:
        image = image.resize(IMAGE_SIZE, Image.BICUBIC)

    image.save(out_path)
    print(f"Saved image to: {out_path}")
    return image


# -----------------------------
# 2. Baseline image
# -----------------------------
baseline_path = os.path.join(OUTPUT_DIR, "baseline.png")
baseline_img = generate_image(PROMPT, BASELINE_SEED, baseline_path)

# -----------------------------
# 3. "Quantized" image via new seed
# -----------------------------
modified_path = os.path.join(OUTPUT_DIR, "modified.png")
modified_img = generate_image(PROMPT, MODIFIED_SEED, modified_path)

# -----------------------------
# 4. Compute LPIPS between the two
# -----------------------------
# LPIPS expects normalized tensors in [-1, 1]
def pil_to_lpips_tensor(img: Image.Image, device: str):
    t = torch.from_numpy(
        (torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
         .view(img.size[1], img.size[0], 3)
         .numpy())
    ).float() / 255.0  # [0, 1]
    # reshape to (1, 3, H, W) and scale to [-1, 1]
    t = t.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
    t = t * 2.0 - 1.0
    return t.to(device)


loss_fn = lpips.LPIPS(net='vgg').to(device)

baseline_t = pil_to_lpips_tensor(baseline_img, device)
modified_t = pil_to_lpips_tensor(modified_img, device)

with torch.no_grad():
    lpips_value = loss_fn(baseline_t, modified_t).item()

print(f"LPIPS distance between baseline and modified images: {lpips_value:.4f}")
