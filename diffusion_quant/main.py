import fire
import copy
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image, ImageDraw, ImageFont
import lpips
import os
from datetime import datetime
from tqdm import tqdm
import csv
import shutil

from torchao.quantization import (
    quantize_,
    Float8WeightOnlyConfig,
    FqnToConfig,
)

# -----------------------------
# Config
# -----------------------------
MODEL_ID = "OFA-Sys/small-stable-diffusion-v0"
PROMPT = "a small cozy cabin in the snowy mountains at sunset, high detail"
IMAGE_SIZE = (512, 512)  # (width, height)
OUTPUT_DIR = "diffusion_quant/outputs"
RANDOM_SEED = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)


def log(message: str):
    """Print a timestamped log message."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")


def print_pipeline_architecture(pipe: StableDiffusionPipeline):
    """
    Print the PyTorch model architecture for each component of a StableDiffusionPipeline.

    Args:
        pipe: The StableDiffusionPipeline to inspect
    """
    print("\n" + "=" * 80)
    print("STABLE DIFFUSION PIPELINE COMPONENTS")
    print("=" * 80)

    # The main neural network components in a StableDiffusionPipeline are:
    # 1. vae (Variational Autoencoder) - encodes/decodes images
    # 2. unet (U-Net) - the main denoising model
    # 3. text_encoder - encodes text prompts
    # 4. Other components: tokenizer, scheduler (not nn.Module)

    print("\n" + "-" * 80)
    print("1. VAE (Variational Autoencoder)")
    print("-" * 80)
    print(pipe.vae)
    print(f"\nVAE Parameter Count: {sum(p.numel() for p in pipe.vae.parameters()):,}")

    print("\n" + "-" * 80)
    print("2. U-Net (Main Denoising Model)")
    print("-" * 80)
    print(pipe.unet)
    print(
        f"\nU-Net Parameter Count: {sum(p.numel() for p in pipe.unet.parameters()):,}"
    )

    print("\n" + "-" * 80)
    print("3. Text Encoder (CLIP)")
    print("-" * 80)
    print(pipe.text_encoder)
    print(
        f"\nText Encoder Parameter Count: {sum(p.numel() for p in pipe.text_encoder.parameters()):,}"
    )

    print("\n" + "-" * 80)
    print("4. Other Components (Non-Neural)")
    print("-" * 80)
    print(f"Tokenizer: {type(pipe.tokenizer).__name__}")
    print(f"Scheduler: {type(pipe.scheduler).__name__}")

    print("\n" + "=" * 80)
    total_params = (
        sum(p.numel() for p in pipe.vae.parameters())
        + sum(p.numel() for p in pipe.unet.parameters())
        + sum(p.numel() for p in pipe.text_encoder.parameters())
    )
    print(f"TOTAL PARAMETERS: {total_params:,}")
    print("=" * 80 + "\n")


def generate_image(
    pipe: StableDiffusionPipeline, prompt: str, seed: int, device: str
) -> Image.Image:
    """
    Generate a single image from a prompt and seed, and return it.

    Args:
        pipe: The StableDiffusionPipeline to use for generation
        prompt: Text prompt for image generation
        seed: Random seed for reproducibility
        device: Device string ('cuda' or 'cpu') for the generator
    """
    generator = torch.Generator(device=device).manual_seed(seed)
    image = pipe(
        prompt,
        num_inference_steps=25,  # can tweak for speed vs quality
        guidance_scale=7.5,
        generator=generator,
    ).images[0]

    # Resize (if needed) to a fixed size so LPIPS sees consistent shapes
    if IMAGE_SIZE is not None:
        image = image.resize(IMAGE_SIZE, Image.BICUBIC)

    return image


def create_comparison_image(
    baseline_img: Image.Image,
    modified_img: Image.Image,
    lpips_score: float,
    margin_top: int = 50,
) -> Image.Image:
    """
    Create a comparison image by stacking two images horizontally with a top margin
    and overlaying the LPIPS score.
    """
    # Get dimensions
    width1, height1 = baseline_img.size
    width2, height2 = modified_img.size

    # Create new image with top margin
    total_width = width1 + width2
    total_height = max(height1, height2) + margin_top

    # Create composite image with dark gray background for margin
    composite = Image.new("RGB", (total_width, total_height), color=(50, 50, 50))

    # Paste the two images side by side, offset by margin_top
    composite.paste(baseline_img, (0, margin_top))
    composite.paste(modified_img, (width1, margin_top))

    # Add text overlay with LPIPS score
    draw = ImageDraw.Draw(composite)

    # Try to use a reasonable font size, fallback to default if truetype fails
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 32)
    except Exception:
        try:
            font = ImageFont.truetype("arial.ttf", 32)
        except Exception:
            font = ImageFont.load_default()

    # Format the text
    text = f"LPIPS: {lpips_score:.4f}"

    # Get text bounding box for centering
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    # Center the text horizontally in the margin area
    text_x = (total_width - text_width) // 2
    text_y = (margin_top - text_height) // 2

    # Draw text in white
    draw.text((text_x, text_y), text, fill=(255, 255, 255), font=font)

    return composite


def pil_to_lpips_tensor(img: Image.Image, device: str):
    """
    Convert a PIL Image to a tensor suitable for LPIPS computation.

    Args:
        img: PIL Image to convert
        device: Device to place the tensor on ('cuda' or 'cpu')

    Returns:
        Tensor in shape (1, 3, H, W) normalized to [-1, 1]
    """
    t = (
        torch.from_numpy(
            (
                torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
                .view(img.size[1], img.size[0], 3)
                .numpy()
            )
        ).float()
        / 255.0
    )  # [0, 1]
    # reshape to (1, 3, H, W) and scale to [-1, 1]
    t = t.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
    t = t * 2.0 - 1.0
    return t.to(device)


def run(mode: str):
    """
    Main execution function: generates baseline and modified images,
    computes LPIPS, and creates a comparison visualization.
    """
    assert mode in ("sweep", "use_sweep_results", "test"), f"unsupported {mode=}"

    log("Starting diffusion quantization experiment")

    # Set seeds for reproducibility
    import random
    import numpy as np

    log("Setting random seeds for reproducibility")
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # -----------------------------
    # 1. Load model from HuggingFace
    # -----------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"Using device: {device}")

    log(f"Loading model from HuggingFace: {MODEL_ID}")
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        # torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        # torch_dtype=torch.bfloat16,
        torch_dtype=torch.bfloat16,
    )
    log("Moving model to device")
    pipe = pipe.to(device)
    # return

    # Print model architecture for inspection
    log("Printing pipeline architecture")
    # print_pipeline_architecture(pipe)

    # Optional: enable memory optimizations
    log("Enabling attention slicing for memory optimization")
    pipe.enable_attention_slicing()

    loss_fn = lpips.LPIPS(net="vgg").to(device)

    # -----------------------------
    # 2. Baseline image
    # -----------------------------
    log("Generating baseline image")
    baseline_img = generate_image(pipe, PROMPT, RANDOM_SEED, device)
    baseline_t = pil_to_lpips_tensor(baseline_img, device)

    # -----------------------------
    # Inspect Linear layers in U-Net
    # -----------------------------
    log("Inspecting Linear layers in U-Net")
    unet_linear_fqns_and_weight_shapes = []
    for fqn, module in pipe.unet.named_modules():
        if isinstance(module, torch.nn.Linear):
            weight_shape = module.weight.shape
            print(f"  {fqn}: {weight_shape}")
            unet_linear_fqns_and_weight_shapes.append([fqn, weight_shape])

    # -----------------------------
    # 3. "Quantized" image
    # -----------------------------
    log("Applying Float8 quantization to U-Net")

    # Create a copy so that we can test multiple experiments vs baseline
    orig_unet = pipe.unet

    # quant_config = Float8DynamicActivationFloat8WeightConfig(granularity=PerRow())
    # quant_config = Float8DynamicActivationFloat8WeightConfig()
    quant_config = Float8WeightOnlyConfig()

    if mode == "sweep":
        # Clear output directory in sweep mode
        log(f"Clearing output directory: {OUTPUT_DIR}")
        if os.path.exists(OUTPUT_DIR):
            shutil.rmtree(OUTPUT_DIR)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        log("Output directory cleared")

        print("sweep")

        # test every single linear and measure impact on LPIPS
        fqn_to_lpips = []
        for fqn, weight_shape in tqdm(
            unet_linear_fqns_and_weight_shapes, desc="Quantizing layers"
        ):
            log(f"{fqn=}")
            fqn_to_config = FqnToConfig(fqn_to_config={fqn: quant_config})

            unet_copy = copy.deepcopy(orig_unet)
            quantize_(unet_copy, fqn_to_config, filter_fn=None)
            pipe.unet = unet_copy

            # print_pipeline_architecture(pipe)
            modified_img = generate_image(pipe, PROMPT, RANDOM_SEED, device)

            # -----------------------------
            # 4. Compute LPIPS between the two
            # -----------------------------
            with torch.no_grad():
                modified_t = pil_to_lpips_tensor(modified_img, device)
                lpips_value = loss_fn(baseline_t, modified_t).item()

            log(f"LPIPS distance: {lpips_value:.4f}")

            # -----------------------------
            # 5. Create and save comparison image
            # -----------------------------
            comparison_img = create_comparison_image(
                baseline_img, modified_img, lpips_value
            )
            comparison_path = os.path.join(OUTPUT_DIR, f"comparison_{fqn}.png")
            comparison_img.save(comparison_path)
            log(f"Saved comparison image to: {comparison_path}")

            fqn_to_lpips.append((fqn, lpips_value))

            # clean up
            pipe.unet = orig_unet
            del unet_copy

        for fqn, lpips_value in fqn_to_lpips:
            print(fqn, lpips_value)

        # Save results to CSV
        csv_path = os.path.join(OUTPUT_DIR, "fqn_to_lpips.csv")
        log(f"Saving results to {csv_path}")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["fqn", "lpips"])
            for fqn, lpips_value in fqn_to_lpips:
                writer.writerow([fqn, lpips_value])
        log(f"Results saved to {csv_path}")

    elif mode == "use_sweep_results":
        lpips_upper_bound = 0.30

        log("Using sweep results to selectively quantize layers")

        # Load CSV file
        csv_path = os.path.join(OUTPUT_DIR, "fqn_to_lpips.csv")
        log(f"Loading results from {csv_path}")

        fqns_to_quantize = []
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                fqn = row["fqn"]
                lpips_value = float(row["lpips"])
                if lpips_value < lpips_upper_bound:
                    fqns_to_quantize.append(fqn)
                    log(f"  {fqn}: {lpips_value:.4f} (below threshold, will quantize)")

        log(f"Found {len(fqns_to_quantize)} layers with LPIPS < {lpips_upper_bound}")

        # Create FqnToConfig mapping only for layers passing the threshold
        fqn_to_config_dict = {}
        for fqn in fqns_to_quantize:
            fqn_to_config_dict[fqn] = quant_config

        fqn_to_config = FqnToConfig(fqn_to_config=fqn_to_config_dict)

        # Quantize the U-Net using this selective config
        unet_copy = copy.deepcopy(orig_unet)
        quantize_(unet_copy, fqn_to_config, filter_fn=None)
        pipe.unet = unet_copy
        # print_pipeline_architecture(pipe)

        # Generate image with selectively quantized model
        modified_img = generate_image(pipe, PROMPT, RANDOM_SEED, device)

        # Compute LPIPS for selectively quantized model
        modified_t = pil_to_lpips_tensor(modified_img, device)

        with torch.no_grad():
            lpips_value = loss_fn(baseline_t, modified_t).item()

        log(f"LPIPS distance (selective quantization): {lpips_value:.4f}")

        # Create and save comparison image
        comparison_img = create_comparison_image(
            baseline_img, modified_img, lpips_value
        )
        comparison_path = os.path.join(
            OUTPUT_DIR, f"comparison_selective_{lpips_upper_bound}.png"
        )
        comparison_img.save(comparison_path)
        log(f"Saved comparison image to: {comparison_path}")

        # Print quantization statistics
        total_linear_layers = len(unet_linear_fqns_and_weight_shapes)
        quantized_layers = len(fqns_to_quantize)
        non_quantized_layers = total_linear_layers - quantized_layers

        log("=" * 80)
        log("Quantization Statistics:")
        log(f"  Total Linear layers: {total_linear_layers}")
        log(f"  Quantized layers: {quantized_layers}")
        log(f"  Non-quantized layers: {non_quantized_layers}")

    elif mode == "test":
        log("Test mode: Quantizing all Linear layers in U-Net")

        # Create FqnToConfig mapping for ALL Linear layers
        fqn_to_config_dict = {}
        for fqn, weight_shape in unet_linear_fqns_and_weight_shapes:
            fqn_to_config_dict[fqn] = quant_config

        fqn_to_config = FqnToConfig(fqn_to_config=fqn_to_config_dict)
        log(f"Created FqnToConfig with all {len(fqn_to_config_dict)} Linear layers")

        # Quantize the U-Net using this config
        log("Applying quantization to all Linear layers in U-Net")
        unet_copy = copy.deepcopy(orig_unet)
        quantize_(unet_copy, fqn_to_config, filter_fn=None)
        pipe.unet = unet_copy
        log("Quantization complete")
        print_pipeline_architecture(pipe)

        # Generate image with fully quantized model
        log("Generating image with fully quantized model")
        modified_img = generate_image(pipe, PROMPT, RANDOM_SEED, device)

        # Compute LPIPS for fully quantized model
        log("Computing LPIPS for fully quantized model")
        modified_t = pil_to_lpips_tensor(modified_img, device)

        with torch.no_grad():
            lpips_value = loss_fn(baseline_t, modified_t).item()

        log(f"LPIPS distance (full quantization): {lpips_value:.4f}")

        # Create and save comparison image
        log("Creating comparison image")
        comparison_img = create_comparison_image(
            baseline_img, modified_img, lpips_value
        )
        comparison_path = os.path.join(OUTPUT_DIR, "comparison_test_full_quant.png")
        comparison_img.save(comparison_path)
        log(f"Saved comparison image to: {comparison_path}")

        # Print summary
        log("=" * 80)
        log("Test Mode Summary:")
        log(f"  Total Linear layers quantized: {len(fqn_to_config_dict)}")
        log(f"  LPIPS distance: {lpips_value:.4f}")
        log("=" * 80)

    else:
        raise AssertionError()

    log("Experiment complete!")


if __name__ == "__main__":
    fire.Fire(run)
