import fire
import copy
import torch
from diffusers import StableDiffusionPipeline, FluxPipeline
from PIL import Image, ImageDraw, ImageFont
import lpips
import os
from datetime import datetime
from tqdm import tqdm
import csv
import shutil

from torchao.quantization import (
    quantize_,
    Float8DynamicActivationFloat8WeightConfig,
    Float8WeightOnlyConfig,
    FqnToConfig,
    PerRow,
)

# import torchao.prototype.mx_formats
from torchao.prototype.mx_formats.inference_workflow import (
    NVFP4DynamicActivationNVFP4WeightConfig,
)

# -----------------------------
# Config
# -----------------------------
MODEL_CONFIGS = {
    "stable-diffusion": {
        "id": "OFA-Sys/small-stable-diffusion-v0",
        "pipeline_class": StableDiffusionPipeline,
        "main_component": "unet",
        "components": ["vae", "unet", "text_encoder"],
    },
    "flux": {
        "id": "black-forest-labs/FLUX.1-dev",
        "pipeline_class": FluxPipeline,
        "main_component": "transformer",
        "components": ["vae", "transformer", "text_encoder", "text_encoder_2"],
    },
}

MODEL_ID = "OFA-Sys/small-stable-diffusion-v0"  # Legacy - deprecated
PROMPTS = [
    "a small cozy cabin in the snowy mountains at sunset, high detail",
    "a wizard doing magic",
    "a enthusiast fisherman in the middle of a lake",
    "an unhappy wolf",
    "striker lining up for a penalty kick",
    "a person enjoying their morning coffee",
    "a robot playing basketball",
    "beautiful flowers in a mountain meadow",
    "a supermarket shelf full of sale items",
    "a yellow taxi",
]
IMAGE_SIZE = (512, 512)  # (width, height)
OUTPUT_DIR = "diffusion_quant/outputs"
RANDOM_SEED = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)


def log(message: str):
    """Print a timestamped log message."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")


def print_pipeline_architecture(pipe, model_config: dict):
    """
    Print the PyTorch model architecture for each component of a diffusion pipeline.

    Args:
        pipe: The diffusion pipeline to inspect
        model_config: Model configuration dict specifying components
    """
    print("\n" + "=" * 80)
    print("DIFFUSION PIPELINE COMPONENTS")
    print("=" * 80)

    # Iterate through components specified in the model config
    total_params = 0
    for idx, component_name in enumerate(model_config["components"], 1):
        component = getattr(pipe, component_name)
        print("\n" + "-" * 80)
        print(f"{idx}. {component_name.upper().replace('_', ' ')}")
        print("-" * 80)
        print(component)
        param_count = sum(p.numel() for p in component.parameters())
        print(f"\n{component_name} Parameter Count: {param_count:,}")
        total_params += param_count

    print("\n" + "-" * 80)
    print("Other Components (Non-Neural)")
    print("-" * 80)
    print(f"Tokenizer: {type(pipe.tokenizer).__name__}")
    print(f"Scheduler: {type(pipe.scheduler).__name__}")

    print("\n" + "=" * 80)
    print(f"TOTAL PARAMETERS: {total_params:,}")
    print("=" * 80 + "\n")


def generate_image(pipe, prompt: str, seed: int, device: str) -> Image.Image:
    """
    Generate a single image from a prompt and seed, and return it.

    Args:
        pipe: The diffusion pipeline to use for generation
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


def create_combined_comparison_image(
    comparison_images: list[Image.Image],
) -> Image.Image:
    """
    Stack multiple comparison images vertically into a single combined image.

    Args:
        comparison_images: List of comparison images to stack vertically

    Returns:
        Combined image with all comparisons stacked vertically
    """
    if not comparison_images:
        raise ValueError("comparison_images list cannot be empty")

    # Calculate dimensions
    total_height = sum(img.size[1] for img in comparison_images)
    max_width = max(img.size[0] for img in comparison_images)

    # Create combined image
    combined_img = Image.new("RGB", (max_width, total_height))
    y_offset = 0
    for comp_img in comparison_images:
        combined_img.paste(comp_img, (0, y_offset))
        y_offset += comp_img.size[1]

    return combined_img


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


def run(mode: str, model: str = "stable-diffusion", num_prompts: int = None):
    """
    Main execution function: generates baseline and modified images,
    computes LPIPS, and creates a comparison visualization.

    Args:
        mode: One of 'sweep', 'use_sweep_results', or 'test'
        model: Model to use ('stable-diffusion' or 'flux')
        num_prompts: Optional limit on number of prompts to use (for debugging)
    """
    assert mode in ("sweep", "use_sweep_results", "test"), f"unsupported {mode=}"
    assert model in MODEL_CONFIGS, (
        f"unsupported {model=}, choose from {list(MODEL_CONFIGS.keys())}"
    )

    # Get model configuration
    model_config = MODEL_CONFIGS[model]

    log("Starting diffusion quantization experiment")
    log(f"Model: {model} ({model_config['id']})")

    # Create model-specific output directory
    output_dir = os.path.join(OUTPUT_DIR, model)
    os.makedirs(output_dir, exist_ok=True)

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

    log(f"Loading model from HuggingFace: {model_config['id']}")
    pipeline_class = model_config["pipeline_class"]
    pipe = pipeline_class.from_pretrained(
        model_config["id"],
        # torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        # torch_dtype=torch.bfloat16,
        torch_dtype=torch.bfloat16,
    )
    log("Moving model to device")
    pipe = pipe.to(device)
    # return

    # Print model architecture for inspection
    log("Printing pipeline architecture")
    # print_pipeline_architecture(pipe, model_config)

    # Optional: enable memory optimizations
    log("Enabling attention slicing for memory optimization")
    pipe.enable_attention_slicing()

    loss_fn = lpips.LPIPS(net="vgg").to(device)

    # -----------------------------
    # 2. Baseline images (for all prompts)
    # -----------------------------
    # Limit prompts for debugging if requested
    prompts_to_use = PROMPTS if num_prompts is None else PROMPTS[:num_prompts]
    log(f"Generating baseline images for {len(prompts_to_use)} prompts")
    baseline_data = []  # List of (prompt_idx, prompt, baseline_img, baseline_t)
    for idx, prompt in enumerate(prompts_to_use):
        prompt_idx = f"prompt_{idx}"
        log(f"Generating baseline for {prompt_idx}: {prompt}")
        baseline_img = generate_image(pipe, prompt, RANDOM_SEED, device)
        baseline_t = pil_to_lpips_tensor(baseline_img, device)
        baseline_data.append((prompt_idx, prompt, baseline_img, baseline_t))
        log(f"  Baseline generated for {prompt_idx}")

    # -----------------------------
    # Inspect Linear layers in main component (U-Net or Transformer)
    # -----------------------------
    main_component_name = model_config["main_component"]
    main_component = getattr(pipe, main_component_name)

    log(f"Inspecting Linear layers in {main_component_name}")
    component_linear_fqns_and_weight_shapes = []
    for fqn, module in main_component.named_modules():
        if isinstance(module, torch.nn.Linear):
            weight_shape = module.weight.shape
            print(f"  {fqn}: {weight_shape}")
            component_linear_fqns_and_weight_shapes.append([fqn, weight_shape])

    # -----------------------------
    # 3. "Quantized" image
    # -----------------------------
    log(f"Applying Float8 quantization to {main_component_name}")

    # Create a copy so that we can test multiple experiments vs baseline
    orig_main_component = main_component

    if False:
        quant_config = Float8WeightOnlyConfig()
    if False:
        quant_config = Float8DynamicActivationFloat8WeightConfig(granularity=PerRow())
    quant_config = NVFP4DynamicActivationNVFP4WeightConfig(
        use_triton_kernel=True, use_dynamic_per_tensor_scale=True
    )

    if mode == "sweep":
        # Clear output directory in sweep mode
        log(f"Clearing output directory: {output_dir}")
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        log("Output directory cleared")

        print("sweep")

        # test every single linear and measure impact on LPIPS
        # Store results: dict mapping fqn to list of lpips values (one per prompt)
        fqn_to_lpips = {}  # {fqn: [lpips_0, lpips_1, ...]}
        for fqn, weight_shape in tqdm(
            component_linear_fqns_and_weight_shapes, desc="Quantizing layers"
        ):
            log(f"{fqn=}")
            fqn_to_config = FqnToConfig(fqn_to_config={fqn: quant_config})

            component_copy = copy.deepcopy(orig_main_component)
            quantize_(component_copy, fqn_to_config, filter_fn=None)
            setattr(pipe, main_component_name, component_copy)
            # print_pipeline_architecture(pipe, model_config)
            # breakpoint()

            # Store LPIPS values for this FQN across all prompts
            lpips_values = []

            # Test on all prompts
            for prompt_idx, prompt, baseline_img, baseline_t in baseline_data:
                # print_pipeline_architecture(pipe)
                modified_img = generate_image(pipe, prompt, RANDOM_SEED, device)

                # -----------------------------
                # 4. Compute LPIPS between the two
                # -----------------------------
                with torch.no_grad():
                    modified_t = pil_to_lpips_tensor(modified_img, device)
                    lpips_value = loss_fn(baseline_t, modified_t).item()

                log(f"LPIPS distance ({prompt_idx}): {lpips_value:.4f}")

                # -----------------------------
                # 5. Create and save comparison image
                # -----------------------------
                comparison_img = create_comparison_image(
                    baseline_img, modified_img, lpips_value
                )
                comparison_path = os.path.join(
                    output_dir, f"comparison_{fqn}_{prompt_idx}.png"
                )
                comparison_img.save(comparison_path)
                log(f"Saved comparison image to: {comparison_path}")

                lpips_values.append(lpips_value)

            # Store normalized: one entry per FQN with list of LPIPS values
            fqn_to_lpips[fqn] = lpips_values

            # clean up
            setattr(pipe, main_component_name, orig_main_component)
            del component_copy

        # Print summary
        for fqn, lpips_values in fqn_to_lpips.items():
            avg_lpips = sum(lpips_values) / len(lpips_values)
            print(f"{fqn}: {lpips_values} (avg={avg_lpips:.4f})")

        # Save results to CSV (normalized format: each prompt_idx as column)
        csv_path = os.path.join(output_dir, "fqn_to_lpips.csv")
        log(f"Saving results to {csv_path}")

        # Create header with prompt_idx columns based on prompts actually used
        num_prompts_used = len(prompts_to_use)
        header = ["fqn"] + [f"prompt_{idx}" for idx in range(num_prompts_used)]

        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for fqn, lpips_values in fqn_to_lpips.items():
                row = [fqn] + lpips_values
                writer.writerow(row)
        log(f"Results saved to {csv_path}")

    elif mode == "use_sweep_results":
        lpips_avg_upper_bound = 0.14
        lpips_max_upper_bound = 0.30

        log("Using sweep results to selectively quantize layers")
        log(
            f"Thresholds: avg LPIPS < {lpips_avg_upper_bound}, max LPIPS < {lpips_max_upper_bound}"
        )

        # Load CSV file and group by FQN (aggregate across prompts)
        csv_path = os.path.join(output_dir, "fqn_to_lpips.csv")
        log(f"Loading results from {csv_path}")

        # Compute average and max LPIPS per FQN across all prompts - read normalized format
        fqn_to_lpips = {}  # {fqn: [lpips_0, lpips_1, ...]}
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                fqn = row["fqn"]
                # Extract LPIPS values from prompt columns
                lpips_values = []
                idx = 0
                while f"prompt_{idx}" in row:
                    lpips_values.append(float(row[f"prompt_{idx}"]))
                    idx += 1
                fqn_to_lpips[fqn] = lpips_values

        # Find FQNs where both average and max LPIPS are below their respective thresholds
        fqns_to_quantize = []
        fqns_rejected_by_avg = []
        fqns_rejected_by_max = []

        for fqn, lpips_values in fqn_to_lpips.items():
            avg_lpips = sum(lpips_values) / len(lpips_values)
            max_lpips = max(lpips_values)

            if avg_lpips < lpips_avg_upper_bound and max_lpips < lpips_max_upper_bound:
                fqns_to_quantize.append(fqn)
                log(
                    f"  ✓ {fqn}: avg={avg_lpips:.4f}, max={max_lpips:.4f} (will quantize)"
                )
            else:
                if avg_lpips >= lpips_avg_upper_bound:
                    fqns_rejected_by_avg.append(fqn)
                    log(
                        f"  ✗ {fqn}: avg={avg_lpips:.4f}, max={max_lpips:.4f} (rejected: avg too high)"
                    )
                else:
                    fqns_rejected_by_max.append(fqn)
                    log(
                        f"  ✗ {fqn}: avg={avg_lpips:.4f}, max={max_lpips:.4f} (rejected: max too high)"
                    )

        log(f"Found {len(fqns_to_quantize)} layers passing both thresholds")
        log(
            f"Rejected {len(fqns_rejected_by_avg)} layers due to high avg, "
            f"{len(fqns_rejected_by_max)} layers due to high max"
        )

        # Create FqnToConfig mapping only for layers passing the threshold
        fqn_to_config_dict = {}
        for fqn in fqns_to_quantize:
            fqn_to_config_dict[fqn] = quant_config

        fqn_to_config = FqnToConfig(fqn_to_config=fqn_to_config_dict)

        # Quantize the main component using this selective config
        component_copy = copy.deepcopy(orig_main_component)
        quantize_(component_copy, fqn_to_config, filter_fn=None)
        setattr(pipe, main_component_name, component_copy)
        print_pipeline_architecture(pipe, model_config)

        # Generate images with selectively quantized model for all prompts
        selective_lpips_values = []
        selective_comparison_images = []
        for prompt_idx, prompt, baseline_img, baseline_t in baseline_data:
            log(f"Generating image for {prompt_idx}")
            modified_img = generate_image(pipe, prompt, RANDOM_SEED, device)

            # Compute LPIPS for selectively quantized model
            modified_t = pil_to_lpips_tensor(modified_img, device)

            with torch.no_grad():
                lpips_value = loss_fn(baseline_t, modified_t).item()

            selective_lpips_values.append(lpips_value)
            log(
                f"LPIPS distance (selective quantization, {prompt_idx}): {lpips_value:.4f}"
            )

            # Create and save comparison image
            comparison_img = create_comparison_image(
                baseline_img, modified_img, lpips_value
            )
            selective_comparison_images.append(comparison_img)
            comparison_path = os.path.join(
                output_dir,
                f"comparison_selective_avg{lpips_avg_upper_bound}_max{lpips_max_upper_bound}_{prompt_idx}.png",
            )
            comparison_img.save(comparison_path)
            log(f"Saved comparison image to: {comparison_path}")

        # Create combined image with all comparisons stacked vertically
        if selective_comparison_images:
            combined_img = create_combined_comparison_image(selective_comparison_images)
            combined_path = os.path.join(
                output_dir,
                f"comparison_selective_avg{lpips_avg_upper_bound}_max{lpips_max_upper_bound}_combined.png",
            )
            combined_img.save(combined_path)
            log(f"Saved combined comparison image to: {combined_path}")

        # Print quantization statistics
        total_linear_layers = len(component_linear_fqns_and_weight_shapes)
        quantized_layers = len(fqns_to_quantize)
        non_quantized_layers = total_linear_layers - quantized_layers

        log("=" * 80)
        log("Quantization Statistics:")
        log(f"  Total Linear layers: {total_linear_layers}")
        log(f"  Quantized layers: {quantized_layers}")
        log(f"  Non-quantized layers: {non_quantized_layers}")
        log("")
        log("LPIPS Results:")
        log(
            f"  Average LPIPS: {sum(selective_lpips_values) / len(selective_lpips_values):.4f}"
        )
        log(f"  Max LPIPS: {max(selective_lpips_values):.4f}")
        log(f"  Min LPIPS: {min(selective_lpips_values):.4f}")
        log(f"  All values: {[f'{v:.4f}' for v in selective_lpips_values]}")
        log("=" * 80)

    elif mode == "test":
        log(f"Test mode: Quantizing all Linear layers in {main_component_name}")

        # Create FqnToConfig mapping for ALL Linear layers
        fqn_to_config_dict = {}
        for fqn, weight_shape in component_linear_fqns_and_weight_shapes:
            fqn_to_config_dict[fqn] = quant_config

        fqn_to_config = FqnToConfig(fqn_to_config=fqn_to_config_dict)
        log(f"Created FqnToConfig with all {len(fqn_to_config_dict)} Linear layers")

        # Quantize the main component using this config
        log(f"Applying quantization to all Linear layers in {main_component_name}")
        component_copy = copy.deepcopy(orig_main_component)
        quantize_(component_copy, fqn_to_config, filter_fn=None)
        setattr(pipe, main_component_name, component_copy)
        log("Quantization complete")
        print_pipeline_architecture(pipe, model_config)

        # Generate images with fully quantized model for all prompts
        log("Generating images with fully quantized model for all prompts")
        test_lpips_values = []
        test_comparison_images = []
        for prompt_idx, prompt, baseline_img, baseline_t in baseline_data:
            log(f"Generating image for {prompt_idx}")
            modified_img = generate_image(pipe, prompt, RANDOM_SEED, device)

            # Compute LPIPS for fully quantized model
            log(f"Computing LPIPS for {prompt_idx}")
            modified_t = pil_to_lpips_tensor(modified_img, device)

            with torch.no_grad():
                lpips_value = loss_fn(baseline_t, modified_t).item()

            test_lpips_values.append(lpips_value)
            log(f"LPIPS distance (full quantization, {prompt_idx}): {lpips_value:.4f}")

            # Create and save comparison image
            log("Creating comparison image")
            comparison_img = create_comparison_image(
                baseline_img, modified_img, lpips_value
            )
            test_comparison_images.append(comparison_img)
            comparison_path = os.path.join(
                output_dir, f"comparison_test_full_quant_{prompt_idx}.png"
            )
            comparison_img.save(comparison_path)
            log(f"Saved comparison image to: {comparison_path}")

        # Create combined image with all comparisons stacked vertically
        if test_comparison_images:
            combined_img = create_combined_comparison_image(test_comparison_images)
            combined_path = os.path.join(
                output_dir, "comparison_test_full_quant_combined.png"
            )
            combined_img.save(combined_path)
            log(f"Saved combined comparison image to: {combined_path}")

        # Print summary
        log("=" * 80)
        log("Test Mode Summary:")
        log(f"  Total Linear layers quantized: {len(fqn_to_config_dict)}")
        log(f"  Prompts tested: {len(baseline_data)}")
        log("")
        log("LPIPS Results:")
        log(f"  Average LPIPS: {sum(test_lpips_values) / len(test_lpips_values):.4f}")
        log(f"  Max LPIPS: {max(test_lpips_values):.4f}")
        log(f"  Min LPIPS: {min(test_lpips_values):.4f}")
        log(f"  All values: {[f'{v:.4f}' for v in test_lpips_values]}")
        log("=" * 80)

    else:
        raise AssertionError()

    log("Experiment complete!")


if __name__ == "__main__":
    fire.Fire(run)
