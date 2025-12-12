import fire
import copy
import time
import torch
from diffusers import StableDiffusionPipeline, FluxPipeline, Flux2Pipeline
from PIL import Image, ImageDraw, ImageFont
import lpips
import os
from datetime import datetime
from tqdm import tqdm
import csv
import shutil
import gc

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
    "flux-2.dev": {
        "id": "black-forest-labs/FLUX.2-dev",
        "pipeline_class": Flux2Pipeline,
        "main_component": "transformer",
        "components": ["vae", "transformer", "text_encoder"],
    },
}

IMAGE_SIZE = (512, 512)  # (width, height)
OUTPUT_DIR = "diffusion_quant/outputs"
RANDOM_SEED = 42
PROMPTS_FILES = {
    "calibration": "diffusion_quant/prompts_calibrate.txt",
    "test": "diffusion_quant/prompts_test.txt",
}

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_prompts(prompts_file: str) -> list[str]:
    """Load prompts from a text file, one prompt per line."""
    with open(prompts_file, "r") as f:
        prompts = [line.strip() for line in f if line.strip()]
    return prompts


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
        prompt=prompt,
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
    prompt: str = None,
    margin_top: int = 80,
) -> Image.Image:
    """
    Create a comparison image by stacking two images horizontally with a top margin
    and overlaying the prompt text and LPIPS score.

    Args:
        baseline_img: The baseline image
        modified_img: The modified/quantized image
        lpips_score: The LPIPS score between the two images
        prompt: Optional prompt text to display at the top
        margin_top: Height of the top margin for text (default 80 to fit prompt + LPIPS)
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

    # Add text overlay with prompt and LPIPS score
    draw = ImageDraw.Draw(composite)

    # Try to use reasonable font sizes, fallback to default if truetype fails
    try:
        prompt_font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20
        )
        lpips_font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24
        )
    except Exception:
        try:
            prompt_font = ImageFont.truetype("arial.ttf", 20)
            lpips_font = ImageFont.truetype("arialbd.ttf", 24)
        except Exception:
            prompt_font = ImageFont.load_default()
            lpips_font = ImageFont.load_default()

    # Draw prompt text at the top if provided
    y_offset = 5
    if prompt:
        # Wrap prompt text if it's too long
        max_width = total_width - 20  # 10px padding on each side
        prompt_lines = []
        words = prompt.split()
        current_line = []

        for word in words:
            test_line = " ".join(current_line + [word])
            bbox = draw.textbbox((0, 0), test_line, font=prompt_font)
            line_width = bbox[2] - bbox[0]

            if line_width <= max_width:
                current_line.append(word)
            else:
                if current_line:
                    prompt_lines.append(" ".join(current_line))
                current_line = [word]

        if current_line:
            prompt_lines.append(" ".join(current_line))

        # Draw each line of the prompt
        for line in prompt_lines:
            bbox = draw.textbbox((0, 0), line, font=prompt_font)
            text_width = bbox[2] - bbox[0]
            text_x = (total_width - text_width) // 2
            draw.text((text_x, y_offset), line, fill=(200, 200, 200), font=prompt_font)
            y_offset += (bbox[3] - bbox[1]) + 2  # line height + small gap

    # Format the LPIPS text
    lpips_text = f"LPIPS: {lpips_score:.4f}"

    # Get text bounding box for centering
    bbox = draw.textbbox((0, 0), lpips_text, font=lpips_font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    # Center the LPIPS text horizontally, place it below the prompt
    text_x = (total_width - text_width) // 2
    text_y = y_offset + 5  # small gap after prompt

    # Draw LPIPS text in white
    draw.text((text_x, text_y), lpips_text, fill=(255, 255, 255), font=lpips_font)

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


def run(
    mode: str,
    model: str = "stable-diffusion",
    num_prompts: int = None,
    prompt_set: str = "calibration",
    quant_config: str = "f8d",
    use_compile: bool = False,
    benchmark_performance: bool = False,
):
    """
    Main execution function: generates baseline and modified images,
    computes LPIPS, and creates a comparison visualization.

    Args:
        mode: One of 'sweep', 'use_sweep_results', or 'full_quant'
        model: Model to use ('stable-diffusion' or 'flux' or 'flux-2.dev')
        num_prompts: Optional limit on number of prompts to use (for debugging)
        prompt_set: Which prompt set to use ('calibration' or 'test')
        quant_config: Quantization config to use ('nvfp4', 'f8d', 'f8wo'). Default: 'f8d'
        use_compile: if true, uses torch.compile
        benchmark_performance: if true, benchmarks performance
    """
    assert mode in ("sweep", "use_sweep_results", "full_quant"), f"unsupported {mode=}"
    assert model in MODEL_CONFIGS, (
        f"unsupported {model=}, choose from {list(MODEL_CONFIGS.keys())}"
    )
    assert prompt_set in PROMPTS_FILES, (
        f"unsupported {prompt_set=}, choose from {list(PROMPTS_FILES.keys())}"
    )
    assert quant_config in ("nvfp4", "f8d", "f8wo"), (
        f"unsupported {quant_config=}, choose from ['nvfp4', 'f8d', 'f8wo']"
    )
    if mode == "sweep":
        assert not use_compile
        assert not benchmark_performance

    # Enforce calibration prompts for sweep mode
    if mode == "sweep" and prompt_set != "calibration":
        raise ValueError(
            f"sweep mode only supports calibration prompts, got {prompt_set=}"
        )

    # Get model configuration
    model_config = MODEL_CONFIGS[model]

    log("Starting diffusion quantization experiment")
    log(f"Model: {model} ({model_config['id']})")
    log(f"Prompt set: {prompt_set}")

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
    # print_pipeline_architecture(pipe, model_config)

    # Optional: enable memory optimizations
    log("Enabling attention slicing for memory optimization")
    pipe.enable_attention_slicing()

    loss_fn = lpips.LPIPS(net="vgg").to(device)

    # For large models like flux-2.dev, avoid deep copying the entire pipeline
    # to prevent OOM errors. Instead, use the original pipe for baseline generation.
    # We'll compile the main component in-place if requested.
    main_component_name = model_config["main_component"]
    main_component = getattr(pipe, main_component_name)

    # Store original for restoration later
    orig_main_component = main_component

    if use_compile:
        log("Compiling main component for baseline generation")
        compiled_component = torch.compile(main_component)
        setattr(pipe, main_component_name, compiled_component)

    # -----------------------------
    # 2. Baseline images (for all prompts)
    # -----------------------------
    # Load prompts from file
    prompts_file = PROMPTS_FILES[prompt_set]
    all_prompts = load_prompts(prompts_file)
    log(f"Loaded {len(all_prompts)} prompts from {prompts_file}")

    # Limit prompts for debugging if requested
    prompts_to_use = all_prompts if num_prompts is None else all_prompts[:num_prompts]
    log(f"Generating baseline images for {len(prompts_to_use)} prompts")
    baseline_data = []  # List of (prompt_idx, prompt, baseline_img, baseline_t)
    baseline_times = []
    for idx, prompt in enumerate(prompts_to_use):
        prompt_idx = f"prompt_{idx}"
        log(f"Generating baseline for {prompt_idx}: {prompt}")
        t0 = time.time()
        baseline_img = generate_image(pipe, prompt, RANDOM_SEED, device)
        t1 = time.time()
        baseline_t = pil_to_lpips_tensor(baseline_img, device)
        baseline_data.append((prompt_idx, prompt, baseline_img, baseline_t))
        baseline_times.append(t1 - t0)
        log(f"  Baseline generated for {prompt_idx}")

    # Restore original main component before quantization if it was compiled
    if use_compile:
        log("Restoring original (uncompiled) main component before quantization")
        setattr(pipe, main_component_name, orig_main_component)

    # -----------------------------
    # Inspect Linear layers in main component (U-Net or Transformer)
    # -----------------------------
    # Get fresh reference to main component (in case it was compiled/restored)
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
    log(f"Applying quantization to {main_component_name}")
    log(f"Using quantization config: {quant_config}")

    # Map quant_config string to actual config class
    if quant_config == "nvfp4":
        config_obj = NVFP4DynamicActivationNVFP4WeightConfig(
            use_triton_kernel=True, use_dynamic_per_tensor_scale=True
        )
    elif quant_config == "f8d":
        config_obj = Float8DynamicActivationFloat8WeightConfig(granularity=PerRow())
    elif quant_config == "f8wo":
        config_obj = Float8WeightOnlyConfig()
    else:
        raise AssertionError(f"Unsupported quant_config: {quant_config}")

    # Create a copy so that we can test multiple experiments vs baseline
    orig_main_component = main_component

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

            if model == 'flux-2.dev':
                # hardcode don't quantize layers we def don't want to quantize
                if 'embed' in fqn:
                    print('skipping for embed')
                    continue
                elif 'modulation' in fqn:
                    print('skipping for modulation')
                    continue
                elif weight_shape[0] < 1000 or weight_shape[1] < 1000:
                    print('skipping for weight shape')
                    continue

            fqn_to_config = FqnToConfig(fqn_to_config={fqn: config_obj})

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
                    baseline_img, modified_img, lpips_value, prompt=prompt
                )
                comparison_path = os.path.join(
                    output_dir, f"comparison_{prompt_set}_{fqn}_{prompt_idx}.png"
                )
                comparison_img.save(comparison_path)
                log(f"Saved comparison image to: {comparison_path}")

                lpips_values.append(lpips_value)

            # Store normalized: one entry per FQN with list of LPIPS values
            fqn_to_lpips[fqn] = lpips_values

            # clean up
            setattr(pipe, main_component_name, orig_main_component)
            del component_copy
            # very basic gpu mem management, TODO improve me
            torch.cuda.synchronize()
            gc.collect()
            torch.cuda.empty_cache()

        # Print summary
        for fqn, lpips_values in fqn_to_lpips.items():
            avg_lpips = sum(lpips_values) / len(lpips_values)
            print(f"{fqn}: {lpips_values} (avg={avg_lpips:.4f})")

        # Save results to CSV (normalized format: each prompt_idx as column)
        csv_path = os.path.join(output_dir, f"fqn_to_lpips_{prompt_set}.csv")
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
        lpips_avg_upper_bound = 0.0175
        lpips_max_upper_bound = 0.06

        log("Using sweep results to selectively quantize layers")
        log(
            f"Thresholds: avg LPIPS < {lpips_avg_upper_bound}, max LPIPS < {lpips_max_upper_bound}"
        )

        # Load CSV file and group by FQN (aggregate across prompts)
        # note: the sweep results are always from calibration data, even if we are testing on the
        # `test` prompt set
        csv_path = os.path.join(output_dir, "fqn_to_lpips_calibration.csv")
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

        # TODO custom rules for flux-2.dev here
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
            fqn_to_config_dict[fqn] = config_obj

        fqn_to_config = FqnToConfig(fqn_to_config=fqn_to_config_dict)

        # Quantize the main component using this selective config
        component_copy = copy.deepcopy(orig_main_component)
        quantize_(component_copy, fqn_to_config, filter_fn=None)
        setattr(pipe, main_component_name, component_copy)
        if use_compile:
            setattr(pipe, main_component_name, torch.compile(component_copy))
        print_pipeline_architecture(pipe, model_config)

        # Generate images with selectively quantized model for all prompts
        selective_lpips_values = []
        selective_comparison_images = []
        times = []
        for prompt_idx, prompt, baseline_img, baseline_t in baseline_data:
            log(f"Generating image for {prompt_idx}")
            t0 = time.time()
            modified_img = generate_image(pipe, prompt, RANDOM_SEED, device)
            t1 = time.time()
            times.append(t1 - t0)

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
                baseline_img, modified_img, lpips_value, prompt=prompt
            )
            selective_comparison_images.append(comparison_img)
            comparison_path = os.path.join(
                output_dir,
                f"comparison_prompt_{prompt_set}_mode_use_sweep_results_avg_{lpips_avg_upper_bound}_max_{lpips_max_upper_bound}_quant_config_{quant_config}_{prompt_idx}.png",
            )
            comparison_img.save(comparison_path)
            log(f"Saved comparison image to: {comparison_path}")

        # Create combined image with all comparisons stacked vertically
        if selective_comparison_images:
            combined_img = create_combined_comparison_image(selective_comparison_images)
            combined_path = os.path.join(
                output_dir,
                f"comparison_prompt_{prompt_set}_mode_use_sweep_results_avg_{lpips_avg_upper_bound}_max_{lpips_max_upper_bound}_quant_config_{quant_config}_combined.png",
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
        avg_lpips = sum(selective_lpips_values) / len(selective_lpips_values)
        max_lpips = max(selective_lpips_values)
        min_lpips = min(selective_lpips_values)
        log(f"  Average LPIPS: {avg_lpips:.4f}")
        log(f"  Max LPIPS: {max_lpips:.4f}")
        log(f"  Min LPIPS: {min_lpips:.4f}")
        log(f"  All values: {[f'{v:.4f}' for v in selective_lpips_values]}")
        log("=" * 80)
        print('baseline_times', baseline_times)
        print('times', times)
        print('speedups', [x / y for (x, y) in zip(baseline_times, times)])

        # Save summary stats to CSV
        summary_csv_path = os.path.join(
            output_dir,
            f"summary_stats_prompt_{prompt_set}_mode_use_sweep_results_quant_config_{quant_config}.csv",
        )
        log(f"Saving summary stats to {summary_csv_path}")
        with open(summary_csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "value"])
            writer.writerow(["mode", "use_sweep_results"])
            writer.writerow(["lpips_avg_upper_bound", f"{lpips_avg_upper_bound:.4f}"])
            writer.writerow(["lpips_max_upper_bound", f"{lpips_max_upper_bound:.4f}"])
            writer.writerow(["total_linear_layers", total_linear_layers])
            writer.writerow(["quantized_layers", quantized_layers])
            writer.writerow(["non_quantized_layers", non_quantized_layers])
            writer.writerow(["prompts_tested", len(baseline_data)])
            writer.writerow(["average_lpips", f"{avg_lpips:.4f}"])
            writer.writerow(["max_lpips", f"{max_lpips:.4f}"])
            writer.writerow(["min_lpips", f"{min_lpips:.4f}"])
            # Write individual LPIPS values
            for idx, val in enumerate(selective_lpips_values):
                writer.writerow([f"lpips_prompt_{idx}", f"{val:.4f}"])
        log(f"Summary stats saved to {summary_csv_path}")

    elif mode == "full_quant":
        log(f"full_quant mode: Quantizing all Linear layers in {main_component_name}")

        # Create FqnToConfig mapping for ALL Linear layers
        fqn_to_config_dict = {}
        for fqn, weight_shape in component_linear_fqns_and_weight_shapes:
            if model == 'flux-2.dev':
                # hardcode don't quantize layers we def don't want to quantize
                if 'embed' in fqn:
                    continue
                elif 'modulation' in fqn:
                    continue
                elif weight_shape[0] < 1000 or weight_shape[1] < 1000:
                    continue
            fqn_to_config_dict[fqn] = config_obj

        fqn_to_config = FqnToConfig(fqn_to_config=fqn_to_config_dict)
        log(f"Created FqnToConfig with all {len(fqn_to_config_dict)} Linear layers")

        # Quantize the main component using this config
        log(f"Applying quantization to all Linear layers in {main_component_name}")
        component_copy = copy.deepcopy(orig_main_component)
        quantize_(component_copy, fqn_to_config, filter_fn=None)
        setattr(pipe, main_component_name, component_copy)
        log("Quantization complete")
        if use_compile:
            setattr(pipe, main_component_name, torch.compile(component_copy))
        print_pipeline_architecture(pipe, model_config)

        # Generate images with fully quantized model for all prompts
        log("Generating images with fully quantized model for all prompts")
        test_lpips_values = []
        test_comparison_images = []
        times = []
        for prompt_idx, prompt, baseline_img, baseline_t in baseline_data:
            log(f"Generating image for {prompt_idx}")
            t0 = time.time()
            modified_img = generate_image(pipe, prompt, RANDOM_SEED, device)
            t1 = time.time()
            times.append(t1 - t0)

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
                baseline_img, modified_img, lpips_value, prompt=prompt
            )
            test_comparison_images.append(comparison_img)
            comparison_path = os.path.join(
                output_dir,
                f"comparison_prompt_{prompt_set}_mode_full_quant_config_{quant_config}_{prompt_idx}.png",
            )
            comparison_img.save(comparison_path)
            log(f"Saved comparison image to: {comparison_path}")

        # Create combined image with all comparisons stacked vertically
        if test_comparison_images:
            combined_img = create_combined_comparison_image(test_comparison_images)
            combined_path = os.path.join(
                output_dir,
                f"comparison_prompt_{prompt_set}_mode_full_quant_config_{quant_config}_combined.png",
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
        avg_lpips = sum(test_lpips_values) / len(test_lpips_values)
        max_lpips = max(test_lpips_values)
        min_lpips = min(test_lpips_values)
        log(f"  Average LPIPS: {avg_lpips:.4f}")
        log(f"  Max LPIPS: {max_lpips:.4f}")
        log(f"  Min LPIPS: {min_lpips:.4f}")
        log(f"  All values: {[f'{v:.4f}' for v in test_lpips_values]}")
        log("=" * 80)
        print('baseline_times', baseline_times)
        print('times', times)
        print('speedups', [x / y for (x, y) in zip(baseline_times, times)])

        # Save summary stats to CSV
        summary_csv_path = os.path.join(
            output_dir,
            f"summary_stats_prompt_{prompt_set}_mode_full_quant_config_{quant_config}.csv",
        )
        log(f"Saving summary stats to {summary_csv_path}")
        with open(summary_csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "value"])
            writer.writerow(["mode", "full_quant"])
            writer.writerow(["total_linear_layers_quantized", len(fqn_to_config_dict)])
            writer.writerow(["prompts_tested", len(baseline_data)])
            writer.writerow(["average_lpips", f"{avg_lpips:.4f}"])
            writer.writerow(["max_lpips", f"{max_lpips:.4f}"])
            writer.writerow(["min_lpips", f"{min_lpips:.4f}"])
            # Write individual LPIPS values
            for idx, val in enumerate(test_lpips_values):
                writer.writerow([f"lpips_prompt_{idx}", f"{val:.4f}"])
        log(f"Summary stats saved to {summary_csv_path}")

    else:
        raise AssertionError()

    log("Experiment complete!")


if __name__ == "__main__":
    fire.Fire(run)
