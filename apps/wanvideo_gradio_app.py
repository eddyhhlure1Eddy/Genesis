"""
WanVideo Gradio Application
Text-to-Video generation interface based on Genesis Core

Author: eddy
Date: 2025-11-12
"""

import gradio as gr
import torch
import numpy as np
import random
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import time
from datetime import datetime
import glob

# Fix console encoding for Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Add project root to Python path
current_dir = Path(__file__).parent  # genesis/apps/
project_root = current_dir.parent.parent  # E:\chai fream\
sys.path.insert(0, str(project_root))

# Apply Gradio API fix for boolean schema handling
try:
    from fix_gradio_api import *
except Exception:
    # Inline fallback patch (handles boolean schemas like additionalProperties: false)
    try:
        import gradio_client.utils as _gradio_utils
        _orig__json_schema_to_python_type = _gradio_utils._json_schema_to_python_type
        def _fixed__json_schema_to_python_type(schema, defs=None):
            if isinstance(schema, bool):
                return "Any"
            if schema is None:
                return "None"
            return _orig__json_schema_to_python_type(schema, defs)
        _gradio_utils._json_schema_to_python_type = _fixed__json_schema_to_python_type
        print("[INFO] Applied Gradio API fix for boolean schema handling (inline)")
    except Exception as _e:
        print(f"[WARN] Could not apply Gradio API fix: {str(_e)}")

# Set up environment
os.environ['COMFYUI_PATH'] = str(project_root)

# Import compatibility layers first
sys.path.insert(0, str(project_root / "genesis"))

# Import triton stub before anything else
from genesis.utils import triton_ops_stub

# Import genesis components
from genesis.compat import comfy_stub
from genesis.core import folder_paths_ext

# Setup ComfyUI-WanVideoWrapper as a module
import importlib.util
wrapper_path = project_root / "genesis" / "custom_nodes" / "Comfyui" / "ComfyUI-WanVideoWrapper"
spec = importlib.util.spec_from_file_location(
    "ComfyUI_WanVideoWrapper",
    wrapper_path / "__init__.py"
)
wrapper_module = importlib.util.module_from_spec(spec)
sys.modules['ComfyUI_WanVideoWrapper'] = wrapper_module

# Execute the module to load all nodes
try:
    spec.loader.exec_module(wrapper_module)
    NODE_CLASS_MAPPINGS = getattr(wrapper_module, 'NODE_CLASS_MAPPINGS', {})
    NODE_DISPLAY_NAME_MAPPINGS = getattr(wrapper_module, 'NODE_DISPLAY_NAME_MAPPINGS', {})
    print(f"[INFO] Successfully loaded {len(NODE_CLASS_MAPPINGS)} nodes from ComfyUI-WanVideoWrapper")

    # Check for essential nodes
    essential_nodes = [
        'LoadWanVideoT5TextEncoder',
        'WanVideoTextEncode',
        'WanVideoModelLoader',
        'WanVideoVAELoader',
        'WanVideoEmptyEmbeds',
        'WanVideoSampler',
        'WanVideoDecode'
    ]

    missing_nodes = [node for node in essential_nodes if node not in NODE_CLASS_MAPPINGS]
    if missing_nodes:
        print(f"[WARNING] Missing essential nodes: {missing_nodes}")
    else:
        print(f"[INFO] All essential nodes loaded successfully")

except Exception as e:
    print(f"[ERROR] Failed to load ComfyUI-WanVideoWrapper: {e}")
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}


class WanVideoWorkflow:
    """WanVideo workflow executor"""

    def __init__(self):
        # Get node mappings directly
        self.nodes = NODE_CLASS_MAPPINGS

        # Initialize node instances (check if nodes exist)
        self.t5_encoder = self.nodes.get("LoadWanVideoT5TextEncoder")() if self.nodes.get("LoadWanVideoT5TextEncoder") else None
        self.text_encoder = self.nodes.get("WanVideoTextEncode")() if self.nodes.get("WanVideoTextEncode") else None
        self.model_loader = self.nodes.get("WanVideoModelLoader")() if self.nodes.get("WanVideoModelLoader") else None
        self.vae_loader = self.nodes.get("WanVideoVAELoader")() if self.nodes.get("WanVideoVAELoader") else None
        self.empty_embeds = self.nodes.get("WanVideoEmptyEmbeds")() if self.nodes.get("WanVideoEmptyEmbeds") else None
        self.sampler = self.nodes.get("WanVideoSampler")() if self.nodes.get("WanVideoSampler") else None
        self.decoder = self.nodes.get("WanVideoDecode")() if self.nodes.get("WanVideoDecode") else None

        # Optimization nodes (optional)
        self.lora_selector = self.nodes.get("WanVideoLoraSelect")() if self.nodes.get("WanVideoLoraSelect") else None
        self.compile_settings = self.nodes.get("WanVideoTorchCompileSettings")() if self.nodes.get("WanVideoTorchCompileSettings") else None
        self.block_swap = self.nodes.get("WanVideoBlockSwap")() if self.nodes.get("WanVideoBlockSwap") else None

        # Check required nodes
        if not all([self.t5_encoder, self.text_encoder, self.model_loader, self.vae_loader,
                    self.empty_embeds, self.sampler, self.decoder]):
            missing = []
            if not self.t5_encoder: missing.append("LoadWanVideoT5TextEncoder")
            if not self.text_encoder: missing.append("WanVideoTextEncode")
            if not self.model_loader: missing.append("WanVideoModelLoader")
            if not self.vae_loader: missing.append("WanVideoVAELoader")
            if not self.empty_embeds: missing.append("WanVideoEmptyEmbeds")
            if not self.sampler: missing.append("WanVideoSampler")
            if not self.decoder: missing.append("WanVideoDecode")
            print(f"Missing nodes: {missing}")
            print("Available nodes:", list(self.nodes.keys()))
            raise RuntimeError("Failed to initialize all required nodes")

        self.current_model = None
        self.current_vae = None
        self.current_t5 = None

    def generate_video(
        self,
        # Text parameters
        positive_prompt: str,
        negative_prompt: str,
        # Model parameters
        model_name: str,
        vae_name: str,
        t5_model: str,
        # Generation parameters
        width: int,
        height: int,
        num_frames: int,
        steps: int,
        cfg: float,
        shift: float,
        seed: int,
        # Advanced parameters
        scheduler: str,
        denoise_strength: float,
        quantization: str,
        attention_mode: str,
        # LoRA parameters
        lora_enabled: bool,
        lora_name: str,
        lora_strength: float,
        # Optimization parameters
        compile_enabled: bool,
        compile_backend: str,
        block_swap_enabled: bool,
        blocks_to_swap: int,
        # Output parameters
        output_format: str,
        fps: int,
        progress_callback=None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Execute the video generation workflow"""

        print("\n" + "="*60)
        print("Starting WanVideo Generation")
        print("="*60)
        print(f"Prompt: {positive_prompt[:100]}...")
        print(f"Model: {model_name}")
        print(f"Resolution: {width}x{height}, Frames: {num_frames}")
        print(f"Steps: {steps}, CFG: {cfg}, Seed: {seed}")
        print("="*60)

        metadata = {
            "timestamp": datetime.now().isoformat(),
            "parameters": locals().copy()
        }
        del metadata["parameters"]["self"]
        del metadata["parameters"]["progress_callback"]

        try:
            if progress_callback:
                progress_callback(0.05, "Loading T5 text encoder...")

            # Step 1: Load T5 encoder
            print(f"\n[Step 1] Loading T5 encoder: {t5_model}")
            t5_result = self.t5_encoder.loadmodel(
                model_name=t5_model,
                precision="bf16",  # Use bf16 for precision
                quantization="fp8_e4m3fn"  # Use fp8_e4m3fn for quantization
            )
            t5_encoder = t5_result[0] if t5_result else None
            print(f"  [OK] T5 encoder loaded")

            if progress_callback:
                progress_callback(0.1, "Encoding text prompts...")

            # Step 2: Encode text
            text_embeds_result = self.text_encoder.process(
                positive_prompt=positive_prompt,
                negative_prompt=negative_prompt if negative_prompt else "",
                t5=t5_encoder,
                force_offload=True,
                use_disk_cache=False,
                device="gpu"
            )
            text_embeds = text_embeds_result[0] if text_embeds_result else None

            if progress_callback:
                progress_callback(0.15, "Setting up compile optimization...")

            # Step 3: Setup compile args if enabled
            compile_args = None
            if compile_enabled and self.compile_settings:
                try:
                    compile_result = self.compile_settings.prepare(
                        backend=compile_backend,
                        fullgraph=False,
                        mode="default",
                        dynamic=False,
                        cache_size=64,
                        enabled=True,
                        batch_size=128
                    )
                    compile_args = compile_result[0] if compile_result else None
                except:
                    compile_args = None

            if progress_callback:
                progress_callback(0.2, "Setting up block swap optimization...")

            # Step 4: Setup block swap if enabled
            block_swap_args = None
            if block_swap_enabled and self.block_swap:
                try:
                    swap_result = self.block_swap.prepare(
                        blocks_to_swap=blocks_to_swap,
                        enable_cuda_optimization=True,
                        enable_dram_optimization=True,
                        auto_hardware_tuning=False,
                        vram_threshold_percent=50,
                        num_cuda_streams=8,
                        bandwidth_target=0.8,
                        offload_txt_emb=False,
                        offload_img_emb=False,
                        vace_blocks_to_swap=0,
                        debug_mode=False
                    )
                    block_swap_args = swap_result[0] if swap_result else None
                except:
                    block_swap_args = None

            if progress_callback:
                progress_callback(0.25, "Loading LoRA...")

            # Step 5: Setup LoRA if enabled
            lora = None
            if lora_enabled and lora_name and self.lora_selector:
                try:
                    lora_result = self.lora_selector.select(
                        lora_name=lora_name,
                        strength=lora_strength,
                        merge=False,
                        enabled=True
                    )
                    lora = lora_result[0] if lora_result else None
                except:
                    lora = None

            if progress_callback:
                progress_callback(0.3, "Loading main model...")

            # Step 6: Load main model with optimizations
            model_result = self.model_loader.loadmodel(
                model=model_name,
                base_precision="fp16_fast",
                quantization=quantization if quantization != "disabled" else "fp8_e4m3fn_fast",
                load_device="offload_device",
                attention_mode=attention_mode if attention_mode else "auto"
            )
            model = model_result[0] if model_result else None

            # Apply optimizations to model if available
            if model and compile_enabled:
                try:
                    # Apply torch.compile optimization to the model
                    import torch
                    if hasattr(model, 'model') and hasattr(model.model, 'diffusion_model'):
                        # Compile the diffusion model with specified backend
                        model.model.diffusion_model = torch.compile(
                            model.model.diffusion_model,
                            mode="default",
                            backend=compile_backend,
                            fullgraph=False
                        )
                        print(f"✓ Applied torch.compile with backend: {compile_backend}")
                except Exception as e:
                    print(f"Warning: Could not apply torch.compile: {e}")

            if model and block_swap_enabled:
                try:
                    # Apply block swap optimization for memory management
                    if hasattr(model, 'model') and hasattr(model.model, 'diffusion_model'):
                        # Set up block swapping for memory optimization
                        diffusion_model = model.model.diffusion_model
                        if hasattr(diffusion_model, 'blocks_to_swap'):
                            diffusion_model.blocks_to_swap = blocks_to_swap
                            print(f"✓ Block swap enabled with {blocks_to_swap} blocks")
                        # Additional memory optimization settings
                        if hasattr(diffusion_model, 'enable_memory_efficient_attention'):
                            diffusion_model.enable_memory_efficient_attention()
                except Exception as e:
                    print(f"Warning: Could not apply block swap: {e}")

            if model and lora:
                try:
                    # Apply LoRA if supported
                    pass
                except:
                    pass

            if progress_callback:
                progress_callback(0.4, "Loading VAE...")

            # Step 7: Load VAE
            vae_result = self.vae_loader.loadmodel(
                model_name=vae_name,  # Changed from vae_name to model_name
                precision="bf16"
            )
            vae = vae_result[0] if vae_result else None

            if progress_callback:
                progress_callback(0.45, "Creating image embeddings...")

            # Step 8: Create empty image embeds
            embeds_result = self.empty_embeds.process(
                width=width,
                height=height,
                num_frames=num_frames
            )
            image_embeds = embeds_result[0] if embeds_result else None
            if image_embeds:
                image_embeds["vae"] = vae

            if progress_callback:
                progress_callback(0.5, "Starting generation...")

            # Step 9: Run sampler
            if seed == -1:
                seed = random.randint(0, 2**63 - 1)

            samples_result = self.sampler.process(
                model=model,
                image_embeds=image_embeds,
                steps=steps,
                cfg=cfg,
                shift=shift,  # Use the shift parameter from user
                seed=seed,
                scheduler="unipc",  # Using default scheduler
                riflex_freq_index=0,  # Default to 0 (disabled)
                force_offload=True,
                text_embeds=text_embeds,
                cache_args=compile_args,  # Pass compile args to cache_args
                experimental_args=block_swap_args  # Pass block swap args to experimental_args
            )
            samples = samples_result[0] if samples_result else None

            if progress_callback:
                progress_callback(0.9, "Decoding video...")

            # Step 10: Decode to video
            video_result = self.decoder.decode(
                vae=vae,
                samples=samples,
                enable_vae_tiling=False,
                tile_x=512,
                tile_y=512,
                tile_stride_x=256,
                tile_stride_y=256
            )
            video_frames = video_result[0] if video_result else None

            if progress_callback:
                progress_callback(0.95, "Finalizing output...")

            # Convert to numpy array for display
            if video_frames is not None and hasattr(video_frames, 'shape'):
                if hasattr(video_frames, 'cpu'):
                    video_array = video_frames.cpu().numpy()
                else:
                    video_array = video_frames

                # Ensure correct shape [frames, height, width, channels]
                if len(video_array.shape) == 5:
                    # [batch, frames, channels, height, width] -> [frames, height, width, channels]
                    video_array = video_array[0].transpose(0, 2, 3, 1)
                elif len(video_array.shape) == 4:
                    # [frames, channels, height, width] -> [frames, height, width, channels]
                    if video_array.shape[1] == 3:
                        video_array = video_array.transpose(0, 2, 3, 1)

                # Ensure uint8
                if video_array.max() <= 1.0:
                    video_array = (video_array * 255).astype(np.uint8)
                else:
                    video_array = video_array.astype(np.uint8)
            else:
                # Fallback if no video generated
                video_array = np.zeros((num_frames, height, width, 3), dtype=np.uint8)

            # Save video to project output directory (genesis/output)
            import cv2
            from pathlib import Path

            output_dir = Path(__file__).parent.parent / "output"
            output_dir.mkdir(parents=True, exist_ok=True)
            video_path = str(output_dir / f"wanvideo_{seed}.mp4")

            # Use OpenCV to write video (MP4)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

            for frame in video_array:
                # Convert RGB to BGR for OpenCV
                bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(bgr_frame)

            out.release()

            metadata["output_shape"] = video_array.shape
            metadata["generation_time"] = time.time()
            metadata["output_path"] = video_path

            if progress_callback:
                progress_callback(1.0, "Generation complete!")

            return video_path, video_array, metadata

        except Exception as e:
            import traceback
            error_msg = f"Error during generation: {str(e)}"
            traceback_str = traceback.format_exc()
            print(f"\n{'='*60}")
            print(f"ERROR: {error_msg}")
            print(f"{'='*60}")
            print(traceback_str)
            print(f"{'='*60}\n")
            if progress_callback:
                progress_callback(0, error_msg)
            raise gr.Error(error_msg)


def scan_model_files(directory: str, extensions: List[str] = [".safetensors", ".ckpt", ".pt", ".pth"]) -> List[str]:
    """
    Scan directory for model files

    Args:
        directory: Directory to scan
        extensions: File extensions to look for

    Returns:
        List of model file names
    """
    if not os.path.exists(directory):
        return ["No models found (directory not exists)"]

    models = []
    for ext in extensions:
        pattern = os.path.join(directory, f"**/*{ext}")
        files = glob.glob(pattern, recursive=True)
        for file in files:
            # Get relative path from directory
            rel_path = os.path.relpath(file, directory)
            models.append(rel_path)

    if not models:
        models = ["No models found"]

    return sorted(models)


def get_model_directories():
    """Get default model directories"""
    # Get genesis models directory (E:\chai fream\genesis\models\)
    current_dir = Path(__file__).parent  # genesis/apps/
    genesis_dir = current_dir.parent  # genesis/
    models_root = genesis_dir / "models"

    return {
        "models": str(models_root / "unet"),  # Main models in unet folder
        "vae": str(models_root / "vae"),
        "t5": str(models_root / "clip"),  # T5 models in clip folder
        "lora": str(models_root / "loras")
    }


def create_interface():
    """Create the Gradio interface"""

    workflow = WanVideoWorkflow()
    model_dirs = get_model_directories()

    # Scan for available models
    available_models = scan_model_files(model_dirs["models"])
    available_vaes = scan_model_files(model_dirs["vae"])
    available_t5 = scan_model_files(model_dirs["t5"])
    available_loras = scan_model_files(model_dirs["lora"])

    # Check if models exist, provide better messages
    if not available_models or available_models == ["No models found"]:
        available_models = ["Please place model files in 'models/' directory"]
    if not available_vaes or available_vaes == ["No models found"]:
        available_vaes = ["Please place VAE files in 'models/vae/' directory"]
    if not available_t5 or available_t5 == ["No models found"]:
        available_t5 = ["Please place T5 files in 'models/t5/' directory"]
    if not available_loras or available_loras == ["No models found"]:
        available_loras = ["Please place LoRA files in 'models/lora/' directory"]

    def generate_with_progress(*args):
        """Wrapper to handle progress updates"""
        progress = gr.Progress()

        def progress_callback(value, desc):
            progress(value, desc=desc)

        # Unpack arguments
        (positive_prompt, negative_prompt, width, height, num_frames,
         steps, cfg, shift, seed, scheduler, denoise_strength,
         model_name, vae_name, t5_model, quantization, attention_mode,
         lora_enabled, lora_name, lora_strength,
         compile_enabled, compile_backend, block_swap_enabled, blocks_to_swap,
         output_format, fps) = args

        video_path, video_array, metadata = workflow.generate_video(
            positive_prompt=positive_prompt,
            negative_prompt=negative_prompt,
            model_name=model_name,
            vae_name=vae_name,
            t5_model=t5_model,
            width=width,
            height=height,
            num_frames=num_frames,
            steps=steps,
            cfg=cfg,
            shift=shift,
            seed=seed,
            scheduler=scheduler,
            denoise_strength=denoise_strength,
            quantization=quantization,
            attention_mode=attention_mode,
            lora_enabled=lora_enabled,
            lora_name=lora_name,
            lora_strength=lora_strength,
            compile_enabled=compile_enabled,
            compile_backend=compile_backend,
            block_swap_enabled=block_swap_enabled,
            blocks_to_swap=blocks_to_swap,
            output_format=output_format,
            fps=fps,
            progress_callback=progress_callback
        )

        # Create sample frames for preview
        preview_frames = []
        frame_indices = np.linspace(0, len(video_array) - 1, min(8, len(video_array)), dtype=int)
        for idx in frame_indices:
            # Convert to PIL Image for gallery
            from PIL import Image
            frame_pil = Image.fromarray(video_array[idx])
            preview_frames.append(frame_pil)

        metadata_text = json.dumps(metadata, indent=2, default=str)

        return video_path, preview_frames, metadata_text

    # Create interface with tabs
    with gr.Blocks(title="WanVideo Genesis", theme=gr.themes.Soft()) as interface:
        gr.Markdown(
            """
            # 🎬 WanVideo Genesis - Text to Video Generation

            Advanced video generation system powered by Genesis Core and WanVideo models
            """
        )

        with gr.Tabs():
            # Main Generation Tab
            with gr.Tab("Generation"):
                with gr.Row():
                    with gr.Column(scale=1):
                        # Text inputs
                        positive_prompt = gr.Textbox(
                            label="Positive Prompt",
                            placeholder="Describe your video...",
                            value="镜头跟随穿深蓝色长裙的女人走在教堂走廊",
                            lines=4
                        )
                        negative_prompt = gr.Textbox(
                            label="Negative Prompt (Optional)",
                            placeholder="What to avoid...",
                            value="",
                            lines=2
                        )

                        # Basic parameters
                        with gr.Group():
                            gr.Markdown("### Video Settings")
                            with gr.Row():
                                width = gr.Slider(64, 2048, value=1280, step=16, label="Width")
                                height = gr.Slider(64, 2048, value=720, step=16, label="Height")
                            with gr.Row():
                                num_frames = gr.Slider(1, 241, value=61, step=1, label="Frames")
                                fps = gr.Slider(8, 60, value=16, step=1, label="FPS")

                        # Generation parameters
                        with gr.Group():
                            gr.Markdown("### Generation Parameters")
                            steps = gr.Slider(1, 100, value=4, step=1, label="Steps")
                            cfg = gr.Slider(0.0, 30.0, value=1.0, step=0.1, label="CFG Scale")
                            shift = gr.Slider(0.0, 100.0, value=5.0, step=0.1, label="Shift")
                            seed = gr.Number(value=-1, label="Seed (-1 for random)")
                            scheduler = gr.Dropdown(
                                choices=["sa_ode_stable/lowstep", "unipc", "ddim", "euler", "euler_a"],
                                value="sa_ode_stable/lowstep",
                                label="Scheduler"
                            )
                            denoise_strength = gr.Slider(0.0, 1.0, value=1.0, step=0.01, label="Denoise Strength")

                        generate_btn = gr.Button("🎬 Generate Video", variant="primary", size="lg")

                    with gr.Column(scale=1):
                        # Output
                        video_output = gr.Video(label="Generated Video")
                        with gr.Accordion("Preview Frames", open=False):
                            frames_gallery = gr.Gallery(label="Frame Preview", columns=4, height=200)
                        with gr.Accordion("Generation Metadata", open=False):
                            metadata_output = gr.Code(language="json", label="Metadata")

            # Model Settings Tab
            with gr.Tab("Model Settings"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Model Configuration")

                        # Model selection with dropdown
                        model_name = gr.Dropdown(
                            choices=available_models if available_models else ["No models found"],
                            value=available_models[0] if available_models and available_models[0] != "No models found" else "Wan2_IceCannon_t2v2.1_nsfw_RCM_Lab_4step.safetensors",
                            label="Select Model",
                            allow_custom_value=True,
                            interactive=True
                        )

                        # VAE selection with dropdown
                        vae_name = gr.Dropdown(
                            choices=available_vaes if available_vaes else ["No VAE found"],
                            value=available_vaes[0] if available_vaes and available_vaes[0] != "No VAE found" else "Wan2_1_VAE_bf16.safetensors",
                            label="Select VAE",
                            allow_custom_value=True,
                            interactive=True
                        )

                        # T5 selection with dropdown
                        t5_model = gr.Dropdown(
                            choices=available_t5 if available_t5 else ["No T5 models found"],
                            value=available_t5[0] if available_t5 and available_t5[0] != "No T5 models found" else "models_t5_umt5-xxl-enc-fp8_fully_uncensored.safetensors",
                            label="Select T5 Model",
                            allow_custom_value=True,
                            interactive=True
                        )

                        # Refresh button to rescan models
                        refresh_models_btn = gr.Button("🔄 Refresh Model List", size="sm")

                    with gr.Column():
                        gr.Markdown("### Advanced Settings")
                        quantization = gr.Dropdown(
                            choices=["disabled", "fp8_scaled", "fp4_scaled", "int8"],
                            value="fp4_scaled",
                            label="Quantization"
                        )
                        attention_mode = gr.Dropdown(
                            choices=["sageattn", "flash_attn", "sdpa", "xformers"],
                            value="sageattn",
                            label="Attention Mode"
                        )
                        output_format = gr.Dropdown(
                            choices=["mp4", "gif", "webm", "frames"],
                            value="mp4",
                            label="Output Format"
                        )

            # LoRA Settings Tab
            with gr.Tab("LoRA"):
                lora_enabled = gr.Checkbox(label="Enable LoRA", value=True)

                # LoRA selection with dropdown
                lora_name = gr.Dropdown(
                    choices=available_loras if available_loras else ["No LoRA found"],
                    value=available_loras[0] if available_loras and available_loras[0] != "No LoRA found" else "Kinesis-T2V-14B_lora_fix.safetensors",
                    label="Select LoRA",
                    allow_custom_value=True,
                    interactive=True
                )

                lora_strength = gr.Slider(-2.0, 2.0, value=1.0, step=0.01, label="LoRA Strength")

            # Optimization Tab
            with gr.Tab("Optimization"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Torch Compile")
                        compile_enabled = gr.Checkbox(label="Enable Torch Compile", value=False)
                        compile_backend = gr.Dropdown(
                            choices=["inductor", "eager", "aot_eager", "cudagraphs"],
                            value="inductor",
                            label="Compile Backend"
                        )

                    with gr.Column():
                        gr.Markdown("### Block Swap")
                        block_swap_enabled = gr.Checkbox(label="Enable Block Swap", value=False)
                        blocks_to_swap = gr.Slider(0, 48, value=16, step=1, label="Blocks to Swap")

            # Presets Tab
            with gr.Tab("Presets"):
                gr.Markdown(
                    """
                    ### Quick Presets

                    Select a preset configuration for common use cases:
                    """
                )

                preset_buttons = gr.Radio(
                    choices=[
                        "Fast Preview (4 steps, low quality)",
                        "Standard (30 steps, balanced)",
                        "High Quality (50 steps, best quality)",
                        "Memory Optimized (Block swap enabled)",
                        "Speed Optimized (Compile enabled)"
                    ],
                    label="Preset Configuration"
                )

                def apply_preset(preset_name):
                    if "Fast Preview" in preset_name:
                        return 4, 1.0, 5.0, "sa_ode_stable/lowstep", False, False
                    elif "Standard" in preset_name:
                        return 30, 6.0, 5.0, "unipc", False, False
                    elif "High Quality" in preset_name:
                        return 50, 8.0, 5.0, "ddim", False, False
                    elif "Memory Optimized" in preset_name:
                        return 30, 6.0, 5.0, "unipc", False, True
                    elif "Speed Optimized" in preset_name:
                        return 30, 6.0, 5.0, "unipc", True, False
                    else:
                        return 30, 6.0, 5.0, "unipc", False, False

                preset_buttons.change(
                    apply_preset,
                    inputs=[preset_buttons],
                    outputs=[steps, cfg, shift, scheduler, compile_enabled, block_swap_enabled]
                )

        # Refresh models function
        def refresh_all_models():
            """Refresh all model lists"""
            model_dirs = get_model_directories()

            # Rescan models
            new_models = scan_model_files(model_dirs["models"])
            new_vaes = scan_model_files(model_dirs["vae"])
            new_t5 = scan_model_files(model_dirs["t5"])
            new_loras = scan_model_files(model_dirs["lora"])

            # Check if models exist, provide better messages
            if not new_models or new_models == ["No models found"]:
                new_models = ["Please place model files in 'models/' directory"]
            if not new_vaes or new_vaes == ["No models found"]:
                new_vaes = ["Please place VAE files in 'models/vae/' directory"]
            if not new_t5 or new_t5 == ["No models found"]:
                new_t5 = ["Please place T5 files in 'models/t5/' directory"]
            if not new_loras or new_loras == ["No models found"]:
                new_loras = ["Please place LoRA files in 'models/lora/' directory"]

            return (
                gr.Dropdown(choices=new_models if new_models else ["No models found"]),
                gr.Dropdown(choices=new_vaes if new_vaes else ["No VAE found"]),
                gr.Dropdown(choices=new_t5 if new_t5 else ["No T5 models found"]),
                gr.Dropdown(choices=new_loras if new_loras else ["No LoRA found"])
            )

        # Connect refresh button
        refresh_models_btn.click(
            refresh_all_models,
            outputs=[model_name, vae_name, t5_model, lora_name]
        )

        # Connect generate button
        generate_btn.click(
            generate_with_progress,
            inputs=[
                positive_prompt, negative_prompt, width, height, num_frames,
                steps, cfg, shift, seed, scheduler, denoise_strength,
                model_name, vae_name, t5_model, quantization, attention_mode,
                lora_enabled, lora_name, lora_strength,
                compile_enabled, compile_backend, block_swap_enabled, blocks_to_swap,
                output_format, fps
            ],
            outputs=[video_output, frames_gallery, metadata_output]
        )

        # Examples
        gr.Examples(
            examples=[
                ["A majestic eagle soaring through mountain peaks at sunset", "", 1280, 720, 61],
                ["镜头跟随穿深蓝色长裙的女人走在教堂走廊", "", 1280, 720, 61],
                ["Underwater coral reef with colorful fish swimming", "", 1280, 720, 61],
                ["Time-lapse of flowers blooming in a garden", "", 1280, 720, 61],
            ],
            inputs=[positive_prompt, negative_prompt, width, height, num_frames],
            label="Example Prompts"
        )

    return interface


if __name__ == "__main__":
    interface = create_interface()
    interface.launch(
        server_name="0.0.0.0",  # Allow all connections
        server_port=7860,  # Use different port to avoid conflict
        share=False,
        inbrowser=False
    )