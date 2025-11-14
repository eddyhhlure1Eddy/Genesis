"""
Stable Diffusion Module
Provides SD generation interface for Genesis WebUI

Author: eddy
Date: 2025-11-14
"""

import sys
import torch
from pathlib import Path
import gradio as gr
from typing import Optional, List
from PIL import Image
import numpy as np
import random

project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import importlib.util
folder_paths_file = project_root / "core" / "folder_paths.py"
if folder_paths_file.exists():
    spec = importlib.util.spec_from_file_location("folder_paths", folder_paths_file)
    folder_paths = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(folder_paths)
else:
    class folder_paths:
        @staticmethod
        def get_full_path(folder_type, filename):
            return str(project_root / "models" / folder_type / filename)
        @staticmethod
        def get_filename_list(folder_type):
            model_dir = project_root / "models" / folder_type
            if not model_dir.exists():
                return []
            return [f.name for f in model_dir.glob("*.safetensors")] + \
                   [f.name for f in model_dir.glob("*.ckpt")]

try:
    from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
    SD_AVAILABLE = True
except ImportError:
    SD_AVAILABLE = False


class SDGenerator:
    """Stable Diffusion Generator"""

    def __init__(self):
        self.pipe = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.current_model = None

    def load_model(self, model_name, progress=None):
        """Load SD model"""
        if not SD_AVAILABLE:
            return "Error: Diffusers library not available"

        try:
            if progress:
                progress(0.1, desc="Loading model...")

            if model_name.startswith("HF:"):
                model_path = model_name[3:]
            else:
                model_path = folder_paths.get_full_path('checkpoints', model_name)

            if self.current_model == model_path:
                return f"Model already loaded: {model_name}"

            if progress:
                progress(0.3, desc="Loading pipeline...")

            self.pipe = StableDiffusionPipeline.from_single_file(
                model_path,
                torch_dtype=self.dtype,
                use_safetensors=model_path.endswith('.safetensors')
            )

            if progress:
                progress(0.6, desc="Moving to device...")

            self.pipe = self.pipe.to(self.device)

            if self.device == "cuda":
                self.pipe.enable_attention_slicing()
                try:
                    self.pipe.enable_xformers_memory_efficient_attention()
                except:
                    pass

            self.current_model = model_path

            if progress:
                progress(1.0, desc="Model loaded!")

            return f"Model loaded successfully: {model_name}"

        except Exception as e:
            return f"Error loading model: {str(e)}"

    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 512,
        height: int = 512,
        steps: int = 20,
        cfg: float = 7.0,
        seed: int = -1,
        batch_count: int = 1,
        batch_size: int = 1,
        progress=gr.Progress()
    ) -> List[Image.Image]:
        """Generate images"""
        if not SD_AVAILABLE:
            return []

        if self.pipe is None:
            progress(0, desc="No model loaded, please load a model first")
            return []

        if seed == -1:
            seed = random.randint(0, 2**32 - 1)

        all_images = []
        total_batches = batch_count
        total_steps = steps * total_batches

        for batch_idx in range(batch_count):
            current_seed = seed + batch_idx
            generator = torch.Generator(device=self.device).manual_seed(int(current_seed))

            progress(
                batch_idx / total_batches,
                desc=f"Generating batch {batch_idx + 1}/{total_batches} (seed: {current_seed})"
            )

            def callback(step, timestep, latents):
                overall_step = batch_idx * steps + step
                progress(
                    overall_step / total_steps,
                    desc=f"Batch {batch_idx + 1}/{total_batches} - Step {step}/{steps}"
                )

            result = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=steps,
                guidance_scale=cfg,
                generator=generator,
                num_images_per_prompt=batch_size,
                callback=callback,
                callback_steps=1
            )

            all_images.extend(result.images)

        progress(1.0, desc=f"Generated {len(all_images)} images successfully!")
        return all_images


def get_available_models():
    """Get list of available models"""
    try:
        models = folder_paths.get_filename_list('checkpoints')
        hf_models = [
            "HF:runwayml/stable-diffusion-v1-5",
            "HF:stabilityai/stable-diffusion-2-1"
        ]
        return hf_models + models
    except:
        return []


def create_sd_tab(generator: SDGenerator):
    """Create SD generation tab"""
    with gr.Tab("txt2img"):
        with gr.Row():
            with gr.Column(scale=1):
                model_dropdown = gr.Dropdown(
                    label="Model",
                    choices=get_available_models(),
                    value=None,
                    interactive=True
                )
                load_btn = gr.Button("Load Model", variant="primary")
                model_status = gr.Textbox(label="Model Status", interactive=False)

                gr.Markdown("### Generation Settings")
                prompt = gr.Textbox(
                    label="Prompt",
                    placeholder="Enter your prompt here...",
                    lines=3
                )
                negative_prompt = gr.Textbox(
                    label="Negative Prompt",
                    value="nsfw, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry",
                    lines=2
                )

                with gr.Row():
                    width = gr.Slider(256, 1024, 512, step=64, label="Width")
                    height = gr.Slider(256, 1024, 512, step=64, label="Height")

                with gr.Row():
                    steps = gr.Slider(1, 100, 20, step=1, label="Steps")
                    cfg = gr.Slider(1, 20, 7, step=0.5, label="CFG Scale")

                seed = gr.Number(label="Seed (-1 for random)", value=-1, precision=0)

                with gr.Row():
                    batch_count = gr.Slider(1, 10, 1, step=1, label="Batch Count")
                    batch_size = gr.Slider(1, 4, 1, step=1, label="Batch Size")

                generate_btn = gr.Button("Generate", variant="primary", size="lg")

            with gr.Column(scale=2):
                output_gallery = gr.Gallery(
                    label="Generated Images",
                    show_label=True,
                    elem_id="gallery",
                    columns=2,
                    rows=2,
                    height="auto"
                )

        load_btn.click(
            fn=generator.load_model,
            inputs=[model_dropdown],
            outputs=[model_status]
        )

        generate_btn.click(
            fn=generator.generate,
            inputs=[
                prompt, negative_prompt, width, height,
                steps, cfg, seed, batch_count, batch_size
            ],
            outputs=[output_gallery]
        )

    return {
        'model_dropdown': model_dropdown,
        'load_btn': load_btn,
        'model_status': model_status,
        'generate_btn': generate_btn,
        'output_gallery': output_gallery
    }
