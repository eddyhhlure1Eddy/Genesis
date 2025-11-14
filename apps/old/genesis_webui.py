#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Genesis WebUI - Unified Interface (Similar to A1111 WebUI)
Author: eddy
"""

import sys
import os
from pathlib import Path
import torch
import time
from datetime import datetime

os.environ['GRADIO_SERVER_NAME'] = '0.0.0.0'
os.environ['GRADIO_SERVER_PORT'] = '7860'

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    import gradio as gr
    print(f"Gradio version: {gr.__version__}")
except ImportError:
    print("Error: Gradio not installed")
    print("Install: pip install gradio")
    sys.exit(1)

try:
    from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False
    print("Warning: Diffusers not available. SD generation will be disabled.")

import importlib.util
spec = importlib.util.spec_from_file_location(
    "folder_paths",
    project_root / "core" / "folder_paths.py"
)
folder_paths = importlib.util.module_from_spec(spec)
spec.loader.exec_module(folder_paths)


class GenesisWebUI:
    """Genesis Unified WebUI"""

    def __init__(self):
        self.sd_pipe = None
        self.wanvideo_engine = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.current_model = None
        self.generation_history = []

    def get_models_list(self):
        """Get available models"""
        return {
            'checkpoints': folder_paths.get_filename_list('checkpoints'),
            'loras': folder_paths.get_filename_list('loras'),
            'vaes': folder_paths.get_filename_list('vae')
        }

    def load_sd_model(self, model_name, progress=None):
        """Load Stable Diffusion model"""
        try:
            if not DIFFUSERS_AVAILABLE:
                return "Error: Diffusers library not installed"

            if progress:
                progress(0.1, desc="Loading model...")

            if model_name.startswith("HF:"):
                model_path = model_name[3:]
                print(f"Loading HuggingFace model: {model_path}")
            else:
                model_path = folder_paths.get_full_path('checkpoints', model_name)
                print(f"Loading local model: {model_path}")

            if progress:
                progress(0.3, desc="Loading pipeline...")

            try:
                self.sd_pipe = StableDiffusionPipeline.from_single_file(
                    model_path,
                    torch_dtype=self.dtype,
                    safety_checker=None,
                    requires_safety_checker=False
                )
            except:
                self.sd_pipe = StableDiffusionPipeline.from_pretrained(
                    model_path,
                    torch_dtype=self.dtype,
                    safety_checker=None,
                    requires_safety_checker=False
                )

            if progress:
                progress(0.6, desc="Moving to device...")

            self.sd_pipe = self.sd_pipe.to(self.device)

            if progress:
                progress(0.8, desc="Configuring scheduler...")

            self.sd_pipe.scheduler = EulerDiscreteScheduler.from_config(
                self.sd_pipe.scheduler.config
            )

            if self.device == "cuda":
                self.sd_pipe.enable_attention_slicing()
                try:
                    self.sd_pipe.enable_xformers_memory_efficient_attention()
                    print("xformers enabled")
                except:
                    print("xformers not available")

            self.current_model = model_name

            if progress:
                progress(1.0, desc="Model loaded!")

            return f"Model loaded successfully: {model_name}"

        except Exception as e:
            import traceback
            error_msg = f"Failed to load model: {e}\n{traceback.format_exc()}"
            print(error_msg)
            return error_msg

    def generate_image(
        self,
        prompt,
        negative_prompt="",
        width=512,
        height=512,
        steps=20,
        cfg_scale=7.0,
        seed=-1,
        batch_count=1,
        batch_size=1,
        progress=gr.Progress()
    ):
        """Generate image with Stable Diffusion"""
        if not DIFFUSERS_AVAILABLE:
            return None, "Error: Diffusers library not installed"

        if self.sd_pipe is None:
            return None, "Error: Please load a model first"

        try:
            if seed == -1:
                seed = torch.randint(0, 2**32 - 1, (1,)).item()

            all_images = []

            for batch_idx in range(batch_count):
                progress(batch_idx / batch_count, desc=f"Batch {batch_idx + 1}/{batch_count}")

                current_seed = seed + batch_idx
                generator = torch.Generator(device=self.device).manual_seed(int(current_seed))

                def callback(step, timestep, latents):
                    total_steps = steps
                    current_progress = (batch_idx / batch_count) + (step / total_steps / batch_count)
                    progress(current_progress, desc=f"Generating... Step {step}/{total_steps}")

                result = self.sd_pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    width=width,
                    height=height,
                    num_inference_steps=steps,
                    guidance_scale=cfg_scale,
                    generator=generator,
                    num_images_per_prompt=batch_size,
                    callback=callback,
                    callback_steps=1
                )

                all_images.extend(result.images)

            progress(1.0, desc="Complete!")

            generation_info = {
                'prompt': prompt,
                'negative_prompt': negative_prompt,
                'width': width,
                'height': height,
                'steps': steps,
                'cfg_scale': cfg_scale,
                'seed': seed,
                'model': self.current_model,
                'timestamp': datetime.now().isoformat()
            }

            self.generation_history.append(generation_info)

            info_text = f"""
### Generation Info

**Prompt:** {prompt}

**Negative:** {negative_prompt}

**Settings:**
- Size: {width} x {height}
- Steps: {steps}
- CFG Scale: {cfg_scale}
- Seed: {seed}
- Model: {self.current_model}
- Device: {self.device}
- Batches: {batch_count} x {batch_size} = {len(all_images)} images
"""

            return all_images[0] if len(all_images) == 1 else all_images, info_text

        except Exception as e:
            import traceback
            error_msg = f"Generation failed: {e}\n{traceback.format_exc()}"
            print(error_msg)
            return None, error_msg

    def get_system_info(self):
        """Get system information"""
        info = f"""
## System Information

**Device:** {self.device}
**Precision:** {self.dtype}
"""

        if torch.cuda.is_available():
            info += f"""
**GPU:** {torch.cuda.get_device_name(0)}
**VRAM:** {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB
**CUDA Version:** {torch.version.cuda}
"""

        info += f"""
**PyTorch:** {torch.__version__}
**Gradio:** {gr.__version__}
**Diffusers:** {'Available' if DIFFUSERS_AVAILABLE else 'Not Available'}
"""

        models = self.get_models_list()
        info += f"""

## Available Models

**Checkpoints:** {len(models['checkpoints'])} models
**LoRAs:** {len(models['loras'])} models
**VAEs:** {len(models['vaes'])} models
"""

        if self.current_model:
            info += f"""

## Currently Loaded

**Model:** {self.current_model}
"""

        return info


def create_txt2img_tab(webui):
    """Create txt2img tab (like A1111)"""

    models = webui.get_models_list()
    model_choices = ["HF:runwayml/stable-diffusion-v1-5", "HF:stabilityai/stable-diffusion-2-1"]
    model_choices.extend(models['checkpoints'])

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Generation Settings")

            model_selector = gr.Dropdown(
                choices=model_choices,
                value=model_choices[0] if model_choices else None,
                label="Model",
                interactive=True
            )

            load_model_btn = gr.Button("Load Model", variant="secondary")
            model_status = gr.Textbox(
                label="Model Status",
                value="No model loaded",
                interactive=False,
                lines=2
            )

            gr.Markdown("---")

            prompt = gr.Textbox(
                label="Prompt",
                placeholder="Describe what you want to generate...",
                lines=3,
                value="a beautiful landscape with mountains and lake, sunset, 4k, highly detailed"
            )

            negative_prompt = gr.Textbox(
                label="Negative Prompt",
                placeholder="What to avoid...",
                lines=2,
                value="ugly, blurry, low quality, distorted, bad anatomy"
            )

            with gr.Row():
                width = gr.Slider(256, 2048, 512, step=64, label="Width")
                height = gr.Slider(256, 2048, 512, step=64, label="Height")

            with gr.Row():
                steps = gr.Slider(1, 150, 20, step=1, label="Sampling Steps")
                cfg_scale = gr.Slider(1.0, 30.0, 7.0, step=0.5, label="CFG Scale")

            with gr.Row():
                batch_count = gr.Slider(1, 10, 1, step=1, label="Batch Count")
                batch_size = gr.Slider(1, 8, 1, step=1, label="Batch Size")

            seed = gr.Number(label="Seed (-1 = random)", value=-1, precision=0)

            generate_btn = gr.Button("Generate", variant="primary", size="lg")

        with gr.Column(scale=1):
            gr.Markdown("### Output")

            output_gallery = gr.Gallery(
                label="Generated Images",
                show_label=True,
                elem_id="gallery",
                columns=2,
                rows=2,
                object_fit="contain",
                height="auto"
            )

            output_info = gr.Markdown("Click 'Generate' to create images...")

    gr.Markdown("### Example Prompts")
    gr.Examples(
        examples=[
            ["a serene mountain landscape at sunset, beautiful colors, 4k", "ugly, blurry, low quality", 512, 512, 20, 7.0, -1],
            ["portrait of a beautiful woman, studio lighting, professional photography", "ugly, deformed, bad anatomy", 512, 768, 25, 7.5, 42],
            ["cyberpunk city at night, neon lights, futuristic, highly detailed", "blurry, low quality", 768, 512, 30, 8.0, 123],
            ["cute cat sitting on windowsill, soft lighting, detailed fur", "distorted, ugly", 512, 512, 20, 7.0, -1],
        ],
        inputs=[prompt, negative_prompt, width, height, steps, cfg_scale, seed]
    )

    load_model_btn.click(
        fn=webui.load_sd_model,
        inputs=[model_selector],
        outputs=[model_status]
    )

    generate_btn.click(
        fn=webui.generate_image,
        inputs=[
            prompt, negative_prompt,
            width, height,
            steps, cfg_scale, seed,
            batch_count, batch_size
        ],
        outputs=[output_gallery, output_info]
    )


def create_img2img_tab(webui):
    """Create img2img tab (placeholder for future)"""
    gr.Markdown("""
    # Image to Image

    Coming soon...

    Features:
    - Image-to-image generation
    - Inpainting
    - Outpainting
    - Sketch
    """)


def create_extras_tab(webui):
    """Create extras tab (placeholder)"""
    gr.Markdown("""
    # Extras

    Coming soon...

    Features:
    - Upscaling
    - Face restoration
    - Color correction
    - Batch processing
    """)


def create_wanvideo_tab(webui):
    """Create WanVideo generation tab"""
    gr.Markdown("""
    # WanVideo Generation

    Video generation with WanVideo models.

    Coming soon...

    Features:
    - Text-to-video generation
    - Video customization
    - Frame control
    - Video export
    """)


def create_settings_tab(webui):
    """Create settings tab"""

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Model Paths")

            config_path = gr.Textbox(
                label="Model Config Path",
                value=str(Path.cwd() / "extra_model_paths.yaml"),
                interactive=False
            )

            gr.Markdown("""
            **Configure model paths in:** `extra_model_paths.yaml`

            This allows sharing models with ComfyUI and other tools.
            """)

            gr.Markdown("### Performance Settings")

            enable_xformers = gr.Checkbox(
                label="Enable xformers (if available)",
                value=True
            )

            attention_slicing = gr.Checkbox(
                label="Enable Attention Slicing (saves VRAM)",
                value=True
            )

            gr.Markdown("### UI Settings")

            theme_selector = gr.Radio(
                choices=["Light", "Dark", "Auto"],
                value="Auto",
                label="Theme"
            )

            show_grid = gr.Checkbox(
                label="Show generation grid",
                value=True
            )

        with gr.Column():
            gr.Markdown("### System Info")

            system_info = gr.Markdown(webui.get_system_info())

            refresh_btn = gr.Button("Refresh System Info")
            refresh_btn.click(
                fn=webui.get_system_info,
                outputs=[system_info]
            )


def create_models_tab(webui):
    """Create model management tab"""

    models = webui.get_models_list()

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Checkpoints")

            checkpoint_list = gr.Dataframe(
                headers=["Model Name"],
                datatype=["str"],
                value=[[m] for m in models['checkpoints']],
                interactive=False
            )

            refresh_checkpoints_btn = gr.Button("Refresh List")

        with gr.Column():
            gr.Markdown("### LoRAs")

            lora_list = gr.Dataframe(
                headers=["LoRA Name"],
                datatype=["str"],
                value=[[m] for m in models['loras']],
                interactive=False
            )

            refresh_loras_btn = gr.Button("Refresh List")

        with gr.Column():
            gr.Markdown("### VAEs")

            vae_list = gr.Dataframe(
                headers=["VAE Name"],
                datatype=["str"],
                value=[[m] for m in models['vaes']],
                interactive=False
            )

            refresh_vaes_btn = gr.Button("Refresh List")

    gr.Markdown("""
    ### Model Download

    Download models from HuggingFace or CivitAI:

    **HuggingFace Models:**
    - runwayml/stable-diffusion-v1-5
    - stabilityai/stable-diffusion-2-1
    - stabilityai/stable-diffusion-xl-base-1.0

    **Local Models:**
    Place your .safetensors or .ckpt files in the configured checkpoints folder.
    """)

    def refresh_models():
        models = webui.get_models_list()
        return (
            [[m] for m in models['checkpoints']],
            [[m] for m in models['loras']],
            [[m] for m in models['vaes']]
        )

    refresh_checkpoints_btn.click(
        fn=refresh_models,
        outputs=[checkpoint_list, lora_list, vae_list]
    )

    refresh_loras_btn.click(
        fn=refresh_models,
        outputs=[checkpoint_list, lora_list, vae_list]
    )

    refresh_vaes_btn.click(
        fn=refresh_models,
        outputs=[checkpoint_list, lora_list, vae_list]
    )


def create_webui():
    """Create the unified WebUI"""

    webui = GenesisWebUI()

    with gr.Blocks(
        title="Genesis WebUI",
        theme=gr.themes.Soft(primary_hue="purple"),
        css="""
        #gallery { min-height: 400px; }
        .gradio-container { max-width: 100% !important; }
        """
    ) as demo:

        gr.Markdown("""
        # Genesis WebUI

        **Unified interface for AI generation** - Stable Diffusion, WanVideo, and more
        """)

        with gr.Tabs() as tabs:
            with gr.Tab("txt2img", id="tab_txt2img"):
                create_txt2img_tab(webui)

            with gr.Tab("img2img", id="tab_img2img"):
                create_img2img_tab(webui)

            with gr.Tab("WanVideo", id="tab_wanvideo"):
                create_wanvideo_tab(webui)

            with gr.Tab("Extras", id="tab_extras"):
                create_extras_tab(webui)

            with gr.Tab("Models", id="tab_models"):
                create_models_tab(webui)

            with gr.Tab("Settings", id="tab_settings"):
                create_settings_tab(webui)

        gr.Markdown("""
        ---
        **Genesis WebUI** - Powered by Stable Diffusion | Author: eddy | Version: 1.0.0
        """)

    return demo


def main():
    """Main entry point"""
    print("=" * 70)
    print("Genesis WebUI - Unified Interface")
    print("=" * 70)
    print()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"VRAM: {vram:.1f} GB")

    print()
    print("Starting WebUI...")
    print("=" * 70)
    print()

    demo = create_webui()

    try:
        demo.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            inbrowser=True,
            show_error=True
        )
    except Exception as e:
        print(f"Launch failed: {e}")
        print("Trying alternative port...")
        try:
            demo.launch(
                server_name="0.0.0.0",
                server_port=0,
                share=False,
                inbrowser=True,
                show_error=True
            )
        except Exception as e2:
            print(f"Failed to launch: {e2}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nShutdown by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
