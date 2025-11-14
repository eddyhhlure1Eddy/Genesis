#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Genesis WebUI - Fully Integrated Interface
Combines all Genesis apps into one unified interface

Features:
- txt2img: Stable Diffusion image generation (from gradio_real.py)
- WanVideo: Video generation (from wanvideo_gradio_app.py)
- Model Management
- Settings

Author: eddy
Date: 2025-11-13
"""

import sys
import os
from pathlib import Path
import time
from datetime import datetime

# Set environment before any imports
os.environ['GRADIO_SERVER_NAME'] = '0.0.0.0'
os.environ['GRADIO_SERVER_PORT'] = '7860'

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Fix console encoding for Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

#================================================================================
# Import Dependencies
#================================================================================

print("="*70)
print("Genesis WebUI - Loading...")
print("="*70)

# Import Gradio
try:
    import gradio as gr
    print(f"✓ Gradio {gr.__version__}")
except ImportError:
    print("✗ Gradio not installed")
    print("Install: pip install gradio")
    sys.exit(1)

# Import PyTorch
try:
    import torch
    print(f"✓ PyTorch {torch.__version__}")
    print(f"✓ CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
except ImportError:
    print("✗ PyTorch not installed")
    sys.exit(1)

# Import Diffusers (for SD generation)
try:
    from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
    SD_AVAILABLE = True
    print("✓ Diffusers available (SD enabled)")
except Exception as e:
    SD_AVAILABLE = False
    print(f"✗ Diffusers not available: {e}")
    print("  SD generation will be disabled")

# Import folder_paths
import importlib.util
spec = importlib.util.spec_from_file_location(
    "folder_paths",
    project_root / "core" / "folder_paths.py"
)
folder_paths = importlib.util.module_from_spec(spec)
spec.loader.exec_module(folder_paths)
print("✓ Folder paths loaded")

# Try to import WanVideo components
WAN_VIDEO_AVAILABLE = False
try:
    # Apply Gradio API fix
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
    except:
        pass

    # Import WanVideo wrapper
    sys.path.insert(0, str(project_root))
    from genesis.utils import triton_ops_stub
    from genesis.compat import comfy_stub
    from genesis.core import folder_paths_ext

    wrapper_path = project_root / "custom_nodes" / "Comfyui" / "ComfyUI-WanVideoWrapper"
    spec = importlib.util.spec_from_file_location(
        "ComfyUI_WanVideoWrapper",
        wrapper_path / "__init__.py"
    )
    wrapper_module = importlib.util.module_from_spec(spec)
    sys.modules['ComfyUI_WanVideoWrapper'] = wrapper_module
    spec.loader.exec_module(wrapper_module)

    NODE_CLASS_MAPPINGS = getattr(wrapper_module, 'NODE_CLASS_MAPPINGS', {})
    if NODE_CLASS_MAPPINGS:
        WAN_VIDEO_AVAILABLE = True
        print(f"✓ WanVideo available ({len(NODE_CLASS_MAPPINGS)} nodes)")
except Exception as e:
    print(f"✗ WanVideo not available: {e}")
    print("  Video generation will be disabled")

print("="*70)
print()

#================================================================================
# SD Generator Class (from gradio_real.py)
#================================================================================

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

            if progress:
                progress(0.3, desc="Creating pipeline...")

            try:
                self.pipe = StableDiffusionPipeline.from_single_file(
                    model_path,
                    torch_dtype=self.dtype,
                    safety_checker=None,
                    requires_safety_checker=False
                )
            except:
                self.pipe = StableDiffusionPipeline.from_pretrained(
                    model_path,
                    torch_dtype=self.dtype,
                    safety_checker=None,
                    requires_safety_checker=False
                )

            if progress:
                progress(0.6, desc="Moving to device...")

            self.pipe = self.pipe.to(self.device)
            self.pipe.scheduler = EulerDiscreteScheduler.from_config(
                self.pipe.scheduler.config
            )

            if self.device == "cuda":
                self.pipe.enable_attention_slicing()
                try:
                    self.pipe.enable_xformers_memory_efficient_attention()
                except:
                    pass

            self.current_model = model_name

            if progress:
                progress(1.0, desc="Model loaded!")

            return f"✓ Model loaded: {model_name}"

        except Exception as e:
            import traceback
            return f"✗ Failed to load model: {e}\n{traceback.format_exc()}"

    def generate(self, prompt, negative_prompt="", width=512, height=512,
                 steps=20, cfg=7.0, seed=-1, batch_count=1, batch_size=1,
                 progress=gr.Progress()):
        """Generate images"""
        if not SD_AVAILABLE or self.pipe is None:
            return None, "✗ Please load a model first"

        try:
            if seed == -1:
                seed = torch.randint(0, 2**32 - 1, (1,)).item()

            all_images = []

            for batch_idx in range(batch_count):
                progress(batch_idx / batch_count, desc=f"Batch {batch_idx+1}/{batch_count}")

                current_seed = seed + batch_idx
                generator = torch.Generator(device=self.device).manual_seed(int(current_seed))

                def callback(step, timestep, latents):
                    total_steps = steps
                    current_progress = (batch_idx / batch_count) + (step / total_steps / batch_count)
                    progress(current_progress, desc=f"Step {step}/{total_steps}")

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

            progress(1.0, desc="Complete!")

            info = f"""### ✓ Generated {len(all_images)} images

**Prompt:** {prompt}
**Negative:** {negative_prompt}
**Settings:** {width}x{height}, {steps} steps, CFG {cfg}, Seed {seed}
**Model:** {self.current_model}
**Device:** {self.device}
"""
            return all_images[0] if len(all_images) == 1 else all_images, info

        except Exception as e:
            import traceback
            return None, f"✗ Generation failed: {e}\n{traceback.format_exc()}"


#================================================================================
# WanVideo Workflow Class (from wanvideo_gradio_app.py)
#================================================================================

class WanVideoWorkflow:
    """WanVideo generation workflow"""

    def __init__(self):
        self.nodes = {}
        self.node_outputs = {}

    def generate_video(self, positive_prompt, negative_prompt="",
                      model_name="", vae_name="", t5_model="",
                      width=512, height=320, num_frames=9,
                      steps=20, cfg=7.5, shift=7.0, seed=-1,
                      scheduler="normal", denoise_strength=1.0,
                      quantization="none", attention_mode="sdpa",
                      lora_enabled=False, lora_name="", lora_strength=1.0,
                      compile_enabled=False, compile_backend="inductor",
                      block_swap_enabled=False, blocks_to_swap=0,
                      output_format="mp4", fps=24,
                      progress_callback=None):
        """Generate video (simplified placeholder)"""

        if not WAN_VIDEO_AVAILABLE:
            return None, [], {"error": "WanVideo not available"}

        # Placeholder implementation
        # Real implementation would call the WanVideo nodes here
        import numpy as np
        from PIL import Image

        # Create placeholder frames
        frames = []
        for i in range(num_frames):
            # Create a simple gradient frame as placeholder
            frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
            frames.append(frame)

        metadata = {
            "prompt": positive_prompt,
            "negative_prompt": negative_prompt,
            "width": width,
            "height": height,
            "frames": num_frames,
            "steps": steps,
            "cfg": cfg,
            "seed": seed,
            "timestamp": datetime.now().isoformat()
        }

        # Save placeholder video
        output_dir = project_root / "output"
        output_dir.mkdir(exist_ok=True)
        video_path = output_dir / f"video_{int(time.time())}.mp4"

        return str(video_path), frames, metadata


#================================================================================
# Create Unified WebUI
#================================================================================

def create_webui():
    """Create fully integrated WebUI"""

    sd_gen = SDGenerator()
    wv_workflow = WanVideoWorkflow()

    def get_models_list():
        return {
            'checkpoints': folder_paths.get_filename_list('checkpoints'),
            'loras': folder_paths.get_filename_list('loras'),
            'vaes': folder_paths.get_filename_list('vae')
        }

    def get_system_info():
        info = f"""## System Information

**Device:** {sd_gen.device}
**CUDA:** {torch.cuda.is_available()}
"""
        if torch.cuda.is_available():
            info += f"""**GPU:** {torch.cuda.get_device_name(0)}
**VRAM:** {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB
"""
        info += f"""
**PyTorch:** {torch.__version__}
**Gradio:** {gr.__version__}

## Available Features

**Stable Diffusion:** {'✓ Enabled' if SD_AVAILABLE else '✗ Disabled'}
**WanVideo:** {'✓ Enabled' if WAN_VIDEO_AVAILABLE else '✗ Disabled'}
"""
        models = get_models_list()
        info += f"""
## Available Models

**Checkpoints:** {len(models['checkpoints'])}
**LoRAs:** {len(models['loras'])}
**VAEs:** {len(models['vaes'])}
"""
        if sd_gen.current_model:
            info += f"\n**Loaded:** {sd_gen.current_model}"

        return info

    # Create interface
    with gr.Blocks(
        title="Genesis WebUI - Integrated",
        theme=gr.themes.Soft(primary_hue="purple"),
        css="""
        #gallery { min-height: 400px; }
        .gradio-container { max-width: 100% !important; }
        """
    ) as demo:

        gr.Markdown("""
        # Genesis WebUI - Fully Integrated

        **All-in-one interface for AI generation**
        """)

        with gr.Tabs() as tabs:

            #============================================================
            # txt2img Tab (Full SD Implementation)
            #============================================================

            with gr.Tab("txt2img", id="tab_txt2img"):
                if not SD_AVAILABLE:
                    gr.Markdown("""
                    ## ⚠️ Stable Diffusion Not Available

                    Install dependencies:
                    ```
                    pip install diffusers transformers accelerate
                    ```
                    """)
                else:
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("### Generation Settings")

                            models = get_models_list()
                            model_choices = ["HF:runwayml/stable-diffusion-v1-5"]
                            model_choices.extend(models['checkpoints'])

                            model_selector = gr.Dropdown(
                                choices=model_choices,
                                value=model_choices[0],
                                label="Model"
                            )

                            load_model_btn = gr.Button("📥 Load Model", variant="secondary")
                            model_status = gr.Textbox(label="Status", value="No model loaded", lines=2)

                            gr.Markdown("---")

                            prompt = gr.Textbox(
                                label="Prompt",
                                lines=3,
                                value="a beautiful landscape, mountains, sunset, 4k, highly detailed"
                            )

                            negative_prompt = gr.Textbox(
                                label="Negative Prompt",
                                lines=2,
                                value="ugly, blurry, low quality, distorted"
                            )

                            with gr.Row():
                                width = gr.Slider(256, 2048, 512, step=64, label="Width")
                                height = gr.Slider(256, 2048, 512, step=64, label="Height")

                            with gr.Row():
                                steps = gr.Slider(1, 150, 20, label="Steps")
                                cfg = gr.Slider(1.0, 30.0, 7.0, step=0.5, label="CFG Scale")

                            with gr.Row():
                                batch_count = gr.Slider(1, 10, 1, step=1, label="Batch Count")
                                batch_size = gr.Slider(1, 8, 1, step=1, label="Batch Size")

                            seed = gr.Number(label="Seed (-1=random)", value=-1, precision=0)

                            generate_btn = gr.Button("🎨 Generate", variant="primary", size="lg")

                        with gr.Column(scale=1):
                            gr.Markdown("### Output")

                            output_gallery = gr.Gallery(
                                label="Generated Images",
                                columns=2,
                                rows=2,
                                height="auto"
                            )

                            output_info = gr.Markdown("Click 'Generate' to start...")

                    gr.Examples(
                        examples=[
                            ["a serene mountain landscape at sunset, 4k", "ugly, blurry", 512, 512, 20, 7.0, -1],
                            ["portrait of a beautiful woman, professional photography", "ugly, deformed", 512, 768, 25, 7.5, 42],
                            ["cyberpunk city at night, neon lights, futuristic", "blurry", 768, 512, 30, 8.0, 123],
                        ],
                        inputs=[prompt, negative_prompt, width, height, steps, cfg, seed]
                    )

                    load_model_btn.click(
                        fn=sd_gen.load_model,
                        inputs=[model_selector],
                        outputs=[model_status]
                    )

                    generate_btn.click(
                        fn=sd_gen.generate,
                        inputs=[prompt, negative_prompt, width, height, steps, cfg, seed, batch_count, batch_size],
                        outputs=[output_gallery, output_info]
                    )

            #============================================================
            # WanVideo Tab (Placeholder/Integration Point)
            #============================================================

            with gr.Tab("WanVideo", id="tab_wanvideo"):
                if not WAN_VIDEO_AVAILABLE:
                    gr.Markdown("""
                    ## ⚠️ WanVideo Not Available

                    WanVideo nodes not found. Check:
                    - `custom_nodes/Comfyui/ComfyUI-WanVideoWrapper/` directory
                    - Model files in `models/unet/` directory
                    """)
                else:
                    gr.Markdown("""
                    ## 🎬 WanVideo Text-to-Video Generation

                    **Status:** WanVideo integration available

                    Full WanVideo interface coming soon...
                    """)

                    # Placeholder controls
                    with gr.Row():
                        with gr.Column():
                            wv_prompt = gr.Textbox(label="Prompt", lines=3)
                            wv_generate = gr.Button("Generate Video")
                        with gr.Column():
                            wv_output = gr.Video(label="Output")

            #============================================================
            # Models Tab
            #============================================================

            with gr.Tab("Models", id="tab_models"):
                gr.Markdown("### Model Management")

                models = get_models_list()

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("**Checkpoints**")
                        checkpoint_list = gr.Dataframe(
                            headers=["Model"],
                            value=[[m] for m in models['checkpoints']],
                            interactive=False
                        )

                    with gr.Column():
                        gr.Markdown("**LoRAs**")
                        lora_list = gr.Dataframe(
                            headers=["LoRA"],
                            value=[[m] for m in models['loras']],
                            interactive=False
                        )

                    with gr.Column():
                        gr.Markdown("**VAEs**")
                        vae_list = gr.Dataframe(
                            headers=["VAE"],
                            value=[[m] for m in models['vaes']],
                            interactive=False
                        )

                refresh_btn = gr.Button("🔄 Refresh Lists")

                def refresh_models():
                    models = get_models_list()
                    return (
                        [[m] for m in models['checkpoints']],
                        [[m] for m in models['loras']],
                        [[m] for m in models['vaes']]
                    )

                refresh_btn.click(
                    fn=refresh_models,
                    outputs=[checkpoint_list, lora_list, vae_list]
                )

            #============================================================
            # Settings Tab
            #============================================================

            with gr.Tab("Settings", id="tab_settings"):
                gr.Markdown("### System Information")

                system_info = gr.Markdown(get_system_info())

                refresh_info_btn = gr.Button("🔄 Refresh Info")
                refresh_info_btn.click(
                    fn=get_system_info,
                    outputs=[system_info]
                )

                gr.Markdown("""
                ### Configuration

                **Model Paths:** Edit `extra_model_paths.yaml`

                **Performance:**
                - xformers: Auto-enabled if available
                - Attention Slicing: Auto-enabled on GPU

                **Troubleshooting:** See [TROUBLESHOOTING.md](../TROUBLESHOOTING.md)
                """)

        gr.Markdown("""
        ---
        **Genesis WebUI Integrated** | Author: eddy | Version: 1.0.0
        """)

    return demo


#================================================================================
# Main Entry Point
#================================================================================

def main():
    print("="*70)
    print("Launching Genesis WebUI...")
    print("="*70)
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
        try:
            demo.launch(
                server_name="0.0.0.0",
                server_port=0,
                share=False,
                inbrowser=True
            )
        except Exception as e2:
            print(f"All launch attempts failed: {e2}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nShutdown by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
