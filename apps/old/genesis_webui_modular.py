#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Genesis WebUI - Modular Architecture
Auto-loads all app modules and combines them into unified interface

Author: eddy
Date: 2025-11-14
"""

import sys
import os
from pathlib import Path
import time

os.environ['GRADIO_SERVER_NAME'] = '0.0.0.0'
os.environ['GRADIO_SERVER_PORT'] = '7860'

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

print("="*70)
print("Genesis WebUI - Modular Loading...")
print("="*70)

try:
    import gradio as gr
    print(f"✓ Gradio {gr.__version__}")
except ImportError:
    print("✗ Gradio not installed")
    print("Install: pip install gradio")
    sys.exit(1)

try:
    import torch
    print(f"✓ PyTorch {torch.__version__}")
    print(f"✓ CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
except ImportError:
    print("✗ PyTorch not installed")
    sys.exit(1)

print()
print("="*70)
print("Loading App Modules...")
print("="*70)

apps_dir = Path(__file__).parent
sys.path.insert(0, str(apps_dir))

sd_generator = None
wanvideo_workflow = None
sd_components = {}
wanvideo_components = {}
SDGenerator = None
create_sd_tab = None
WanVideoWorkflow = None
create_wanvideo_tab = None
WAN_VIDEO_AVAILABLE = False

try:
    from sd_module import SDGenerator, create_sd_tab
    sd_generator = SDGenerator()
    print("✓ SD Module loaded")
except Exception as e:
    print(f"✗ SD Module failed: {e}")

try:
    from wanvideo_module import WanVideoWorkflow, create_wanvideo_tab, WAN_VIDEO_AVAILABLE
    if WAN_VIDEO_AVAILABLE:
        wanvideo_workflow = WanVideoWorkflow()
        print("✓ WanVideo Module loaded")
    else:
        print("✗ WanVideo Module: nodes not available")
except Exception as e:
    print(f"✗ WanVideo Module failed: {e}")

print("="*70)
print()


def create_models_tab():
    """Create model management tab"""
    with gr.Tab("Models"):
        gr.Markdown("""
        ## Model Management

        Manage your Stable Diffusion and WanVideo models here.

        ### Model Directories
        - **Checkpoints**: `models/checkpoints/`
        - **VAE**: `models/vae/`
        - **LoRA**: `models/loras/`
        - **WanVideo**: `models/unet/`

        Models are automatically detected from these directories.
        """)


def create_settings_tab():
    """Create settings tab"""
    with gr.Tab("Settings"):
        gr.Markdown("""
        ## Settings

        ### System Information
        """)

        with gr.Row():
            with gr.Column():
                gr.Markdown(f"""
                **Device**: {'CUDA' if torch.cuda.is_available() else 'CPU'}

                **GPU**: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}

                **PyTorch**: {torch.__version__}

                **Gradio**: {gr.__version__}
                """)

            with gr.Column():
                gr.Markdown("""
                ### Available Modules

                - **SD Generator**: {}
                - **WanVideo**: {}
                """.format(
                    "✓ Loaded" if sd_generator else "✗ Not Available",
                    "✓ Loaded" if wanvideo_workflow else "✗ Not Available"
                ))

        gr.Markdown("""
        ### Configuration

        Model paths are configured in `extra_model_paths.yaml`

        See [docs/MODEL_PATHS_CONFIG.md](../docs/MODEL_PATHS_CONFIG.md) for details.
        """)


def main():
    """Main WebUI application"""

    print("="*70)
    print("Building WebUI Interface...")
    print("="*70)

    with gr.Blocks(
        title="Genesis WebUI",
        theme=gr.themes.Soft(),
        css="""
        #gallery { min-height: 400px; }
        .gradio-container { max-width: 100% !important; }
        """
    ) as demo:
        gr.Markdown("""
        # Genesis WebUI

        Unified interface for AI generation - Stable Diffusion and WanVideo
        """)

        with gr.Tabs():
            if sd_generator:
                sd_components.update(create_sd_tab(sd_generator))
            else:
                with gr.Tab("txt2img"):
                    gr.Markdown("""
                    ## ⚠️ SD Generator Not Available

                    Please install required dependencies:
                    ```
                    pip install diffusers transformers accelerate
                    ```
                    """)

            with gr.Tab("img2img"):
                gr.Markdown("""
                ## img2img (Coming Soon)

                Image-to-image generation will be available in future updates.
                """)

            if create_wanvideo_tab:
                if wanvideo_workflow:
                    wanvideo_components.update(create_wanvideo_tab(wanvideo_workflow))
                else:
                    wanvideo_components.update(create_wanvideo_tab(None))
            else:
                with gr.Tab("WanVideo"):
                    gr.Markdown("""
                    ## ⚠️ WanVideo Module Not Loaded

                    Could not load WanVideo module. See console for details.
                    """)

            create_models_tab()
            create_settings_tab()

        gr.Markdown("""
        ---

        **Genesis AI Engine** | [Documentation](../docs/) | [GitHub](https://github.com/eddyhhlure1Eddy/Genesis)
        """)

    print()
    print("="*70)
    print("Launching Genesis WebUI...")
    print("="*70)

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
