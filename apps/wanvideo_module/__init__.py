"""
WanVideo Module
Provides video generation interface for Genesis WebUI

Author: eddy
Date: 2025-11-14
"""

import sys
import os
from pathlib import Path
import gradio as gr
import importlib.util
from typing import Optional, Dict, Any

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

WAN_VIDEO_AVAILABLE = False
NODE_CLASS_MAPPINGS = {}

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

try:
    import sys
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from utils import triton_ops_stub
    from compat import comfy_stub

    spec_fp = importlib.util.spec_from_file_location(
        "folder_paths",
        project_root / "core" / "folder_paths.py"
    )
    folder_paths = importlib.util.module_from_spec(spec_fp)
    sys.modules['folder_paths'] = folder_paths
    spec_fp.loader.exec_module(folder_paths)

    spec_ext = importlib.util.spec_from_file_location(
        "folder_paths_ext",
        project_root / "core" / "folder_paths_ext.py"
    )
    folder_paths_ext = importlib.util.module_from_spec(spec_ext)
    spec_ext.loader.exec_module(folder_paths_ext)

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
        print(f"[WanVideo Module] Loaded {len(NODE_CLASS_MAPPINGS)} nodes")
except Exception as e:
    print(f"[WanVideo Module] Not available: {e}")


class WanVideoWorkflow:
    """WanVideo workflow executor"""

    def __init__(self):
        if not WAN_VIDEO_AVAILABLE:
            raise RuntimeError("WanVideo nodes not available")

        self.nodes = NODE_CLASS_MAPPINGS

        self.t5_encoder = self.nodes.get("LoadWanVideoT5TextEncoder")() if self.nodes.get("LoadWanVideoT5TextEncoder") else None
        self.text_encoder = self.nodes.get("WanVideoTextEncode")() if self.nodes.get("WanVideoTextEncode") else None
        self.model_loader = self.nodes.get("WanVideoModelLoader")() if self.nodes.get("WanVideoModelLoader") else None
        self.vae_loader = self.nodes.get("WanVideoVAELoader")() if self.nodes.get("WanVideoVAELoader") else None
        self.empty_embeds = self.nodes.get("WanVideoEmptyEmbeds")() if self.nodes.get("WanVideoEmptyEmbeds") else None
        self.sampler = self.nodes.get("WanVideoSampler")() if self.nodes.get("WanVideoSampler") else None
        self.decoder = self.nodes.get("WanVideoDecode")() if self.nodes.get("WanVideoDecode") else None

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
            raise RuntimeError(f"Missing WanVideo nodes: {missing}")

        self.current_model = None
        self.current_vae = None
        self.current_t5 = None

    def generate_video(
        self,
        positive_prompt: str,
        negative_prompt: str,
        model_name: str = "WanVideo_0.1.safetensors",
        vae_name: str = "wanvideo_vae.safetensors",
        width: int = 848,
        height: int = 480,
        num_frames: int = 129,
        steps: int = 50,
        cfg: float = 6.0,
        seed: int = -1,
        progress=gr.Progress()
    ):
        """Generate video using WanVideo workflow"""
        try:
            progress(0.1, desc="Loading T5 encoder...")
            t5_result = self.t5_encoder.load_model("google-t5/t5-v1_1-xxl")
            t5_model = t5_result[0]

            progress(0.2, desc="Encoding prompts...")
            positive_embeds = self.text_encoder.process(positive_prompt, t5_model)
            negative_embeds = self.text_encoder.process(negative_prompt, t5_model)

            progress(0.3, desc="Loading WanVideo model...")
            model_result = self.model_loader.load_model(model_name)
            model = model_result[0]

            progress(0.4, desc="Loading VAE...")
            vae_result = self.vae_loader.load_vae(vae_name)
            vae = vae_result[0]

            progress(0.5, desc="Preparing empty embeds...")
            empty_result = self.empty_embeds.process(positive_embeds[0])

            progress(0.6, desc="Running sampler...")
            latent_result = self.sampler.process(
                model=model,
                positive=positive_embeds[0],
                negative=negative_embeds[0],
                empty_embeds=empty_result,
                width=width,
                height=height,
                num_frames=num_frames,
                steps=steps,
                cfg=cfg,
                seed=seed
            )

            progress(0.9, desc="Decoding video...")
            video_result = self.decoder.decode(latent_result[0], vae)

            progress(1.0, desc="Video generation complete!")
            return video_result[0]

        except Exception as e:
            print(f"Error generating video: {e}")
            raise


def create_wanvideo_tab(workflow: Optional[WanVideoWorkflow] = None):
    """Create WanVideo generation tab with nested tabs"""
    with gr.Tab("WanVideo"):
        # Import the UI builder
        try:
            from .ui_builder import create_wanvideo_ui_nested
            components = create_wanvideo_ui_nested(workflow)
            return components
        except Exception as e:
            print(f"[WanVideo Tab] Error loading UI: {e}")
            gr.Markdown(f"""
            ## ⚠️ Error Loading WanVideo UI

            Error: {str(e)}

            See [docs/TROUBLESHOOTING.md](../../docs/TROUBLESHOOTING.md) for help.
            """)
            return {}
