"""
WanVideo UI Builder
Builds the complete WanVideo UI with nested tabs for integration into Genesis WebUI

Author: eddy
Date: 2025-11-14
"""

import gradio as gr
from typing import Optional


def create_wanvideo_ui_nested(workflow=None):
    """
    Create WanVideo UI with nested tabs
    This function is designed to be called inside a gr.Tab("WanVideo") context

    Returns dict of UI components
    """

    if workflow is None:
        gr.Markdown("""
        ## ⚠️ WanVideo Not Available

        WanVideo nodes not found. Please check:
        1. `custom_nodes/Comfyui/ComfyUI-WanVideoWrapper/` directory exists
        2. Model files in `models/unet/` directory
        3. VAE files in `models/vae/` directory

        See [docs/TROUBLESHOOTING.md](../../docs/TROUBLESHOOTING.md) for help.
        """)
        return {}

    gr.Markdown("""
    # 🎬 WanVideo - Text to Video Generation

    Advanced video generation system with 120 ComfyUI nodes
    """)

    # Create nested tabs
    with gr.Tabs():
        # Generation Tab
        with gr.Tab("Generation"):
            with gr.Row():
                with gr.Column(scale=1):
                    positive_prompt = gr.Textbox(
                        label="Positive Prompt",
                        placeholder="Describe your video...",
                        value="A beautiful sunset over mountains",
                        lines=4
                    )
                    negative_prompt = gr.Textbox(
                        label="Negative Prompt",
                        value="blurry, low quality, distorted",
                        lines=2
                    )

                    with gr.Group():
                        gr.Markdown("### Video Settings")
                        with gr.Row():
                            width = gr.Slider(64, 2048, 1280, step=16, label="Width")
                            height = gr.Slider(64, 2048, 720, step=16, label="Height")
                        with gr.Row():
                            num_frames = gr.Slider(1, 241, 61, step=1, label="Frames")
                            fps = gr.Slider(8, 60, 16, step=1, label="FPS")

                    with gr.Group():
                        gr.Markdown("### Generation Parameters")
                        steps = gr.Slider(1, 100, 50, step=1, label="Steps")
                        cfg = gr.Slider(0.0, 30.0, 6.0, step=0.1, label="CFG Scale")
                        seed = gr.Number(value=-1, label="Seed (-1 for random)")

                    generate_btn = gr.Button("🎬 Generate Video", variant="primary", size="lg")

                with gr.Column(scale=1):
                    video_output = gr.Video(label="Generated Video")
                    with gr.Accordion("Generation Info", open=False):
                        gen_info = gr.Textbox(label="Details", lines=10)

        # Model Settings Tab
        with gr.Tab("Model Settings"):
            gr.Markdown("### Model Configuration")
            model_name = gr.Textbox(
                label="Model Name",
                value="WanVideo_0.1.safetensors"
            )
            vae_name = gr.Textbox(
                label="VAE Name",
                value="wanvideo_vae.safetensors"
            )
            t5_model = gr.Textbox(
                label="T5 Model",
                value="google-t5/t5-v1_1-xxl"
            )

        # Advanced Tab
        with gr.Tab("Advanced"):
            gr.Markdown("### Advanced Settings")
            quantization = gr.Dropdown(
                choices=["disabled", "fp8", "fp4", "int8"],
                value="fp4",
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

        # LoRA Tab
        with gr.Tab("LoRA"):
            lora_enabled = gr.Checkbox(label="Enable LoRA", value=True)
            lora_name = gr.Textbox(
                label="LoRA Name",
                value="Kinesis-T2V-14B_lora_fix.safetensors"
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
            gr.Markdown("""
            ### Quick Presets

            Select a preset configuration for common use cases:
            """)
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

    # Connect generate button (simplified for now)
    if workflow and hasattr(workflow, 'generate_video'):
        def generate_wrapper(*args):
            try:
                result = workflow.generate_video(*args)
                return result, "Video generated successfully"
            except Exception as e:
                return None, f"Error: {str(e)}"

        generate_btn.click(
            fn=generate_wrapper,
            inputs=[positive_prompt, negative_prompt, model_name, vae_name,
                   width, height, num_frames, steps, cfg, seed],
            outputs=[video_output, gen_info]
        )

    return {
        'generate_btn': generate_btn,
        'video_output': video_output,
        'positive_prompt': positive_prompt
    }
