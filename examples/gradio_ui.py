"""
Genesis Gradio UI
Interactive web interface using Gradio
Author: eddy
"""

import sys
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import gradio as gr
    GRADIO_AVAILABLE = True
except ImportError:
    GRADIO_AVAILABLE = False
    print("Error: Gradio not installed")
    print("Install: pip install gradio")
    sys.exit(1)

import requests
from typing import Optional, Tuple


class GenesisGradioClient:
    """Gradio client for Genesis Advanced Server"""
    
    def __init__(self, server_url: str = "http://localhost:5000"):
        self.server_url = server_url.rstrip('/')
        self.session_id = None
        self._create_session()
    
    def _create_session(self):
        """Create session with server"""
        try:
            response = requests.post(f"{self.server_url}/api/session/create", json={
                'client_type': 'gradio'
            })
            data = response.json()
            self.session_id = data['session_id']
        except Exception as e:
            print(f"Warning: Could not create session: {e}")
    
    def submit_task(self, task_type: str, params: dict) -> str:
        """Submit task to server"""
        response = requests.post(f"{self.server_url}/api/task/submit", json={
            'task_type': task_type,
            'params': params,
            'session_id': self.session_id
        })
        data = response.json()
        
        if not data['success']:
            raise RuntimeError(data.get('error', 'Unknown error'))
        
        return data['task_id']
    
    def get_task_status(self, task_id: str) -> dict:
        """Get task status"""
        response = requests.get(f"{self.server_url}/api/task/{task_id}")
        return response.json()['task']
    
    def wait_for_task(self, task_id: str, progress_callback=None) -> dict:
        """Wait for task completion with progress updates"""
        while True:
            task = self.get_task_status(task_id)
            status = task['status']
            progress = task['progress']
            
            # Update progress if callback provided
            if progress_callback:
                progress_callback(progress, f"Status: {status}")
            
            # Check if done
            if status in ['completed', 'failed', 'cancelled']:
                return task
            
            time.sleep(1)
    
    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 512,
        height: int = 512,
        steps: int = 20,
        cfg_scale: float = 7.0,
        seed: int = -1,
        sampler: str = "euler",
        scheduler: str = "normal",
        progress=gr.Progress()
    ) -> Tuple[str, str]:
        """
        Generate image
        
        Returns:
            (status_message, result_info)
        """
        try:
            # Update progress
            progress(0, desc="Submitting task...")
            
            # Submit task
            task_id = self.submit_task('generate', {
                'prompt': prompt,
                'negative_prompt': negative_prompt,
                'width': width,
                'height': height,
                'steps': steps,
                'cfg_scale': cfg_scale,
                'seed': seed,
                'sampler': sampler,
                'scheduler': scheduler
            })
            
            progress(0.1, desc=f"Task ID: {task_id}")
            
            # Wait for completion
            last_progress = 0
            while True:
                task = self.get_task_status(task_id)
                status = task['status']
                current_progress = task['progress'] / 100.0
                
                # Update progress
                if current_progress > last_progress:
                    progress(current_progress, desc=f"Generating... {task['progress']}%")
                    last_progress = current_progress
                
                # Check if done
                if status == 'completed':
                    progress(1.0, desc="Completed!")
                    result = task.get('result', {})
                    return (
                        f"✅ Generation completed!\n\nTask ID: {task_id}\nTime: {result.get('execution_time', 0):.2f}s",
                        f"**Prompt:** {prompt}\n\n**Parameters:**\n- Steps: {steps}\n- CFG: {cfg_scale}\n- Size: {width}x{height}\n- Sampler: {sampler}\n- Scheduler: {scheduler}"
                    )
                
                elif status == 'failed':
                    error = task.get('error', 'Unknown error')
                    return (
                        f"❌ Generation failed!\n\nError: {error}",
                        f"Task ID: {task_id}"
                    )
                
                elif status == 'cancelled':
                    return (
                        f"⚠️ Generation cancelled",
                        f"Task ID: {task_id}"
                    )
                
                time.sleep(0.5)
        
        except Exception as e:
            return (
                f"❌ Error: {str(e)}",
                "Please check if the server is running"
            )


def create_gradio_ui(server_url: str = "http://localhost:5000"):
    """
    Create Gradio UI for Genesis
    
    Args:
        server_url: Genesis server URL
    """
    client = GenesisGradioClient(server_url)
    
    # Custom CSS
    custom_css = """
    .gradio-container {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .main-title {
        text-align: center;
        color: #2c3e50;
    }
    """
    
    with gr.Blocks(css=custom_css, title="Genesis AI Generator") as demo:
        gr.Markdown("""
        # 🎨 Genesis AI Image Generator
        
        Powered by Genesis Engine - Advanced Flask Server
        """, elem_classes=["main-title"])
        
        with gr.Row():
            with gr.Column(scale=2):
                # Input section
                gr.Markdown("### 📝 Generation Settings")
                
                prompt = gr.Textbox(
                    label="Prompt",
                    placeholder="Describe what you want to generate...",
                    lines=3,
                    value="a beautiful sunset over mountains"
                )
                
                negative_prompt = gr.Textbox(
                    label="Negative Prompt",
                    placeholder="What to avoid in the image...",
                    lines=2,
                    value="ugly, blurry, low quality"
                )
                
                with gr.Row():
                    width = gr.Slider(
                        label="Width",
                        minimum=256,
                        maximum=2048,
                        step=64,
                        value=512
                    )
                    
                    height = gr.Slider(
                        label="Height",
                        minimum=256,
                        maximum=2048,
                        step=64,
                        value=512
                    )
                
                with gr.Row():
                    steps = gr.Slider(
                        label="Steps",
                        minimum=1,
                        maximum=150,
                        step=1,
                        value=20
                    )
                    
                    cfg_scale = gr.Slider(
                        label="CFG Scale",
                        minimum=1.0,
                        maximum=20.0,
                        step=0.5,
                        value=7.0
                    )
                
                with gr.Row():
                    seed = gr.Number(
                        label="Seed (-1 for random)",
                        value=-1,
                        precision=0
                    )
                    
                    sampler = gr.Dropdown(
                        label="Sampler",
                        choices=["euler", "euler_a", "heun", "dpm_2", "dpm_2_a", "ddim"],
                        value="euler"
                    )
                    
                    scheduler = gr.Dropdown(
                        label="Scheduler",
                        choices=["normal", "karras", "exponential", "simple"],
                        value="normal"
                    )
                
                # Generate button
                generate_btn = gr.Button(
                    "🎨 Generate Image",
                    variant="primary",
                    size="lg"
                )
            
            with gr.Column(scale=1):
                # Output section
                gr.Markdown("### 📊 Results")
                
                status_output = gr.Textbox(
                    label="Status",
                    lines=8,
                    interactive=False
                )
                
                info_output = gr.Markdown(
                    label="Info",
                    value="Ready to generate..."
                )
        
        # Examples
        gr.Markdown("### 💡 Examples")
        
        gr.Examples(
            examples=[
                [
                    "a serene mountain landscape at sunset, beautiful colors, 4k",
                    "ugly, blurry, low quality",
                    512, 512, 20, 7.0
                ],
                [
                    "a cute cat sitting on a windowsill, soft lighting, detailed fur",
                    "distorted, ugly, bad anatomy",
                    512, 512, 25, 7.5
                ],
                [
                    "cyberpunk city at night, neon lights, futuristic, highly detailed",
                    "blurry, low quality, bad composition",
                    768, 512, 30, 8.0
                ],
                [
                    "portrait of a beautiful woman, studio lighting, professional photography",
                    "ugly, deformed, bad anatomy, worst quality",
                    512, 768, 25, 7.5
                ],
            ],
            inputs=[prompt, negative_prompt, width, height, steps, cfg_scale]
        )
        
        # Server info
        with gr.Accordion("ℹ️ Server Information", open=False):
            gr.Markdown(f"""
            **Server URL:** {server_url}
            
            **Session ID:** {client.session_id}
            
            **Features:**
            - Real-time progress tracking
            - Parameter validation
            - High tolerance for input errors
            - Automatic parameter normalization
            
            **Supported Clients:**
            - Gradio (this interface)
            - Web browsers (JavaScript)
            - Tkinter apps
            - PyQt apps
            - Command-line scripts
            """)
        
        # Connect generate button
        generate_btn.click(
            fn=client.generate,
            inputs=[
                prompt, negative_prompt,
                width, height,
                steps, cfg_scale, seed,
                sampler, scheduler
            ],
            outputs=[status_output, info_output]
        )
    
    return demo


def main():
    """Launch Gradio UI"""
    print("="*60)
    print("Genesis Gradio UI")
    print("="*60)
    
    # Server URL
    SERVER_URL = "http://localhost:5000"
    
    print(f"\nConnecting to server: {SERVER_URL}")
    print("\nMake sure Genesis Advanced Server is running:")
    print("  python -m genesis.examples.start_advanced_server")
    print()
    
    # Create and launch UI
    demo = create_gradio_ui(SERVER_URL)
    
    print("Launching Gradio UI...")
    print("="*60)
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True
    )


if __name__ == "__main__":
    if not GRADIO_AVAILABLE:
        print("Please install Gradio: pip install gradio")
    else:
        main()
