#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Genesis Gradio 简化版 - 避免 Gradio 5.x 的启动问题
"""

import sys
from pathlib import Path
import torch
import os

# 设置环境变量避免一些问题
os.environ['GRADIO_SERVER_NAME'] = '127.0.0.1'
os.environ['GRADIO_SERVER_PORT'] = '7860'

sys.path.insert(0, str(Path(__file__).parent))

import gradio as gr
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler

# 导入 folder_paths
import importlib.util
spec = importlib.util.spec_from_file_location(
    "folder_paths", 
    Path(__file__).parent / "core" / "folder_paths.py"
)
folder_paths = importlib.util.module_from_spec(spec)
spec.loader.exec_module(folder_paths)


class SimpleGenerator:
    def __init__(self):
        self.pipe = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        
    def load_model(self, model_name):
        """加载模型"""
        try:
            print(f"\n{'='*60}")
            print(f"加载模型: {model_name}")
            print(f"{'='*60}")
            
            if model_name.startswith("HF:"):
                model_path = model_name[3:]
                print(f"HuggingFace 模型: {model_path}")
            else:
                model_path = folder_paths.get_full_path('checkpoints', model_name)
                print(f"本地模型路径: {model_path}")
            
            print(f"设备: {self.device}")
            print("正在加载...")
            
            # 加载模型
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
            
            print("✅ 模型加载成功!")
            return f"✅ 模型加载成功: {model_name}"
        except Exception as e:
            error = f"❌ 加载失败: {str(e)}"
            print(error)
            return error
    
    def generate(self, prompt, negative_prompt, width, height, steps, cfg, seed):
        """生成图像"""
        if self.pipe is None:
            return None, "❌ 请先加载模型！"
        
        try:
            if seed == -1:
                seed = torch.randint(0, 2**32-1, (1,)).item()
            
            generator = torch.Generator(device=self.device).manual_seed(int(seed))
            
            print(f"\n生成中: {prompt[:50]}...")
            
            result = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=steps,
                guidance_scale=cfg,
                generator=generator
            )
            
            image = result.images[0]
            info = f"✅ 生成完成!\n\n**提示词:** {prompt}\n\n**参数:** {width}x{height}, {steps}步, CFG {cfg}, 种子 {seed}"
            
            return image, info
        except Exception as e:
            return None, f"❌ 生成失败: {str(e)}"


# 全局生成器
gen = SimpleGenerator()

# 获取模型列表
models = folder_paths.get_filename_list('checkpoints')
model_choices = ["HF:runwayml/stable-diffusion-v1-5", "HF:stabilityai/stable-diffusion-2-1"]
if models:
    model_choices.extend(models)

# 创建界面
with gr.Blocks(title="Genesis AI", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎨 Genesis AI 图像生成器")
    
    with gr.Row():
        with gr.Column():
            model_select = gr.Dropdown(
                label="选择模型",
                choices=model_choices,
                value=model_choices[0]
            )
            load_btn = gr.Button("📥 加载模型", variant="secondary")
            status = gr.Textbox(label="状态", value="未加载", lines=2)
            
            gr.Markdown("---")
            
            prompt = gr.Textbox(
                label="提示词",
                lines=3,
                value="a beautiful landscape, sunset, 4k"
            )
            neg_prompt = gr.Textbox(
                label="负向提示词",
                lines=2,
                value="ugly, blurry, low quality"
            )
            
            with gr.Row():
                width = gr.Slider(256, 1024, 512, step=64, label="宽度")
                height = gr.Slider(256, 1024, 512, step=64, label="高度")
            
            with gr.Row():
                steps = gr.Slider(1, 100, 20, step=1, label="步数")
                cfg = gr.Slider(1, 20, 7, step=0.5, label="CFG")
            
            seed = gr.Number(label="种子 (-1随机)", value=-1)
            
            gen_btn = gr.Button("🎨 生成图像", variant="primary")
        
        with gr.Column():
            output_img = gr.Image(label="生成结果", type="pil")
            output_info = gr.Markdown("等待生成...")
    
    # 事件
    load_btn.click(gen.load_model, inputs=[model_select], outputs=[status])
    gen_btn.click(
        gen.generate,
        inputs=[prompt, neg_prompt, width, height, steps, cfg, seed],
        outputs=[output_img, output_info]
    )

if __name__ == "__main__":
    print("="*60)
    print("Genesis AI - 简化版界面")
    print("="*60)
    print(f"设备: {gen.device}")
    print(f"可用模型: {len(models)} 个本地模型")
    print("="*60)
    
    # 使用最简单的启动方式
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        inbrowser=True,
        quiet=False
    )
