#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Genesis Gradio 真实生成界面
使用 diffusers 库实现真实的图像生成

依赖:
    pip install gradio diffusers transformers accelerate
"""

import sys
from pathlib import Path
import time
import torch

sys.path.insert(0, str(Path(__file__).parent))

try:
    import gradio as gr
    print(f"Gradio 版本: {gr.__version__}")
except ImportError:
    print("请安装 Gradio: pip install gradio")
    sys.exit(1)

try:
    from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
    print("✓ diffusers 已安装")
except ImportError:
    print("❌ 请安装 diffusers: pip install diffusers transformers accelerate")
    sys.exit(1)

# 导入 folder_paths
import importlib.util
spec = importlib.util.spec_from_file_location(
    "folder_paths", 
    Path(__file__).parent / "core" / "folder_paths.py"
)
folder_paths = importlib.util.module_from_spec(spec)
spec.loader.exec_module(folder_paths)


class GenesisGenerator:
    """Genesis 图像生成器"""
    
    def __init__(self):
        self.pipe = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.current_model = None
        
    def load_model(self, model_name, progress=None):
        """加载模型"""
        try:
            if progress:
                progress(0.1, desc="正在加载模型...")
            
            # 判断是本地模型还是 HuggingFace 模型
            if model_name.startswith("HF:"):
                # HuggingFace 模型
                model_id = model_name[3:]  # 去掉 "HF:" 前缀
                print(f"加载 HuggingFace 模型: {model_id}")
                model_path = model_id
            else:
                # 本地模型
                print(f"加载本地模型: {model_name}")
                model_path = folder_paths.get_full_path('checkpoints', model_name)
                print(f"模型路径: {model_path}")
            
            print(f"设备: {self.device}")
            
            if progress:
                progress(0.3, desc="加载模型文件...")
            
            # 加载 Stable Diffusion pipeline
            try:
                from diffusers import StableDiffusionPipeline
                self.pipe = StableDiffusionPipeline.from_single_file(
                    model_path,
                    torch_dtype=self.dtype,
                    safety_checker=None,
                    requires_safety_checker=False,
                    load_safety_checker=False
                )
            except Exception as e:
                # 如果 from_single_file 失败，尝试 from_pretrained
                print(f"尝试使用 from_pretrained 加载...")
                self.pipe = StableDiffusionPipeline.from_pretrained(
                    model_path,
                    torch_dtype=self.dtype,
                    safety_checker=None,
                    requires_safety_checker=False
                )
            
            if progress:
                progress(0.6, desc="将模型移至设备...")
            
            self.pipe = self.pipe.to(self.device)
            
            if progress:
                progress(0.8, desc="配置调度器...")
            
            # 使用 Euler 调度器
            self.pipe.scheduler = EulerDiscreteScheduler.from_config(
                self.pipe.scheduler.config
            )
            
            # 启用优化
            if self.device == "cuda":
                # 启用注意力切片以节省显存
                self.pipe.enable_attention_slicing()
                
                # 如果有 xformers，启用内存高效注意力
                try:
                    self.pipe.enable_xformers_memory_efficient_attention()
                    print("✓ 启用 xformers 优化")
                except:
                    print("ℹ xformers 未安装，使用默认注意力")
            
            self.current_model = model_name
            
            if progress:
                progress(1.0, desc="模型加载完成!")
            
            print("✓ 模型加载成功")
            return f"✅ 模型加载成功: {model_name}"
            
        except Exception as e:
            import traceback
            error_msg = f"❌ 加载模型失败: {e}\n\n{traceback.format_exc()}"
            print(error_msg)
            return error_msg
    
    def generate(
        self,
        prompt,
        negative_prompt="",
        width=512,
        height=512,
        steps=20,
        cfg_scale=7.0,
        seed=-1,
        progress=gr.Progress()
    ):
        """生成图像"""
        if self.pipe is None:
            return None, "❌ 请先加载模型！"
        
        try:
            # 设置随机种子
            if seed == -1:
                seed = torch.randint(0, 2**32 - 1, (1,)).item()
            
            generator = torch.Generator(device=self.device).manual_seed(int(seed))
            
            # 生成图像
            progress(0, desc="开始生成...")
            
            def callback(step, timestep, latents):
                progress(step / steps, desc=f"生成中... {step}/{steps}")
            
            result = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=steps,
                guidance_scale=cfg_scale,
                generator=generator,
                callback=callback,
                callback_steps=1
            )
            
            image = result.images[0]
            
            progress(1.0, desc="完成!")
            
            info = f"""
## ✅ 生成成功！

**提示词:** {prompt}

**参数:**
- 尺寸: {width} x {height}
- 步数: {steps}
- CFG: {cfg_scale}
- 种子: {seed}
- 设备: {self.device}
"""
            
            return image, info
            
        except Exception as e:
            import traceback
            error_msg = f"❌ 生成失败: {str(e)}\n\n{traceback.format_exc()}"
            print(error_msg)
            return None, error_msg


def get_models():
    """获取模型列表"""
    return {
        'checkpoints': folder_paths.get_filename_list('checkpoints'),
        'loras': folder_paths.get_filename_list('loras'),
        'vaes': folder_paths.get_filename_list('vae')
    }


def show_models():
    """显示模型信息"""
    models = get_models()
    
    info = f"""
## 📦 可用模型统计

- **Checkpoints:** {len(models['checkpoints'])} 个
- **LoRAs:** {len(models['loras'])} 个
- **VAEs:** {len(models['vaes'])} 个

---

### Checkpoints
"""
    
    if models['checkpoints']:
        for i, name in enumerate(models['checkpoints'][:10], 1):
            info += f"{i}. `{name}`\n"
        if len(models['checkpoints']) > 10:
            info += f"\n... 还有 {len(models['checkpoints']) - 10} 个\n"
    else:
        info += "*未找到模型*\n"
    
    return info


# 创建全局生成器
generator = GenesisGenerator()


def create_ui():
    """创建 Gradio 界面"""
    
    # 获取可用模型
    models = get_models()
    
    # 构建模型选择列表
    checkpoint_choices = []
    
    # 添加 HuggingFace 默认模型
    checkpoint_choices.append("HF:runwayml/stable-diffusion-v1-5")
    checkpoint_choices.append("HF:stabilityai/stable-diffusion-2-1")
    
    # 添加本地模型
    if models['checkpoints']:
        checkpoint_choices.extend(models['checkpoints'])
    
    with gr.Blocks(
        title="Genesis AI - 真实生成",
        theme=gr.themes.Soft(primary_hue="purple")
    ) as demo:
        
        gr.Markdown("""
        # 🎨 Genesis AI 图像生成器
        
        **真实图像生成** - 基于 Stable Diffusion
        """)
        
        with gr.Tabs():
            # 生成标签页
            with gr.Tab("🎨 图像生成"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 🎯 模型选择")
                        
                        model_selector = gr.Dropdown(
                            label="选择模型",
                            choices=checkpoint_choices,
                            value=checkpoint_choices[0] if checkpoint_choices else None,
                            interactive=True
                        )
                        
                        load_model_btn = gr.Button(
                            "📥 加载模型",
                            variant="secondary"
                        )
                        
                        model_status = gr.Textbox(
                            label="模型状态",
                            value="未加载模型",
                            interactive=False,
                            lines=2
                        )
                        
                        gr.Markdown("### 📝 生成设置")
                        
                        prompt = gr.Textbox(
                            label="正向提示词",
                            placeholder="描述你想生成的图像...",
                            lines=3,
                            value="a beautiful landscape with mountains and lake, sunset, 4k, highly detailed"
                        )
                        
                        negative_prompt = gr.Textbox(
                            label="负向提示词",
                            placeholder="要避免的内容...",
                            lines=2,
                            value="ugly, blurry, low quality, distorted, bad anatomy"
                        )
                        
                        gr.Markdown("### ⚙️ 参数设置")
                        
                        with gr.Row():
                            width = gr.Slider(
                                label="宽度",
                                minimum=256,
                                maximum=1024,
                                step=64,
                                value=512
                            )
                            
                            height = gr.Slider(
                                label="高度",
                                minimum=256,
                                maximum=1024,
                                step=64,
                                value=512
                            )
                        
                        with gr.Row():
                            steps = gr.Slider(
                                label="采样步数",
                                minimum=1,
                                maximum=100,
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
                        
                        seed = gr.Number(
                            label="种子 (-1 为随机)",
                            value=-1,
                            precision=0
                        )
                        
                        generate_btn = gr.Button(
                            "🎨 生成图像",
                            variant="primary",
                            size="lg"
                        )
                    
                    with gr.Column(scale=1):
                        gr.Markdown("### 🖼️ 生成结果")
                        
                        output_image = gr.Image(
                            label="生成的图像",
                            type="pil"
                        )
                        
                        output_info = gr.Markdown(
                            value="点击「生成图像」按钮开始..."
                        )
                
                # 示例
                gr.Markdown("### 💡 预设示例")
                
                gr.Examples(
                    examples=[
                        [
                            "a serene mountain landscape at sunset, beautiful colors, 4k",
                            "ugly, blurry, low quality",
                            512, 512, 20, 7.0, -1
                        ],
                        [
                            "a cute cat sitting on a windowsill, soft lighting, detailed fur",
                            "distorted, ugly, bad anatomy",
                            512, 512, 25, 7.5, 42
                        ],
                        [
                            "cyberpunk city at night, neon lights, futuristic, highly detailed",
                            "blurry, low quality, bad composition",
                            768, 512, 30, 8.0, 123
                        ],
                    ],
                    inputs=[prompt, negative_prompt, width, height, steps, cfg_scale, seed]
                )
                
                # 连接加载模型按钮
                load_model_btn.click(
                    fn=generator.load_model,
                    inputs=[model_selector],
                    outputs=[model_status]
                )
                
                # 连接生成按钮
                generate_btn.click(
                    fn=generator.generate,
                    inputs=[
                        prompt, negative_prompt,
                        width, height,
                        steps, cfg_scale, seed
                    ],
                    outputs=[output_image, output_info]
                )
            
            # 模型管理标签页
            with gr.Tab("📦 模型管理"):
                gr.Markdown(show_models())
                
                gr.Markdown(f"""
                ### 🔧 模型选择说明
                
                **本地模型:**
                - 你有 {len(models['checkpoints'])} 个本地 checkpoint 模型
                - 这些模型来自你配置的路径（extra_model_paths.yaml）
                - 直接选择模型名称即可加载
                
                **HuggingFace 模型:**
                - 以 `HF:` 开头的是在线模型
                - `HF:runwayml/stable-diffusion-v1-5` - SD 1.5（推荐）
                - `HF:stabilityai/stable-diffusion-2-1` - SD 2.1
                - 首次使用会自动下载（约 4GB）
                - 模型会缓存到: `~/.cache/huggingface/`
                
                ### 💡 使用步骤
                
                1. 在「图像生成」标签页选择模型
                2. 点击「📥 加载模型」按钮
                3. 等待加载完成
                4. 开始生成图像
                """)
            
            # 关于标签页
            with gr.Tab("ℹ️ 关于"):
                gr.Markdown(f"""
                # Genesis AI Engine
                
                ## 🚀 当前状态
                
                - **设备:** {generator.device}
                - **精度:** {generator.dtype}
                - **CUDA 可用:** {'是' if torch.cuda.is_available() else '否'}
                
                ## 📦 依赖
                
                - **Gradio:** {gr.__version__}
                - **PyTorch:** {torch.__version__}
                - **Diffusers:** 已安装
                
                ## 💡 使用说明
                
                1. 首次运行会自动下载 SD 1.5 模型（约4GB）
                2. 输入提示词描述想要生成的图像
                3. 调整参数（尺寸、步数、CFG等）
                4. 点击生成按钮
                5. 等待生成完成
                
                ## ⚡ 性能优化
                
                - 使用 GPU 加速（如果可用）
                - 启用注意力切片节省显存
                - 支持 xformers 优化（如已安装）
                
                ## 👨‍💻 作者
                
                **eddy** - 2025-11-13
                """)
    
    return demo


def main():
    """主函数"""
    print("=" * 70)
    print("Genesis AI - 真实图像生成界面")
    print("=" * 70)
    print()
    
    # 显示设备信息
    print(f"🖥️  设备: {generator.device}")
    print(f"📊 精度: {generator.dtype}")
    
    if torch.cuda.is_available():
        print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"💾 显存: {vram:.1f} GB")
    
    print()
    
    # 显示模型信息
    models = get_models()
    print(f"📦 可用模型:")
    print(f"  - Checkpoints: {len(models['checkpoints'])} 个")
    print(f"  - LoRAs: {len(models['loras'])} 个")
    print(f"  - VAEs: {len(models['vaes'])} 个")
    print()
    
    print("🚀 启动 Gradio 界面...")
    print("=" * 70)
    print()
    
    print("💡 提示:")
    print("   - 在界面中选择模型后，点击「加载模型」按钮")
    print("   - 支持本地 checkpoint 模型和 HuggingFace 模型")
    print("   - HuggingFace 模型首次使用会自动下载")
    print()
    
    # 创建并启动界面
    demo = create_ui()
    
    # 尝试多种启动方式
    try:
        print("尝试启动方式 1: 默认端口 7860...")
        demo.launch(
            server_port=7860,
            share=False,
            inbrowser=True
        )
    except Exception as e:
        print(f"方式 1 失败: {e}")
        print("\n尝试启动方式 2: 自动选择端口...")
        try:
            demo.launch(
                server_port=0,  # 自动选择可用端口
                share=False,
                inbrowser=True
            )
        except Exception as e2:
            print(f"方式 2 失败: {e2}")
            print("\n尝试启动方式 3: 使用公共链接...")
            try:
                demo.queue()  # 启用队列
                demo.launch(
                    share=True,
                    inbrowser=True
                )
            except Exception as e3:
                print(f"所有方式都失败了: {e3}")
                print("\n请尝试:")
                print("1. 关闭其他占用端口的程序")
                print("2. 重启电脑")
                print("3. 检查防火墙设置")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n用户中断")
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
