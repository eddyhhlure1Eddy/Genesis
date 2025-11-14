#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Genesis Gradio 演示界面
兼容 Gradio 5.x 版本
"""

import sys
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).parent))

try:
    import gradio as gr
    print(f"Gradio 版本: {gr.__version__}")
except ImportError:
    print("请安装 Gradio: pip install gradio")
    sys.exit(1)

# 导入 folder_paths
import importlib.util
spec = importlib.util.spec_from_file_location(
    "folder_paths", 
    Path(__file__).parent / "core" / "folder_paths.py"
)
folder_paths = importlib.util.module_from_spec(spec)
spec.loader.exec_module(folder_paths)


def get_models():
    """获取模型列表"""
    return {
        'checkpoints': folder_paths.get_filename_list('checkpoints'),
        'loras': folder_paths.get_filename_list('loras'),
        'vaes': folder_paths.get_filename_list('vae')
    }


def generate_demo(
    prompt,
    negative_prompt,
    checkpoint,
    lora,
    width,
    height,
    steps,
    cfg_scale,
    seed,
    progress=gr.Progress()
):
    """演示生成功能"""
    try:
        # 模拟生成过程
        progress(0, desc="初始化...")
        time.sleep(0.3)
        
        progress(0.2, desc="加载模型...")
        time.sleep(0.3)
        
        progress(0.4, desc="编码提示词...")
        time.sleep(0.3)
        
        progress(0.6, desc="生成中...")
        time.sleep(0.5)
        
        progress(0.9, desc="后处理...")
        time.sleep(0.3)
        
        progress(1.0, desc="完成!")
        
        result = f"""
## ✅ 生成完成（演示模式）

### 📝 提示词
**正向:** {prompt}

**负向:** {negative_prompt}

### 🎨 模型设置
- **Checkpoint:** {checkpoint}
- **LoRA:** {lora}

### ⚙️ 生成参数
- **尺寸:** {width} x {height}
- **步数:** {steps}
- **CFG Scale:** {cfg_scale}
- **种子:** {seed if seed >= 0 else "随机"}

---
💡 这是演示界面，展示 Genesis 的参数配置功能。
要实际生成图像，请集成完整的 Genesis 引擎。
"""
        return result
        
    except Exception as e:
        return f"❌ 错误: {str(e)}"


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
    
    info += "\n### LoRAs\n"
    if models['loras']:
        for i, name in enumerate(models['loras'][:10], 1):
            info += f"{i}. `{name}`\n"
        if len(models['loras']) > 10:
            info += f"\n... 还有 {len(models['loras']) - 10} 个\n"
    else:
        info += "*未找到模型*\n"
    
    info += "\n### VAEs\n"
    if models['vaes']:
        for i, name in enumerate(models['vaes'][:10], 1):
            info += f"{i}. `{name}`\n"
        if len(models['vaes']) > 10:
            info += f"\n... 还有 {len(models['vaes']) - 10} 个\n"
    else:
        info += "*未找到模型*\n"
    
    return info


# 创建界面
def create_ui():
    models = get_models()
    checkpoint_choices = ["(不使用)"] + models['checkpoints']
    lora_choices = ["(不使用)"] + models['loras']
    
    with gr.Blocks(
        title="Genesis AI Demo",
        theme=gr.themes.Soft(
            primary_hue="purple",
            secondary_hue="blue",
        )
    ) as demo:
        
        gr.Markdown("""
        # 🎨 Genesis AI 图像生成器
        
        轻量级、高性能的 AI 生成引擎演示界面
        """)
        
        with gr.Tabs():
            # 生成标签页
            with gr.Tab("🎨 图像生成"):
                with gr.Row():
                    with gr.Column(scale=2):
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
                            value="ugly, blurry, low quality, distorted"
                        )
                        
                        with gr.Row():
                            checkpoint = gr.Dropdown(
                                label="Checkpoint 模型",
                                choices=checkpoint_choices,
                                value=checkpoint_choices[0]
                            )
                            
                            lora = gr.Dropdown(
                                label="LoRA 模型",
                                choices=lora_choices,
                                value=lora_choices[0]
                            )
                        
                        gr.Markdown("### ⚙️ 参数设置")
                        
                        with gr.Row():
                            width = gr.Slider(
                                label="宽度",
                                minimum=256,
                                maximum=2048,
                                step=64,
                                value=512
                            )
                            
                            height = gr.Slider(
                                label="高度",
                                minimum=256,
                                maximum=2048,
                                step=64,
                                value=512
                            )
                        
                        with gr.Row():
                            steps = gr.Slider(
                                label="采样步数",
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
                        gr.Markdown("### 📊 生成结果")
                        
                        output = gr.Markdown(
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
                
                # 连接生成按钮
                generate_btn.click(
                    fn=generate_demo,
                    inputs=[
                        prompt, negative_prompt,
                        checkpoint, lora,
                        width, height,
                        steps, cfg_scale, seed
                    ],
                    outputs=output
                )
            
            # 模型管理标签页
            with gr.Tab("📦 模型管理"):
                gr.Markdown("### 可用模型列表")
                
                model_info = gr.Markdown(value=show_models())
                
                refresh_btn = gr.Button("🔄 刷新模型列表", variant="secondary")
                refresh_btn.click(
                    fn=show_models,
                    outputs=model_info
                )
                
                gr.Markdown("""
                ### 📁 模型路径配置
                
                编辑 `extra_model_paths.yaml` 文件来添加模型路径：
                
                ```yaml
                comfyui:
                  base_path: E:\\你的路径\\ComfyUI\\models
                  checkpoints: checkpoints
                  loras: loras
                  vae: vae
                ```
                
                修改后重启此界面即可。
                """)
            
            # 关于标签页
            with gr.Tab("ℹ️ 关于"):
                gr.Markdown("""
                # Genesis AI Engine
                
                ## 🚀 特性
                
                - **轻量级架构** - 纯执行引擎，无 UI 依赖
                - **高性能优化** - GPU 加速，支持 TF32/FP8
                - **灵活集成** - 可集成到任何应用
                - **模型复用** - 支持读取 ComfyUI 模型
                
                ## 📦 支持的模型格式
                
                - SafeTensors (.safetensors)
                - PyTorch (.pt, .pth, .ckpt)
                - Pickle (.pkl)
                
                ## 🔧 系统要求
                
                **推荐配置:**
                - Python 3.11+
                - 16GB+ RAM
                - NVIDIA RTX 30/40 系列 GPU
                - CUDA 12.0+
                
                ## 👨‍💻 作者
                
                **eddy** - 2025-11-13
                
                ---
                
                ### 💡 提示
                
                这是一个演示界面，展示 Genesis 的参数配置和界面设计。
                要使用完整的图像生成功能，请集成 Genesis 引擎。
                """)
    
    return demo


def main():
    """主函数"""
    print("=" * 70)
    print("Genesis AI Gradio 演示界面")
    print("=" * 70)
    print()
    
    # 显示模型信息
    models = get_models()
    print(f"📦 可用模型:")
    print(f"  - Checkpoints: {len(models['checkpoints'])} 个")
    print(f"  - LoRAs: {len(models['loras'])} 个")
    print(f"  - VAEs: {len(models['vaes'])} 个")
    print()
    
    if sum(len(v) for v in models.values()) == 0:
        print("⚠️  提示: 未找到模型文件")
        print("   请在 extra_model_paths.yaml 中配置模型路径")
        print()
    
    print("🚀 启动 Gradio 界面...")
    print("=" * 70)
    print()
    
    # 创建并启动界面
    demo = create_ui()
    
    try:
        # 尝试方法1: 使用 share=True
        print("尝试启动方式 1: 使用公共链接...")
        demo.launch(
            server_port=7861,
            inbrowser=True,
            share=True  # 使用公共链接
        )
    except Exception as e:
        print(f"\n❌ 方式1失败: {e}")
        print("\n尝试启动方式 2: 使用 queue...")
        try:
            demo.queue()
            demo.launch(
                server_port=7862,
                inbrowser=True,
                share=False
            )
        except Exception as e2:
            print(f"\n❌ 方式2失败: {e2}")
            print("\n尝试启动方式 3: 最简单模式...")
            try:
                demo.launch()
            except Exception as e3:
                print(f"\n❌ 所有方式都失败了: {e3}")
                print("\n可能的解决方案:")
                print("1. 检查防火墙是否阻止了 Python")
                print("2. 尝试以管理员身份运行")
                print("3. 检查是否有代理设置")
                print("4. 尝试: pip install --upgrade gradio")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n用户中断")
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
