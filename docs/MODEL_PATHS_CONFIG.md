# Genesis 模型路径配置指南

## 概述

Genesis 支持复用 ComfyUI 或其他 AI 工具的模型文件夹，无需重复下载模型。

## 配置方法

### 方法 1: 使用配置文件（推荐）

1. **编辑 `extra_model_paths.yaml` 文件**
   
   该文件已在 Genesis 根目录创建，内容如下：

   ```yaml
   comfyui:
     base_path: e:\Comfyu3.13---test\ComfyUI\models
     checkpoints: checkpoints
     loras: loras
     vae: vae
     # ... 其他模型类型
   ```

2. **修改路径**（如果需要）
   
   如果你的 ComfyUI 在其他位置，修改 `base_path` 为你的实际路径：
   
   ```yaml
   comfyui:
     base_path: D:\AI\ComfyUI\models  # 修改为你的路径
   ```

3. **启动 Genesis**
   
   配置会自动加载，你会看到类似的日志：
   
   ```
   [Genesis] Added extra model path: checkpoints -> e:\Comfyu3.13---test\ComfyUI\models\checkpoints
   [Genesis] Added extra model path: loras -> e:\Comfyu3.13---test\ComfyUI\models\loras
   [Genesis] Successfully loaded extra model paths from extra_model_paths.yaml
   ```

### 方法 2: 代码中动态添加

在你的 Python 代码中：

```python
from genesis.core import folder_paths

# 添加 ComfyUI 的 checkpoints 路径
folder_paths.add_model_folder_path('checkpoints', r'e:\Comfyu3.13---test\ComfyUI\models\checkpoints')

# 添加 ComfyUI 的 loras 路径
folder_paths.add_model_folder_path('loras', r'e:\Comfyu3.13---test\ComfyUI\models\loras')

# 添加 ComfyUI 的 VAE 路径
folder_paths.add_model_folder_path('vae', r'e:\Comfyu3.13---test\ComfyUI\models\vae')
```

## 支持的模型类型

Genesis 支持以下模型类型的路径配置：

| 模型类型 | 说明 |
|---------|------|
| `checkpoints` | Stable Diffusion 主模型 |
| `loras` | LoRA 模型 |
| `vae` | VAE 模型 |
| `text_encoders` / `clip` | 文本编码器 |
| `diffusion_models` / `unet` | UNet 扩散模型 |
| `clip_vision` | CLIP Vision 模型 |
| `embeddings` | Textual Inversion 嵌入 |
| `controlnet` | ControlNet 模型 |
| `t2i_adapter` | T2I-Adapter 模型 |
| `upscale_models` | 放大模型 |
| `hypernetworks` | 超网络 |
| `photomaker` | PhotoMaker 模型 |
| `gligen` | GLIGEN 模型 |
| `style_models` | 风格模型 |
| `diffusers` | Diffusers 格式模型 |

## 路径优先级

当配置多个路径时，Genesis 会按以下优先级搜索模型：

1. **extra_model_paths.yaml 中配置的路径**（优先级最高）
2. Genesis 自己的 models 文件夹
3. 代码中动态添加的路径

## 添加多个模型源

你可以在 `extra_model_paths.yaml` 中配置多个模型源：

```yaml
# ComfyUI 模型
comfyui:
  base_path: e:\Comfyu3.13---test\ComfyUI\models
  checkpoints: checkpoints
  loras: loras

# 自定义模型库
my_models:
  base_path: D:\MyAIModels
  checkpoints: sd_models
  loras: lora_collection

# Stable Diffusion WebUI 模型
webui:
  base_path: C:\stable-diffusion-webui\models
  checkpoints: Stable-diffusion
  loras: Lora
  vae: VAE
```

## 验证配置

启动 Genesis 后，可以通过以下代码验证模型路径：

```python
from genesis.core import folder_paths

# 查看所有 checkpoints 路径
print("Checkpoints paths:", folder_paths.get_folder_paths('checkpoints'))

# 列出所有可用的 checkpoint 文件
print("Available checkpoints:", folder_paths.get_filename_list('checkpoints'))

# 获取特定模型的完整路径
model_path = folder_paths.get_full_path('checkpoints', 'your_model.safetensors')
print("Model path:", model_path)
```

## 常见问题

### Q: 配置后看不到模型？

**A:** 检查以下几点：
1. 路径是否正确（Windows 使用 `\` 或 `\\`，或使用 `/`）
2. 模型文件扩展名是否支持（`.safetensors`, `.ckpt`, `.pt`, `.pth`, `.bin`）
3. 查看启动日志是否有加载成功的提示

### Q: 可以使用相对路径吗？

**A:** 建议使用绝对路径以避免路径解析问题。

### Q: 修改配置后需要重启吗？

**A:** 是的，修改 `extra_model_paths.yaml` 后需要重启 Genesis 才能生效。

### Q: 会复制模型文件吗？

**A:** 不会！Genesis 只是读取配置的路径，不会复制或移动任何模型文件。

## 示例：完整配置

```yaml
# extra_model_paths.yaml
comfyui:
  base_path: e:\Comfyu3.13---test\ComfyUI\models
  checkpoints: checkpoints
  configs: configs
  loras: loras
  vae: vae
  text_encoders: text_encoders
  clip: clip
  diffusion_models: diffusion_models
  unet: unet
  clip_vision: clip_vision
  style_models: style_models
  embeddings: embeddings
  diffusers: diffusers
  vae_approx: vae_approx
  controlnet: controlnet
  t2i_adapter: t2i_adapter
  gligen: gligen
  upscale_models: upscale_models
  hypernetworks: hypernetworks
  photomaker: photomaker
  classifiers: classifiers
  model_patches: model_patches
  audio_encoders: audio_encoders
```

## 技术支持

如有问题，请检查：
1. Genesis 启动日志
2. 路径是否存在
3. 文件权限是否正确

---

**作者**: eddy  
**更新日期**: 2025-11-13
