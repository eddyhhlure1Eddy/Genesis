# Genesis 快速开始指南（中文）

## 🚀 使用 ComfyUI 模型（无需重复下载）

### 第一步：配置已完成 ✓

我已经为你创建了配置文件 `extra_model_paths.yaml`，它会让 Genesis 读取你的 ComfyUI 模型文件夹：

```yaml
comfyui:
  base_path: e:\Comfyu3.13---test\ComfyUI\models
  checkpoints: checkpoints
  loras: loras
  vae: vae
  # ... 等等
```

### 第二步：测试配置

运行测试脚本验证配置是否正确：

**Windows:**
```bash
test_config.bat
```

**或者直接运行 Python:**
```bash
python test_model_paths.py
```

你会看到类似输出：
```
Genesis 模型路径配置测试
======================================================================

1. 检查配置的路径:
----------------------------------------------------------------------

checkpoints:
  1. [✓] e:\Comfyu3.13---test\ComfyUI\models\checkpoints
  2. [✓] e:\Comfyu3.13---test\Genesis-main\models\checkpoints

loras:
  1. [✓] e:\Comfyu3.13---test\ComfyUI\models\loras
  2. [✓] e:\Comfyu3.13---test\Genesis-main\models\loras

...
```

### 第三步：开始使用

在你的代码中直接使用模型：

```python
from genesis import GenesisEngine, GenesisConfig
from genesis.core import folder_paths

# 列出所有可用的模型
checkpoints = folder_paths.get_filename_list('checkpoints')
print("可用的模型:", checkpoints)

# 获取模型的完整路径
model_path = folder_paths.get_full_path('checkpoints', 'your_model.safetensors')

# 创建引擎并使用
config = GenesisConfig(device='cuda')
engine = GenesisEngine(config)
engine.initialize()

# ... 使用模型生成图像
```

### 查看示例代码

```bash
python examples/use_comfyui_models.py
```

## 📁 支持的模型类型

Genesis 会自动读取以下类型的模型：

- ✅ **Checkpoints** - Stable Diffusion 主模型
- ✅ **LoRAs** - LoRA 微调模型
- ✅ **VAE** - VAE 模型
- ✅ **ControlNet** - ControlNet 模型
- ✅ **Embeddings** - Textual Inversion
- ✅ **Upscale Models** - 放大模型
- ✅ 等等...

## 🔧 修改配置

如果你的 ComfyUI 在其他位置，编辑 `extra_model_paths.yaml`：

```yaml
comfyui:
  base_path: D:\你的路径\ComfyUI\models  # 修改这里
  checkpoints: checkpoints
  loras: loras
  # ...
```

## 💡 优势

1. **不占用额外空间** - 直接读取 ComfyUI 的模型，不复制文件
2. **自动同步** - ComfyUI 下载新模型后，Genesis 自动可用
3. **灵活配置** - 可以添加多个模型源

## 📚 详细文档

查看完整配置指南：
- [MODEL_PATHS_CONFIG.md](MODEL_PATHS_CONFIG.md) - 详细配置说明
- [README.md](README.md) - Genesis 完整文档

## ❓ 常见问题

**Q: 看不到模型？**
- 检查 `extra_model_paths.yaml` 中的路径是否正确
- 运行 `test_model_paths.py` 查看详细信息
- 确保 ComfyUI models 文件夹中有模型文件

**Q: 需要重启吗？**
- 修改配置文件后需要重启 Genesis

**Q: 会影响 ComfyUI 吗？**
- 不会！Genesis 只是读取模型，不会修改任何文件

## 🎉 开始使用

现在你可以：
1. 使用 ComfyUI 的所有模型
2. 不需要重复下载
3. 节省磁盘空间

祝你使用愉快！

---
**作者**: eddy  
**日期**: 2025-11-13
