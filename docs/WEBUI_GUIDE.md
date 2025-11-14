# Genesis WebUI 使用指南

**类似 Stable Diffusion WebUI (A1111) 的统一界面**

---

## 快速开始

### 启动WebUI

**方法1: 双击启动**
```
start_webui.bat
```

**方法2: 命令行**
```bash
py apps\genesis_webui.py
```

**访问地址:** http://localhost:7860

---

## 界面概览

Genesis WebUI 采用标签页设计,类似 A1111 WebUI:

```
┌─────────────────────────────────────────────────────┐
│ Genesis WebUI                                        │
├─────────────────────────────────────────────────────┤
│ [txt2img] [img2img] [WanVideo] [Extras] [Models] [Settings] │
├─────────────────────────────────────────────────────┤
│                                                      │
│  当前标签页内容                                         │
│                                                      │
└─────────────────────────────────────────────────────┘
```

---

## 标签页功能

### 1. txt2img (文生图)

**核心功能:** 使用 Stable Diffusion 从文本生成图像

#### 左侧面板 - 生成设置

**模型选择:**
- 下拉菜单选择模型
- 支持本地模型和 HuggingFace 模型
- 点击 "Load Model" 加载

**提示词:**
- **Prompt (正向提示词):** 描述想要的内容
- **Negative Prompt (负向提示词):** 描述不想要的内容

**尺寸设置:**
- **Width:** 图像宽度 (256-2048, 步进64)
- **Height:** 图像高度 (256-2048, 步进64)
- 常用尺寸: 512x512, 768x768, 1024x1024

**生成参数:**
- **Sampling Steps:** 采样步数 (1-150)
  - 推荐: 20-30 (质量与速度平衡)
  - 高质量: 40-50
- **CFG Scale:** 提示词引导强度 (1.0-30.0)
  - 推荐: 7.0-9.0
  - 更高 = 更贴近提示词
- **Seed:** 随机种子
  - -1 = 随机
  - 固定值 = 可复现

**批量生成:**
- **Batch Count:** 批次数量 (1-10)
- **Batch Size:** 每批图像数 (1-8)
- 总图像数 = Batch Count × Batch Size

#### 右侧面板 - 输出

**图像画廊:**
- 显示生成的所有图像
- 2列网格布局
- 点击放大查看

**生成信息:**
- 显示使用的参数
- 包含提示词、尺寸、种子等
- 便于复现

#### 示例提示词

界面底部提供预设示例,点击即可快速尝试:

1. **风景:** "a serene mountain landscape at sunset, beautiful colors, 4k"
2. **人物:** "portrait of a beautiful woman, studio lighting, professional photography"
3. **赛博朋克:** "cyberpunk city at night, neon lights, futuristic, highly detailed"
4. **动物:** "cute cat sitting on windowsill, soft lighting, detailed fur"

---

### 2. img2img (图生图)

**状态:** 即将推出

**计划功能:**
- 图像到图像转换
- 修复 (Inpainting)
- 扩展 (Outpainting)
- 草图转图像

---

### 3. WanVideo (视频生成)

**状态:** 即将推出

**计划功能:**
- 文本生成视频
- 视频参数自定义
- 帧控制
- 视频导出

---

### 4. Extras (附加功能)

**状态:** 即将推出

**计划功能:**
- 图像放大 (Upscaling)
- 面部修复
- 色彩校正
- 批量处理

---

### 5. Models (模型管理)

**功能:** 查看和管理所有可用模型

#### 三个子类别

**Checkpoints (主模型):**
- 列出所有可用的 Stable Diffusion 模型
- 显示模型名称
- 支持 .safetensors 和 .ckpt 格式

**LoRAs (微调模型):**
- 列出所有 LoRA 模型
- 用于风格调整和细节控制

**VAEs (变分自编码器):**
- 列出所有 VAE 模型
- 改善图像质量和色彩

#### 模型来源

**HuggingFace 在线模型:**
- runwayml/stable-diffusion-v1-5
- stabilityai/stable-diffusion-2-1
- stabilityai/stable-diffusion-xl-base-1.0

**本地模型:**
- 放置在配置的 checkpoints 文件夹
- 支持 .safetensors 和 .ckpt 格式

#### 刷新列表

点击 "Refresh List" 重新扫描模型文件夹

---

### 6. Settings (设置)

#### 模型路径配置

**配置文件:** `extra_model_paths.yaml`

支持与 ComfyUI 共享模型,避免重复下载

#### 性能设置

**Enable xformers:**
- 启用 xformers 优化 (如果可用)
- 显著提升生成速度
- 需要安装: `pip install xformers`

**Attention Slicing:**
- 启用注意力切片
- 节省显存 (VRAM)
- 允许更大的图像尺寸

#### UI 设置

**Theme (主题):**
- Light: 浅色主题
- Dark: 深色主题
- Auto: 自动跟随系统

**Generation Grid:**
- 启用/禁用生成网格显示

#### 系统信息

实时显示:
- 设备信息 (CPU/GPU)
- 显存大小
- CUDA 版本
- PyTorch 版本
- Gradio 版本
- 已加载的模型
- 可用模型统计

---

## 使用流程

### 基础流程 (txt2img)

1. **启动 WebUI**
   ```
   start_webui.bat
   ```

2. **加载模型**
   - 选择模型 (本地或 HuggingFace)
   - 点击 "Load Model"
   - 等待加载完成

3. **输入提示词**
   - 正向: 描述想要的内容
   - 负向: 描述不想要的内容

4. **调整参数**
   - 尺寸: 根据需求选择
   - 步数: 20-30 通常足够
   - CFG: 7.0-9.0 为佳

5. **生成图像**
   - 点击 "Generate"
   - 等待生成完成
   - 查看结果

6. **迭代优化**
   - 根据结果调整提示词
   - 修改参数
   - 重新生成

---

## 提示词技巧

### 好的提示词结构

```
[主体] + [细节] + [风格] + [质量词]
```

**示例:**
```
a beautiful landscape with mountains and lake,
sunset, golden hour,
highly detailed, 4k, professional photography
```

### 常用质量词

**正向:**
- highly detailed
- 4k, 8k
- masterpiece
- best quality
- professional photography
- photorealistic
- sharp focus

**负向:**
- ugly, bad quality
- blurry, out of focus
- distorted, deformed
- bad anatomy
- watermark, text
- low resolution

### 提示词权重

暂不支持权重语法,计划在后续版本添加。

---

## 参数推荐

### 快速预览 (测试用)
```
尺寸: 512 x 512
步数: 15-20
CFG: 7.0
批次: 1 x 1
```
**生成时间:** ~10-20秒 (GPU)

### 标准质量 (日常使用)
```
尺寸: 768 x 768
步数: 25-30
CFG: 7.5
批次: 1 x 1
```
**生成时间:** ~30-45秒 (GPU)

### 高质量 (最终输出)
```
尺寸: 1024 x 1024
步数: 40-50
CFG: 8.0-9.0
批次: 1 x 1
```
**生成时间:** ~60-90秒 (GPU)

### 批量探索
```
尺寸: 512 x 512
步数: 20
CFG: 7.0
批次: 4 x 2 (8张图)
```
**生成时间:** ~80-160秒 (GPU)

---

## 系统要求

### 最低配置 (CPU 模式)
- Python 3.10+
- 8GB RAM
- 10GB 磁盘空间

**性能:** 很慢 (每张图 5-10 分钟)

### 推荐配置 (GPU 模式)
- Python 3.10+
- 16GB+ RAM
- NVIDIA GPU (6GB+ 显存)
- CUDA 11.8+
- 20GB 磁盘空间

**性能:** 快速 (每张图 10-30 秒)

### 高端配置
- Python 3.11+
- 32GB+ RAM
- NVIDIA GPU (12GB+ 显存)
- CUDA 12.0+
- 50GB+ 磁盘空间

**性能:** 极快 (每张图 5-15 秒)

---

## 常见问题

### Q: 首次启动很慢?
**A:** 首次运行会下载模型 (~4GB),请耐心等待。模型会被缓存,后续启动很快。

### Q: 生成失败?
**A:** 检查:
1. 模型是否成功加载
2. 显存是否足够 (降低分辨率)
3. 参数是否合理
4. 查看控制台错误信息

### Q: 如何使用本地模型?
**A:**
1. 将 .safetensors 或 .ckpt 文件放入 models/checkpoints/
2. 或配置 extra_model_paths.yaml
3. 刷新模型列表
4. 在下拉菜单中选择

### Q: 可以同时生成多张图吗?
**A:** 可以,使用 Batch Count 和 Batch Size:
- Batch Count × Batch Size = 总图像数
- 例: 4 × 2 = 8 张图

### Q: GPU 内存不足怎么办?
**A:**
1. 降低图像尺寸 (512x512)
2. 减少 Batch Size
3. 启用 Attention Slicing (Settings 标签)
4. 关闭其他占用显存的程序

### Q: 如何获得一致的结果?
**A:** 使用固定的 Seed 值:
1. 生成一张满意的图
2. 记录 Seed 值
3. 下次使用相同 Seed + 相同参数

### Q: 支持中文提示词吗?
**A:** 不建议。Stable Diffusion 主要针对英文训练,中文效果较差。建议使用英文提示词。

---

## 快捷键 (计划中)

暂无快捷键支持,计划在后续版本添加。

---

## 对比其他界面

| 功能 | Genesis WebUI | A1111 WebUI | ComfyUI |
|------|---------------|-------------|---------|
| **易用性** | ★★★★★ | ★★★★★ | ★★★☆☆ |
| **txt2img** | ✓ | ✓ | ✓ |
| **img2img** | 计划中 | ✓ | ✓ |
| **Batch** | ✓ | ✓ | ✓ |
| **LoRA** | 计划中 | ✓ | ✓ |
| **ControlNet** | 计划中 | ✓ | ✓ |
| **工作流** | 计划中 | ✗ | ✓ |
| **视频生成** | ✓ | ✗ | 部分 |
| **API** | ✓ | ✓ | ✓ |

---

## 更新计划

### v1.1 (近期)
- [ ] img2img 完整实现
- [ ] LoRA 支持
- [ ] 提示词权重语法
- [ ] 更多采样器

### v1.2 (中期)
- [ ] ControlNet 支持
- [ ] Inpainting
- [ ] Outpainting
- [ ] 图像放大

### v1.3 (长期)
- [ ] WanVideo 完整集成
- [ ] 工作流编辑器
- [ ] 插件系统
- [ ] 云端同步

---

## 技术支持

**文档:**
- README.md
- QUICK_START_CN.md
- MODEL_PATHS_CONFIG.md

**日志:**
- 查看控制台输出
- 错误信息会显示在界面

**反馈:**
- 报告问题到 GitHub Issues
- 提供详细的错误信息和步骤

---

## 开发者信息

**作者:** eddy
**版本:** 1.0.0
**日期:** 2025-11-13
**许可:** MIT License

**技术栈:**
- Gradio 5.x - Web UI 框架
- PyTorch - 深度学习框架
- Diffusers - Stable Diffusion 实现
- Flask - API 服务器 (可选)

---

**享受创作!**
