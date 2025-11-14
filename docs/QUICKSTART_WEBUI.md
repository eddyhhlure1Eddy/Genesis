# Genesis WebUI 快速启动指南

**5分钟开始使用统一界面**

---

## 步骤1: 安装依赖

```bash
pip install gradio diffusers transformers accelerate
```

**可选 (性能优化):**
```bash
pip install xformers
```

---

## 步骤2: 启动 WebUI

**Windows:**
```
双击 start_webui.bat
```

**或命令行:**
```bash
py apps\genesis_webui.py
```

---

## 步骤3: 访问界面

浏览器自动打开,或手动访问:
```
http://localhost:7860
```

---

## 步骤4: 加载模型

1. 进入 **txt2img** 标签页
2. 在 "Model" 下拉菜单选择:
   - `HF:runwayml/stable-diffusion-v1-5` (推荐,首次会下载~4GB)
   - 或本地模型
3. 点击 **"Load Model"**
4. 等待加载完成

---

## 步骤5: 生成第一张图

**使用预设示例:**
1. 滚动到页面底部 "Example Prompts"
2. 点击任意示例
3. 点击 **"Generate"** 按钮
4. 等待生成完成 (~20秒)

**或自定义:**
1. 在 "Prompt" 输入:
   ```
   a beautiful cat, fluffy fur, cute, professional photography
   ```
2. 在 "Negative Prompt" 输入:
   ```
   ugly, blurry, bad quality
   ```
3. 保持默认参数
4. 点击 **"Generate"**

---

## 界面导航

```
┌───────────────────────────────────────┐
│ Genesis WebUI                          │
├───────────────────────────────────────┤
│ [txt2img] ← 从这里开始                 │
│ [img2img] ← 即将推出                   │
│ [WanVideo] ← 视频生成                  │
│ [Extras] ← 图像增强                    │
│ [Models] ← 查看所有模型                │
│ [Settings] ← 系统设置                  │
└───────────────────────────────────────┘
```

---

## txt2img 标签页布局

```
┌─────────────────┬─────────────────┐
│ 左侧: 设置       │ 右侧: 输出       │
│                 │                 │
│ • 模型选择       │ • 图像画廊       │
│ • 提示词        │ • 生成信息       │
│ • 尺寸参数      │                 │
│ • 生成设置      │                 │
│ • 批量设置      │                 │
│ • 种子         │                 │
│                 │                 │
│ [Generate]      │                 │
│                 │                 │
│ 示例提示词      │                 │
└─────────────────┴─────────────────┘
```

---

## 基础参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| **Width** | 512 | 图像宽度 |
| **Height** | 512 | 图像高度 |
| **Steps** | 20 | 采样步数 (更多=更好质量) |
| **CFG Scale** | 7.0 | 提示词遵循程度 |
| **Seed** | -1 | 随机种子 (-1=随机) |
| **Batch Count** | 1 | 批次数量 |
| **Batch Size** | 1 | 每批图像数 |

---

## 推荐参数组合

### 快速测试
```
尺寸: 512 x 512
步数: 15
CFG: 7.0
批次: 1 x 1
时间: ~10秒
```

### 标准质量
```
尺寸: 768 x 768
步数: 25
CFG: 7.5
批次: 1 x 1
时间: ~30秒
```

### 高质量
```
尺寸: 1024 x 1024
步数: 40
CFG: 8.5
批次: 1 x 1
时间: ~60秒
```

---

## 提示词技巧

### ✅ 好的提示词
```
主体 + 细节 + 风格 + 质量词

示例:
"a majestic mountain landscape,
snow-capped peaks, clear blue sky,
golden hour lighting,
highly detailed, 4k, professional photography"
```

### ✅ 好的负向提示词
```
ugly, blurry, low quality, distorted,
bad anatomy, watermark, text
```

### 常用质量词
- highly detailed
- 4k, 8k
- masterpiece
- best quality
- professional photography
- photorealistic

---

## 常见问题

**Q: 首次启动要等多久?**
A: 首次下载模型约5-10分钟 (取决于网速),后续启动很快。

**Q: 生成一张图要多久?**
A: GPU: 10-30秒 | CPU: 5-10分钟

**Q: 需要多少显存?**
A: 最少4GB,推荐6GB+

**Q: 可以用 CPU 吗?**
A: 可以但很慢,强烈推荐使用 GPU。

**Q: 如何使用本地模型?**
A: 将 .safetensors 文件放入 models/checkpoints/ 文件夹,刷新模型列表。

---

## 下一步

探索其他功能:

1. **Models 标签页** - 查看所有可用模型
2. **Settings 标签页** - 查看系统信息,调整设置
3. **批量生成** - 调整 Batch Count 和 Batch Size
4. **固定种子** - 使用相同种子复现图像

---

## 获取帮助

**详细文档:** [WEBUI_GUIDE.md](WEBUI_GUIDE.md)

**配置模型路径:** [MODEL_PATHS_CONFIG.md](MODEL_PATHS_CONFIG.md)

**API文档:** [apps/README.md](apps/README.md)

---

**开始创作!**

作者: eddy | 版本: 1.0.0 | 日期: 2025-11-13
