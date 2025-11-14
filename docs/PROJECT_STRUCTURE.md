# Genesis 项目结构说明

**最后更新:** 2025-11-14

---

## 📁 项目文件结构

### 启动脚本 (项目根目录)

| 文件 | 用途 | 说明 |
|------|------|------|
| **start_webui_integrated.bat** | 启动统一WebUI | ⭐ 推荐使用 |
| **fix_dependencies.bat** | 修复依赖问题 | 解决bitsandbytes冲突 |

**使用方法:**
```batch
# 启动WebUI
start_webui_integrated.bat

# 修复依赖
fix_dependencies.bat
```

---

### 核心应用 (apps/ 目录)

| 文件 | 功能 | 状态 |
|------|------|------|
| **genesis_webui_integrated.py** | 统一WebUI | ⭐ 推荐 - 集成所有功能 |
| **genesis_webui.py** | 基础WebUI模板 | 备用 |
| **gradio_real.py** | SD图像生成 | 单独使用 |
| **gradio_demo.py** | 演示界面 | 测试用 |
| **gradio_simple.py** | 简化界面 | 测试用 |
| **wanvideo_gradio_app.py** | 视频生成 | 单独使用 |
| **start_api_server_real.py** | API服务器 | 后端服务 |

**推荐使用:**
- 统一界面: `genesis_webui_integrated.py` (start_webui_integrated.bat)
- 单独SD: `gradio_real.py`
- 单独视频: `wanvideo_gradio_app.py`

---

### 文档 (项目根目录)

#### 必读文档

| 文档 | 内容 | 适合 |
|------|------|------|
| **README.md** | 项目总览 | 所有人 |
| **PYTHON_ENV.md** | 嵌套环境说明 | ⭐ 重要 - 必读 |
| **QUICKSTART_WEBUI.md** | 5分钟快速开始 | 新手 |
| **INTEGRATION_COMPLETE.md** | 完整集成说明 | 了解架构 |

#### 功能文档

| 文档 | 内容 | 用途 |
|------|------|------|
| **WEBUI_GUIDE.md** | 完整使用指南 | 详细操作 |
| **TROUBLESHOOTING.md** | 故障排除 | 解决问题 |
| **MODEL_PATHS_CONFIG.md** | 模型路径配置 | ComfyUI集成 |

#### 其他文档

| 文档 | 内容 |
|------|------|
| **QUICK_START_CN.md** | 中文快速开始 |
| **README_GRADIO.md** | Gradio说明 |

---

## 🚀 快速开始

### 1. 理解Python环境

**必读:** [PYTHON_ENV.md](PYTHON_ENV.md)

嵌套Python路径:
```
C:\Users\Administrator\Desktop\fork\python313\python.exe
```

### 2. 修复依赖(如需要)

```batch
fix_dependencies.bat
```

### 3. 启动WebUI

```batch
start_webui_integrated.bat
```

### 4. 访问界面

```
http://localhost:7860
```

---

## 📚 文档导航

### 新手入门

1. **[PYTHON_ENV.md](PYTHON_ENV.md)** - 理解嵌套环境 ⭐
2. **[QUICKSTART_WEBUI.md](QUICKSTART_WEBUI.md)** - 5分钟上手
3. **[WEBUI_GUIDE.md](WEBUI_GUIDE.md)** - 详细使用

### 问题解决

1. **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)** - 常见问题
2. **[PYTHON_ENV.md](PYTHON_ENV.md)** - 环境问题

### 高级配置

1. **[MODEL_PATHS_CONFIG.md](MODEL_PATHS_CONFIG.md)** - 模型路径
2. **[INTEGRATION_COMPLETE.md](INTEGRATION_COMPLETE.md)** - 技术细节

---

## 🗂️ 完整目录树

```
original_Genesis/
│
├── 📝 启动脚本
│   ├── start_webui_integrated.bat  ⭐ 主要启动脚本
│   └── fix_dependencies.bat         工具脚本
│
├── 📚 文档
│   ├── README.md                    项目说明
│   ├── PYTHON_ENV.md               ⭐ 嵌套环境说明
│   ├── QUICKSTART_WEBUI.md         快速开始
│   ├── WEBUI_GUIDE.md              使用指南
│   ├── TROUBLESHOOTING.md          故障排除
│   ├── INTEGRATION_COMPLETE.md     集成说明
│   ├── MODEL_PATHS_CONFIG.md       模型配置
│   ├── QUICK_START_CN.md           中文快速开始
│   ├── README_GRADIO.md            Gradio说明
│   └── PROJECT_STRUCTURE.md        本文档
│
├── 📁 apps/ (应用目录)
│   ├── genesis_webui_integrated.py ⭐ 统一WebUI
│   ├── genesis_webui.py             基础模板
│   ├── gradio_real.py               SD生成
│   ├── gradio_demo.py               演示
│   ├── gradio_simple.py             简化版
│   ├── wanvideo_gradio_app.py       视频生成
│   ├── start_api_server_real.py     API服务器
│   └── README.md                    Apps说明
│
├── 📁 api/ (API目录)
│   ├── __init__.py
│   ├── advanced_server.py
│   ├── flask_server.py
│   └── server.py
│
├── 📁 core/ (核心引擎)
│   ├── engine.py
│   ├── config.py
│   ├── pipeline.py
│   └── ...
│
├── 📁 models/ (模型目录)
│   ├── checkpoints/
│   ├── loras/
│   ├── vae/
│   └── ...
│
├── 📁 custom_nodes/ (自定义节点)
│   └── Comfyui/
│       └── ComfyUI-WanVideoWrapper/
│
└── 其他核心目录...
```

---

## ⚙️ 配置文件

| 文件 | 用途 |
|------|------|
| **extra_model_paths.yaml** | 模型路径配置 |
| **requirements.txt** | Python依赖 |
| **requirements_ai.txt** | AI相关依赖 |

---

## 🔧 常用操作

### 安装依赖

```batch
# 设置Python路径(方便操作)
set PYTHON=C:\Users\Administrator\Desktop\fork\python313\python.exe

# 核心依赖
%PYTHON% -m pip install gradio torch torchvision

# Stable Diffusion
%PYTHON% -m pip install diffusers transformers accelerate

# 性能优化
%PYTHON% -m pip install xformers
```

### 启动应用

```batch
# 统一WebUI(推荐)
start_webui_integrated.bat

# 单独应用
%PYTHON% apps\gradio_real.py
%PYTHON% apps\wanvideo_gradio_app.py
```

### 检查状态

```batch
# Python版本
%PYTHON% --version

# 已安装包
%PYTHON% -m pip list

# CUDA状态
%PYTHON% -c "import torch; print(torch.cuda.is_available())"
```

---

## 📊 文件统计

| 类型 | 数量 | 说明 |
|------|------|------|
| **启动脚本 (.bat)** | 2 | 已清理无用脚本 |
| **文档 (.md)** | 10 | 保留有用文档 |
| **应用 (.py in apps/)** | 7 | 包含统一WebUI |
| **核心代码** | - | 完整保留 |

---

## ✅ 已清理的文件

### 删除的启动脚本
- ~~start_demo_ui.bat~~ - 使用统一WebUI代替
- ~~start_real_ui.bat~~ - 使用统一WebUI代替
- ~~start_simple.bat~~ - 使用统一WebUI代替
- ~~start_webui.bat~~ - 使用集成版代替
- ~~test_config.bat~~ - 功能已整合

### 删除的分析文档
- ~~API_IMPROVEMENT_REPORT.md~~ - 技术分析(已完成)
- ~~API_IMPROVEMENTS_SUMMARY.md~~ - 技术分析(已完成)
- ~~APPS_CONSOLIDATION_ANALYSIS.md~~ - 技术分析(已完成)
- ~~INTEGRATION_NOTES.md~~ - 已合并到INTEGRATION_COMPLETE.md
- ~~WEBUI_IMPLEMENTATION.md~~ - 已合并到INTEGRATION_COMPLETE.md

---

## 📝 文件用途速查

### 我应该使用哪个文件?

**启动WebUI:**
```
start_webui_integrated.bat
```

**修复依赖问题:**
```
fix_dependencies.bat
```

**学习环境配置:**
```
PYTHON_ENV.md (必读!)
```

**快速上手:**
```
QUICKSTART_WEBUI.md
```

**详细使用:**
```
WEBUI_GUIDE.md
```

**遇到问题:**
```
TROUBLESHOOTING.md
```

**配置模型路径:**
```
MODEL_PATHS_CONFIG.md
```

**了解技术细节:**
```
INTEGRATION_COMPLETE.md
```

---

## 🎯 推荐学习路径

### 第一步: 理解环境
1. 阅读 [PYTHON_ENV.md](PYTHON_ENV.md)
2. 理解嵌套Python环境

### 第二步: 快速开始
1. 阅读 [QUICKSTART_WEBUI.md](QUICKSTART_WEBUI.md)
2. 运行 `fix_dependencies.bat`
3. 运行 `start_webui_integrated.bat`

### 第三步: 深入学习
1. 阅读 [WEBUI_GUIDE.md](WEBUI_GUIDE.md)
2. 探索各个标签页功能

### 第四步: 解决问题
1. 遇到问题查看 [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
2. 配置模型查看 [MODEL_PATHS_CONFIG.md](MODEL_PATHS_CONFIG.md)

---

## 🔄 维护和更新

### 更新依赖

```batch
set PYTHON=C:\Users\Administrator\Desktop\fork\python313\python.exe
%PYTHON% -m pip install --upgrade gradio diffusers transformers
```

### 清理缓存

```batch
# 清理Python缓存
%PYTHON% -m pip cache purge
```

### 检查版本

```batch
%PYTHON% -m pip list --outdated
```

---

**项目已优化整理完成!**

**主要启动:** `start_webui_integrated.bat`

**环境说明:** [PYTHON_ENV.md](PYTHON_ENV.md)

**快速开始:** [QUICKSTART_WEBUI.md](QUICKSTART_WEBUI.md)

---

**最后更新:** 2025-11-14
**维护者:** eddy
