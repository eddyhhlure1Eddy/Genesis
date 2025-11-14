# Genesis Apps Directory

Modular application architecture for Genesis AI engine.

## 🚀 Main Entry Point

**genesis_webui_integrated.py** - Modular unified interface with dynamic module loading

Launch with: `start.bat` (from project root) or `start_webui_integrated.bat` (from scripts/)

**How it works:**
- Automatically discovers and loads all app modules from `sd_module/` and `wanvideo_module/`
- Each module registers its own UI tab
- Clean separation of concerns
- Easy to add new modules

**Features:**
- ✅ **txt2img**: Full Stable Diffusion generation
- ✅ **img2img**: Coming soon
- ✅ **WanVideo**: Video generation (120 nodes loaded)
- ✅ **Models**: Unified model management
- ✅ **Settings**: System info and configuration

---

## 📦 Modular Structure

### sd_module/
**Stable Diffusion Module**

Contains:
- `__init__.py` - Module entry point
  - `SDGenerator` class
  - `create_sd_tab()` function
  - `SD_AVAILABLE` flag
- `gradio_real.py` - Full implementation

Features:
- Model loading (local + HuggingFace)
- Batch generation
- Progress display
- GPU optimization with xformers

### wanvideo_module/
**WanVideo Module**

Contains:
- `__init__.py` - Module entry point
  - `WanVideoWorkflow` class
  - `create_wanvideo_tab()` function
  - `WAN_VIDEO_AVAILABLE` flag
- `wanvideo_gradio_app.py` - Full implementation

Features:
- 120 ComfyUI-WanVideoWrapper nodes
- T5 text encoding
- Video generation workflow
- Advanced optimization options

---

## 🏗️ Adding New Modules

To create a new module:

1. Create module directory:
```
apps/your_module/
├── __init__.py        # Module entry point
└── implementation.py  # Full implementation (optional)
```

2. In `__init__.py`, export:
```python
# Module availability flag
YOUR_MODULE_AVAILABLE = True

# Your main class
class YourModuleClass:
    def __init__(self):
        pass

# UI creation function
def create_your_tab(instance):
    with gr.Tab("YourModule"):
        # Build your UI here
        pass
    return {}
```

3. The module will be automatically loaded by `genesis_webui_integrated.py`

---

## 📂 old/ Directory

Archived legacy files:
- `genesis_webui.py` - Old basic template
- `genesis_webui_integrated_old.py` - Previous monolithic version
- `genesis_webui_modular.py` - Experimental version
- `gradio_demo.py` - Demo interface
- `gradio_real.py` - Now in sd_module/
- `gradio_simple.py` - Simple version
- `run_comfy_workflow.py` - Workflow runner
- `start_api_server_real.py` - API server

These are kept for reference but not actively maintained.

---

## 🔧 Development

### Module Loading Flow

1. `genesis_webui_integrated.py` starts
2. Dynamically imports modules:
```python
sd_module = importlib.import_module('sd_module')
wanvideo_module = importlib.import_module('wanvideo_module')
```
3. Checks availability flags
4. Instantiates module classes
5. Calls `create_tab()` functions to build UI

### Requirements

**Core:**
```bash
pip install gradio
```

**For SD Module:**
```bash
pip install diffusers transformers accelerate
```

**For WanVideo Module:**
- ComfyUI-WanVideoWrapper nodes in `custom_nodes/`
- Model files in `models/unet/`
- VAE files in `models/vae/`

---

## 📖 Documentation

See [docs/](../docs/) for detailed documentation:
- [QUICKSTART_WEBUI.md](../docs/QUICKSTART_WEBUI.md) - Quick start guide
- [WEBUI_GUIDE.md](../docs/WEBUI_GUIDE.md) - Complete usage guide
- [INTEGRATION_COMPLETE.md](../docs/INTEGRATION_COMPLETE.md) - Technical details
- [TROUBLESHOOTING.md](../docs/TROUBLESHOOTING.md) - Common issues

---

## 🎯 Architecture Benefits

**Before (Monolithic):**
- Single large file (600+ lines)
- Hard to maintain
- Difficult to add features
- Code duplication

**After (Modular):**
- Clean separation by module
- Easy to add new features
- Reusable components
- Dynamic loading
- Main file only 200 lines

---

**Author:** eddy
**Last Updated:** 2025-11-14
