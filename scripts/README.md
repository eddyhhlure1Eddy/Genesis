# Scripts Directory

Launch scripts and utility tools for Genesis.

## Launch Scripts

### start_webui_integrated.bat
Main launcher for the integrated WebUI interface.

**Usage:**
```batch
scripts\start_webui_integrated.bat
```

**What it does:**
- Launches the unified WebUI with all integrated features
- Starts Gradio server on http://localhost:7860
- Uses nested Python environment

## Utility Scripts

### fix_dependencies.bat
Fix common dependency conflicts.

**Usage:**
```batch
scripts\fix_dependencies.bat
```

**What it fixes:**
- Removes bitsandbytes (causes triton.ops errors on Windows)
- Updates diffusers, transformers, accelerate to latest versions
- Resolves Windows compatibility issues

## Running Scripts

All scripts use the nested Python environment located at:
```
C:\Users\Administrator\Desktop\fork\python313\python.exe
```

To run from project root:
```batch
cd C:\Users\Administrator\Desktop\fork\original_Genesis
scripts\start_webui_integrated.bat
```
