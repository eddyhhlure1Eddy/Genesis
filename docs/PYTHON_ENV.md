# Python 嵌套环境说明

**项目使用嵌套的Python环境,不是系统全局Python**

---

## 环境路径

**Python可执行文件:**
```
C:\Users\Administrator\Desktop\fork\python313\python.exe
```

**环境位置:**
```
C:\Users\Administrator\Desktop\fork\python313\
```

---

## 如何使用

### 运行Python脚本

**正确方式:**
```batch
C:\Users\Administrator\Desktop\fork\python313\python.exe script.py
```

**错误方式:**
```batch
py script.py          # 错误 - 会使用系统Python
python script.py      # 错误 - 会使用系统Python
```

### 安装依赖包

**正确方式:**
```batch
C:\Users\Administrator\Desktop\fork\python313\python.exe -m pip install package_name
```

**示例:**
```batch
# 安装Gradio
C:\Users\Administrator\Desktop\fork\python313\python.exe -m pip install gradio

# 安装Diffusers
C:\Users\Administrator\Desktop\fork\python313\python.exe -m pip install diffusers transformers accelerate

# 卸载包
C:\Users\Administrator\Desktop\fork\python313\python.exe -m pip uninstall package_name -y
```

### 查看已安装的包

```batch
C:\Users\Administrator\Desktop\fork\python313\python.exe -m pip list
```

### 检查包是否安装

```batch
C:\Users\Administrator\Desktop\fork\python313\python.exe -m pip show package_name
```

---

## 启动脚本配置

所有启动脚本已配置使用嵌套环境:

### start_webui_integrated.bat
```batch
C:\Users\Administrator\Desktop\fork\python313\python.exe apps\genesis_webui_integrated.py
```

### fix_dependencies.bat
```batch
C:\Users\Administrator\Desktop\fork\python313\python.exe -m pip install ...
```

---

## 为什么使用嵌套环境

**优点:**
- ✅ 独立的依赖管理
- ✅ 不影响系统Python
- ✅ 避免版本冲突
- ✅ 便于项目迁移

**注意事项:**
- ⚠️ 必须使用完整路径
- ⚠️ 不能使用 `py` 或 `python` 命令
- ⚠️ 所有脚本必须指定完整路径

---

## 环境变量设置(可选)

如果想简化命令,可以设置临时环境变量:

**PowerShell:**
```powershell
$env:PYTHON = "C:\Users\Administrator\Desktop\fork\python313\python.exe"
& $env:PYTHON script.py
& $env:PYTHON -m pip install package
```

**CMD:**
```batch
set PYTHON=C:\Users\Administrator\Desktop\fork\python313\python.exe
%PYTHON% script.py
%PYTHON% -m pip install package
```

---

## 常用命令速查

| 操作 | 命令 |
|------|------|
| **运行脚本** | `C:\Users\Administrator\Desktop\fork\python313\python.exe script.py` |
| **安装包** | `C:\Users\Administrator\Desktop\fork\python313\python.exe -m pip install pkg` |
| **卸载包** | `C:\Users\Administrator\Desktop\fork\python313\python.exe -m pip uninstall pkg -y` |
| **升级包** | `C:\Users\Administrator\Desktop\fork\python313\python.exe -m pip install --upgrade pkg` |
| **查看包列表** | `C:\Users\Administrator\Desktop\fork\python313\python.exe -m pip list` |
| **检查Python版本** | `C:\Users\Administrator\Desktop\fork\python313\python.exe --version` |
| **检查CUDA** | `C:\Users\Administrator\Desktop\fork\python313\python.exe -c "import torch; print(torch.cuda.is_available())"` |

---

## 依赖安装指南

### 核心依赖
```batch
set PYTHON=C:\Users\Administrator\Desktop\fork\python313\python.exe

%PYTHON% -m pip install gradio
%PYTHON% -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Stable Diffusion
```batch
%PYTHON% -m pip install diffusers transformers accelerate
```

### 性能优化(可选)
```batch
%PYTHON% -m pip install xformers
```

### 修复冲突
```batch
# 卸载有问题的包
%PYTHON% -m pip uninstall bitsandbytes triton -y

# 重装核心依赖
%PYTHON% -m pip install --upgrade diffusers transformers accelerate
```

---

## 故障排查

### 问题: "py不是内部或外部命令"

**原因:** 使用了 `py` 而不是完整路径

**解决:**
```batch
# 错误
py script.py

# 正确
C:\Users\Administrator\Desktop\fork\python313\python.exe script.py
```

### 问题: 包安装到了错误的Python

**原因:** 使用了系统pip

**解决:**
```batch
# 错误
pip install package

# 正确
C:\Users\Administrator\Desktop\fork\python313\python.exe -m pip install package
```

### 问题: 导入错误 "No module named xxx"

**检查:**
```batch
# 1. 检查包是否安装在嵌套环境
C:\Users\Administrator\Desktop\fork\python313\python.exe -m pip show package_name

# 2. 如果未安装,安装到嵌套环境
C:\Users\Administrator\Desktop\fork\python313\python.exe -m pip install package_name
```

---

## 环境信息检查

### Python版本
```batch
C:\Users\Administrator\Desktop\fork\python313\python.exe --version
```
**期望输出:** Python 3.13.x

### 已安装包
```batch
C:\Users\Administrator\Desktop\fork\python313\python.exe -m pip list
```

### CUDA可用性
```batch
C:\Users\Administrator\Desktop\fork\python313\python.exe -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### GPU信息
```batch
C:\Users\Administrator\Desktop\fork\python313\python.exe -c "import torch; print(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else print('No GPU')"
```

---

## 项目启动流程

### 1. 检查环境
```batch
C:\Users\Administrator\Desktop\fork\python313\python.exe --version
```

### 2. 安装依赖(首次)
```batch
fix_dependencies.bat
```

### 3. 启动WebUI
```batch
start_webui_integrated.bat
```

---

## 注意事项

### ✅ 正确做法
- 使用完整Python路径
- 使用 `-m pip` 安装包
- 所有脚本指定完整路径

### ❌ 错误做法
- 使用 `py` 或 `python` 命令
- 直接使用 `pip install`
- 混用系统Python和嵌套Python

---

**环境路径:** `C:\Users\Administrator\Desktop\fork\python313\`

**Python版本:** 3.13

**最后更新:** 2025-11-14
