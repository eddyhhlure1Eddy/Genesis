# Genesis WebUI 故障排除

---

## 常见错误解决方案

### ❌ 错误: `No module named 'triton.ops'`

**完整错误信息:**
```
ModuleNotFoundError: No module named 'triton.ops'
RuntimeError: Failed to import diffusers.loaders.single_file
```

**原因:**
bitsandbytes库与triton版本不兼容,导致diffusers无法导入。

**解决方案:**

#### 方法1: 运行修复脚本 (推荐)
```
双击运行: fix_dependencies.bat
```

#### 方法2: 手动修复
```bash
# 使用嵌套Python环境
C:\Users\Administrator\Desktop\fork\python313\python.exe -m pip uninstall bitsandbytes -y
C:\Users\Administrator\Desktop\fork\python313\python.exe -m pip install --upgrade diffusers transformers accelerate
```

#### 方法3: 完全重装依赖
```bash
# 卸载问题库
C:\Users\Administrator\Desktop\fork\python313\python.exe -m pip uninstall bitsandbytes triton -y

# 重装核心依赖
C:\Users\Administrator\Desktop\fork\python313\python.exe -m pip install torch torchvision
C:\Users\Administrator\Desktop\fork\python313\python.exe -m pip install diffusers transformers accelerate gradio
```

**验证修复:**
```bash
C:\Users\Administrator\Desktop\fork\python313\python.exe -c "from diffusers import StableDiffusionPipeline; print('OK')"
```

---

### ❌ 错误: 启动脚本找不到Python

**错误:**
```
'py' 不是内部或外部命令
```

**原因:**
使用了嵌套Python环境,不是系统全局Python。

**解决方案:**
已修复! `start_webui.bat` 现在使用完整路径:
```
C:\Users\Administrator\Desktop\fork\python313\python.exe
```

---

### ❌ 错误: `ModuleNotFoundError: No module named 'gradio'`

**解决方案:**
```bash
C:\Users\Administrator\Desktop\fork\python313\python.exe -m pip install gradio
```

---

### ❌ 错误: `ModuleNotFoundError: No module named 'torch'`

**解决方案:**
```bash
C:\Users\Administrator\Desktop\fork\python313\python.exe -m pip install torch torchvision
```

---

### ❌ 错误: 端口7860已被占用

**错误信息:**
```
OSError: [Errno 98] Address already in use
```

**解决方案:**

#### 方法1: 关闭占用端口的程序
```bash
# 查找占用端口的进程
netstat -ano | findstr :7860

# 结束进程 (PID是上面命令的最后一列)
taskkill /PID <进程ID> /F
```

#### 方法2: 修改端口
编辑 `apps/genesis_webui.py`:
```python
os.environ['GRADIO_SERVER_PORT'] = '7861'  # 改为其他端口
```

---

### ❌ 错误: CUDA out of memory

**错误信息:**
```
RuntimeError: CUDA out of memory
```

**解决方案:**

1. **降低图像尺寸**
   - 从 1024x1024 降到 512x512

2. **减少批次大小**
   - Batch Count: 1
   - Batch Size: 1

3. **启用内存优化**
   - Settings标签 → Enable Attention Slicing

4. **关闭其他GPU程序**
   - 关闭浏览器硬件加速
   - 关闭其他AI应用

---

### ❌ 错误: 模型加载失败

**错误信息:**
```
Failed to load model: [Error details]
```

**解决方案:**

#### HuggingFace模型
1. **检查网络连接**
2. **使用代理** (如果在中国)
   ```bash
   set HF_ENDPOINT=https://hf-mirror.com
   ```
3. **手动下载模型**

#### 本地模型
1. **检查文件路径**
   - 确保模型在 `models/checkpoints/` 目录
   - 或配置 `extra_model_paths.yaml`

2. **检查文件格式**
   - 支持: .safetensors, .ckpt
   - 文件完整,未损坏

3. **刷新模型列表**
   - Models标签 → Refresh List

---

### ❌ 生成速度非常慢

**可能原因和解决方案:**

#### 1. 使用CPU而非GPU
**检查:**
- Settings标签查看 Device 信息

**解决:**
- 确保安装了CUDA版本的PyTorch
```bash
C:\Users\Administrator\Desktop\fork\python313\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

#### 2. 未启用优化
**解决:**
- Settings标签 → 启用 xformers
- Settings标签 → 启用 Attention Slicing

#### 3. 参数设置过高
**解决:**
- 降低采样步数到 20-25
- 降低图像尺寸到 512x512

---

### ❌ 生成的图像质量差

**解决方案:**

1. **优化提示词**
   ```
   正向: 添加质量词 (4k, highly detailed, masterpiece)
   负向: 添加 (ugly, blurry, low quality)
   ```

2. **调整参数**
   - 增加采样步数 (30-40)
   - 调整CFG Scale (7.0-9.0)

3. **尝试不同模型**
   - SD 1.5 vs SD 2.1
   - 专业调优的模型

---

## 完整依赖安装

### 最小安装 (仅WebUI)
```bash
set PYTHON=C:\Users\Administrator\Desktop\fork\python313\python.exe
%PYTHON% -m pip install gradio
```

### 标准安装 (含SD生成)
```bash
set PYTHON=C:\Users\Administrator\Desktop\fork\python313\python.exe
%PYTHON% -m pip install gradio
%PYTHON% -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
%PYTHON% -m pip install diffusers transformers accelerate
```

### 完整安装 (含优化)
```bash
set PYTHON=C:\Users\Administrator\Desktop\fork\python313\python.exe
%PYTHON% -m pip install gradio
%PYTHON% -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
%PYTHON% -m pip install diffusers transformers accelerate
%PYTHON% -m pip install xformers
```

---

## 环境检查

### 检查Python版本
```bash
C:\Users\Administrator\Desktop\fork\python313\python.exe --version
```
**要求:** Python 3.10 或更高

### 检查已安装的包
```bash
C:\Users\Administrator\Desktop\fork\python313\python.exe -m pip list
```

### 检查CUDA可用性
```bash
C:\Users\Administrator\Desktop\fork\python313\python.exe -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### 检查GPU信息
```bash
C:\Users\Administrator\Desktop\fork\python313\python.exe -c "import torch; print(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else print('No GPU')"
```

---

## 日志和调试

### 查看详细错误
运行时查看控制台输出,包含:
- 导入错误
- 模型加载信息
- 生成进度
- 错误堆栈

### 启用调试模式
编辑 `apps/genesis_webui.py`:
```python
demo.launch(
    server_name="0.0.0.0",
    server_port=7860,
    share=False,
    inbrowser=True,
    show_error=True,
    debug=True  # 添加这行
)
```

---

## 获取帮助

### 文档
- [QUICKSTART_WEBUI.md](QUICKSTART_WEBUI.md) - 快速开始
- [WEBUI_GUIDE.md](WEBUI_GUIDE.md) - 完整指南
- [MODEL_PATHS_CONFIG.md](MODEL_PATHS_CONFIG.md) - 模型配置

### 社区
- GitHub Issues
- 项目讨论区

### 报告问题时请提供
1. 完整错误信息
2. Python版本
3. 已安装的包列表
4. 系统信息 (Windows版本, GPU型号)
5. 重现步骤

---

**更新日期:** 2025-11-13
**作者:** eddy
