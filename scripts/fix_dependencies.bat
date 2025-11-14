@echo off
echo ========================================
echo Fix Dependencies for Genesis WebUI
echo ========================================
echo.
echo This script will fix the bitsandbytes/triton conflict
echo.
echo ========================================
echo.

echo Step 1: Uninstalling bitsandbytes...
C:\Users\Administrator\Desktop\fork\python313\python.exe -m pip uninstall bitsandbytes -y

echo.
echo Step 2: Upgrading diffusers...
C:\Users\Administrator\Desktop\fork\python313\python.exe -m pip install --upgrade diffusers transformers accelerate

echo.
echo Step 3: Installing core dependencies...
C:\Users\Administrator\Desktop\fork\python313\python.exe -m pip install gradio torch torchvision

echo.
echo ========================================
echo Dependencies fixed!
echo ========================================
echo.
echo You can now run: start_webui.bat
echo.

pause
