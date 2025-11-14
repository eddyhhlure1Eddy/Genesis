@echo off
chcp 65001 >nul
title Genesis WebUI - Modular Interface

echo ========================================
echo Genesis WebUI Launcher
echo ========================================
echo.

C:\Users\Administrator\Desktop\fork\python313\python.exe apps\genesis_webui_integrated.py

if errorlevel 1 (
    echo.
    echo [ERROR] Failed to start WebUI
    echo Please check the error message above
    pause
)
