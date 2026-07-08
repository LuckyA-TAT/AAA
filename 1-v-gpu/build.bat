@echo off
chcp 65001 >nul 2>&1
echo ============================================
echo   Body ^& Hand Tracker - Build Script
echo   一键打包为独立 EXE
echo ============================================
echo.

:: 检查 Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] 未找到 Python，请先安装 Python 3.9-3.12
    pause
    exit /b 1
)

:: 安装依赖
echo [1/3] 安装依赖...
pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] 依赖安装失败
    pause
    exit /b 1
)

:: 清理旧构建
echo [2/3] 清理旧构建...
if exist dist rmdir /s /q dist
if exist build rmdir /s /q build

:: 打包
echo [3/3] 正在打包 EXE（首次可能较慢）...
pyinstaller body_hand_tracker.spec --clean --noconfirm
if errorlevel 1 (
    echo [ERROR] 打包失败
    pause
    exit /b 1
)

echo.
echo ============================================
echo   打包完成！
echo   输出: dist\BodyHandTracker.exe
echo ============================================
echo.
echo 使用方式:
echo   BodyHandTracker.exe                       (默认摄像头)
echo   BodyHandTracker.exe --camera 1            (外接摄像头)
echo   BodyHandTracker.exe -c 1 -W 1920 -H 1080  (指定分辨率)
echo   BodyHandTracker.exe --record output.mp4   (录制)
echo   BodyHandTracker.exe --help                (查看帮助)
echo.
pause
