#!/usr/bin/env bash
set -euo pipefail

echo "============================================"
echo "  Body & Hand Tracker - Build Script"
echo "  一键打包为独立 EXE"
echo "============================================"
echo

# 安装依赖
echo "[1/3] 安装依赖..."
pip install -r requirements.txt

# 清理旧构建
echo "[2/3] 清理旧构建..."
rm -rf dist build

# 打包
echo "[3/3] 正在打包 EXE..."
pyinstaller body_hand_tracker.spec --clean --noconfirm

echo
echo "============================================"
echo "  打包完成！"
echo "  输出: dist/BodyHandTracker.exe"
echo "============================================"
