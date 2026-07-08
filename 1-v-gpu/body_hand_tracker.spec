# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec file - Body & Hand Tracker
# 构建命令: pyinstaller body_hand_tracker.spec

import os
import sys
from PyInstaller.utils.hooks import collect_data_files

# 收集 MediaPipe 数据文件（模型等）
mediapipe_datas = collect_data_files('mediapipe')

a = Analysis(
    ['app.py'],
    pathex=[],
    binaries=[],
    datas=mediapipe_datas + [
        ('models/*.task', 'models'),
    ],
    hiddenimports=[
        'mediapipe',
        'cv2',
        'numpy',
        'mediapipe.tasks',
        'mediapipe.tasks.python',
        'mediapipe.tasks.python.vision',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'tkinter',  # 排除不需要的 GUI 库
        'matplotlib',
        'scipy',
        'pandas',
    ],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='BodyHandTracker',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,          # 保留控制台（方便看日志和传参）
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,             # 可替换为 icon='app.ico'
)
