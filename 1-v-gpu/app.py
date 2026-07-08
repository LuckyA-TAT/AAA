"""
Body & Hand Tracker
===================
实时人体姿态 + 手部关键点检测程序
- 实时视频输出（镜像画面）
- 黑色细线标记人体骨架 + 手部形态
- 支持外接摄像头 / 本机摄像头
- GPU/CPU 自动回退

兼容 MediaPipe 0.10.14+ 新版 Task API
"""

import sys
import os
import time
import argparse
import urllib.request
import cv2
import numpy as np

# ──────────────────────────────────────────────
# 模型文件 URL 与本地路径
# ──────────────────────────────────────────────
MODELS = {
    "pose": {
        "file": "pose_landmarker.task",
        "url": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
    },
    "hand": {
        "file": "hand_landmarker.task",
        "url": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
    },
}

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")


def ensure_models():
    """下载缺失的模型文件"""
    os.makedirs(MODEL_DIR, exist_ok=True)
    for name, info in MODELS.items():
        path = os.path.join(MODEL_DIR, info["file"])
        if not os.path.exists(path):
            print(f"[INFO] 下载 {name} 模型: {info['url']}")
            try:
                urllib.request.urlretrieve(info["url"], path)
                print(f"[INFO] 已保存: {path}")
            except Exception as e:
                print(f"[ERROR] 下载失败 {name}: {e}")
                print(f"  请手动下载放到: {path}")
                sys.exit(1)
    return {
        "pose": os.path.join(MODEL_DIR, MODELS["pose"]["file"]),
        "hand": os.path.join(MODEL_DIR, MODELS["hand"]["file"]),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Body & Hand Tracker")
    parser.add_argument("--camera", "-c", type=int, default=0, help="摄像头索引 (0=默认, 1=外接)")
    parser.add_argument("--width", "-W", type=int, default=1280, help="采集宽度")
    parser.add_argument("--height", "-H", type=int, default=720, help="采集高度")
    parser.add_argument("--delegate", type=str, default="gpu", choices=["gpu", "cpu"], help="GPU/CPU")
    parser.add_argument("--model-complexity", type=int, default=1, choices=[0, 1, 2])
    parser.add_argument("--min-detection-confidence", type=float, default=0.6)
    parser.add_argument("--record", type=str, default=None, help="录制输出路径")
    parser.add_argument("--fullscreen", action="store_true", default=False)
    parser.add_argument("--line-thickness", type=int, default=1, help="骨架线条粗细 (默认 1)")
    return parser.parse_args()


# ──────────────────────────────────────────────
# 骨架连接定义
# ──────────────────────────────────────────────
POSE_CONNECTIONS = [
    # 躯干
    (11, 12), (11, 23), (12, 24), (23, 24),
    # 左臂
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
    # 右臂
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),
    # 左腿
    (23, 25), (25, 27), (27, 29), (27, 31),
    # 右腿
    (24, 26), (26, 28), (28, 30), (28, 32),
    # 面部
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10), (0, 9), (0, 10),
]

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]

POSE_NAMES = {
    0: "NOSE", 11: "L_SHOULDER", 12: "R_SHOULDER",
    13: "L_ELBOW", 14: "R_ELBOW", 15: "L_WRIST", 16: "R_WRIST",
    23: "L_HIP", 24: "R_HIP", 25: "L_KNEE", 26: "R_KNEE",
    27: "L_ANKLE", 28: "R_ANKLE",
}

HAND_TIP_NAMES = {4: "THUMB", 8: "INDEX", 12: "MIDDLE", 16: "RING", 20: "PINKY"}


class BodyHandTracker:
    """核心追踪器：实时视频 + 黑色细线骨架"""

    def __init__(self, args, model_paths):
        self.args = args
        self.fps = 0.0
        self.frame_count = 0
        self.prev_time = time.time()

        # 统一使用黑色细线
        self.LINE_COLOR = (0, 0, 0)         # 黑色 (BGR)
        self.LINE_THICKNESS = args.line_thickness
        self.POINT_COLOR = (0, 0, 0)        # 关键点也用黑色
        self.POINT_RADIUS = 2               # 关键点小圆点
        self.LABEL_COLOR = (0, 0, 0)        # 标签黑色
        self.LABEL_BG = (255, 255, 255)     # 标签白色背景
        self.TEXT_COLOR = (0, 0, 0)         # HUD 文字黑色
        self.HUD_BG = (240, 240, 240)       # HUD 灰白背景

        # 导入 MediaPipe Task API
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

        def _build_detectors(use_gpu):
            if use_gpu:
                try:
                    delegate = mp_python.BaseOptions.Delegate.GPU
                except Exception:
                    delegate = mp_python.BaseOptions.Delegate.CPU
            else:
                delegate = mp_python.BaseOptions.Delegate.CPU

            pose_opts = mp_vision.PoseLandmarkerOptions(
                base_options=mp_python.BaseOptions(
                    model_asset_path=model_paths["pose"], delegate=delegate,
                ),
                running_mode=mp_vision.RunningMode.VIDEO,
                min_pose_detection_confidence=args.min_detection_confidence,
                min_pose_presence_confidence=args.min_detection_confidence,
                min_tracking_confidence=args.min_detection_confidence,
                num_poses=1,
            )
            hand_opts = mp_vision.HandLandmarkerOptions(
                base_options=mp_python.BaseOptions(
                    model_asset_path=model_paths["hand"], delegate=delegate,
                ),
                running_mode=mp_vision.RunningMode.VIDEO,
                min_hand_detection_confidence=args.min_detection_confidence,
                min_hand_presence_confidence=args.min_detection_confidence,
                min_tracking_confidence=args.min_detection_confidence,
                num_hands=2,
            )
            return (
                mp_vision.PoseLandmarker.create_from_options(pose_opts),
                mp_vision.HandLandmarker.create_from_options(hand_opts),
                "GPU" if use_gpu else "CPU",
            )

        try:
            if args.delegate == "gpu":
                print("[INFO] 尝试 GPU 加速...")
                self.pose_detector, self.hand_detector, backend = _build_detectors(True)
            else:
                self.pose_detector, self.hand_detector, backend = _build_detectors(False)
        except (NotImplementedError, RuntimeError, Exception) as e:
            err_msg = str(e)
            if "GPU" in err_msg or "gpu" in err_msg.lower():
                print(f"[WARN] GPU 初始化失败，自动回退 CPU")
                self.pose_detector, self.hand_detector, backend = _build_detectors(False)
            else:
                raise

        print(f"[INFO] 后端: {backend} | 线条: 黑色 {self.LINE_THICKNESS}px")
        print("[INFO] 模型加载完成")

    def open_camera(self):
        """打开摄像头"""
        cap = cv2.VideoCapture(self.args.camera, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(self.args.camera)
            if not cap.isOpened():
                print(f"[ERROR] 无法打开摄像头 index={self.args.camera}")
                print("  → 确认摄像头已连接，尝试 --camera 1")
                sys.exit(1)

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.args.height)
        cap.set(cv2.CAP_PROP_FPS, 30)

        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[INFO] 摄像头 {self.args.camera} 已打开: {actual_w}x{actual_h}")
        return cap

    def _update_fps(self):
        self.frame_count += 1
        now = time.time()
        elapsed = now - self.prev_time
        if elapsed >= 1.0:
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.prev_time = now

    def _draw_label(self, frame, text, x, y, color=(0, 0, 0)):
        """绘制带白色背景的标签"""
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
        cv2.rectangle(frame, (x - 2, y - th - 4), (x + tw + 2, y + 2), self.LABEL_BG, -1)
        cv2.putText(frame, text, (x, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)

    def _draw_hud(self, frame, pose_count, hand_count):
        """绘制状态 HUD"""
        h, w = frame.shape[:2]
        model_name = {0: "Lite", 1: "Full", 2: "Heavy"}.get(self.args.model_complexity, "Full")
        backend = "GPU" if self.args.delegate == "gpu" else "CPU"
        lines = [
            f"FPS: {self.fps:.1f}",
            f"Camera: {self.args.camera}  |  {w}x{h}",
            f"Model: {model_name}  |  Backend: {backend}",
            f"Pose: {'DETECTED' if pose_count > 0 else '--'}",
            f"Hands: {hand_count}" if hand_count > 0 else "Hands: --",
        ]

        # 半透明白色背景
        overlay = frame.copy()
        cv2.rectangle(overlay, (8, 6), (260, 24 + len(lines) * 24), (255, 255, 255), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        for i, line in enumerate(lines):
            cv2.putText(
                frame, line, (14, 22 + i * 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.TEXT_COLOR, 1, cv2.LINE_AA,
            )

        # 底部操作提示
        hint = "Q/ESC: Quit  |  S: Screenshot  |  F: Fullscreen  |  R: Record"
        (tw, _), _ = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.rectangle(frame, (0, h - 28), (w, h), (255, 255, 255), -1)
        cv2.putText(
            frame, hint, ((w - tw) // 2, h - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.TEXT_COLOR, 1, cv2.LINE_AA,
        )

    def draw_pose(self, frame, pose_result):
        """绘制人体骨架 — 黑色细线"""
        if not pose_result or not pose_result.pose_landmarks:
            return

        h, w = frame.shape[:2]
        landmarks = pose_result.pose_landmarks[0]
        color = self.LINE_COLOR
        thick = self.LINE_THICKNESS

        # 骨架连线
        for start_idx, end_idx in POSE_CONNECTIONS:
            if start_idx < len(landmarks) and end_idx < len(landmarks):
                lm_s = landmarks[start_idx]
                lm_e = landmarks[end_idx]
                x1, y1 = int(lm_s.x * w), int(lm_s.y * h)
                x2, y2 = int(lm_e.x * w), int(lm_e.y * h)
                cv2.line(frame, (x1, y1), (x2, y2), color, thick, cv2.LINE_AA)

        # 关键点
        for i, lm in enumerate(landmarks):
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (cx, cy), self.POINT_RADIUS, self.POINT_COLOR, -1, cv2.LINE_AA)

        # 标注关键关节
        for idx, name in POSE_NAMES.items():
            if idx < len(landmarks):
                lm = landmarks[idx]
                cx, cy = int(lm.x * w), int(lm.y * h)
                self._draw_label(frame, name, cx + 6, cy - 4)

    def draw_hands(self, frame, hand_result):
        """绘制手部骨架 — 黑色细线"""
        if not hand_result or not hand_result.hand_landmarks:
            return

        h, w = frame.shape[:2]
        color = self.LINE_COLOR
        thick = self.LINE_THICKNESS

        for i, (hand_lm, handedness) in enumerate(
            zip(hand_result.hand_landmarks, hand_result.handedness)
        ):
            label = handedness[0].category_name
            score = handedness[0].score
            # 镜像翻转
            display_label = "R Hand" if label == "Left" else "L Hand"

            # 手部连线
            for start_idx, end_idx in HAND_CONNECTIONS:
                if start_idx < len(hand_lm) and end_idx < len(hand_lm):
                    lm_s = hand_lm[start_idx]
                    lm_e = hand_lm[end_idx]
                    x1, y1 = int(lm_s.x * w), int(lm_s.y * h)
                    x2, y2 = int(lm_e.x * w), int(lm_e.y * h)
                    cv2.line(frame, (x1, y1), (x2, y2), color, thick, cv2.LINE_AA)

            # 关键点
            for j, lm in enumerate(hand_lm):
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), self.POINT_RADIUS, self.POINT_COLOR, -1, cv2.LINE_AA)

            # 手部名称标签
            wrist = hand_lm[0]
            cx, cy = int(wrist.x * w), int(wrist.y * h)
            self._draw_label(frame, f"{display_label} ({score:.0%})", cx + 8, cy - 6)

            # 指尖标注
            for idx, name in HAND_TIP_NAMES.items():
                if idx < len(hand_lm):
                    lm = hand_lm[idx]
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    self._draw_label(frame, name, cx + 4, cy - 4)

    def process_frame(self, frame, timestamp_ms):
        """处理单帧：检测 + 绘制"""
        import mediapipe as mp

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        pose_result = self.pose_detector.detect_for_video(mp_image, timestamp_ms)
        hand_result = self.hand_detector.detect_for_video(mp_image, timestamp_ms)

        pose_count = len(pose_result.pose_landmarks) if pose_result.pose_landmarks else 0
        hand_count = len(hand_result.hand_landmarks) if hand_result.hand_landmarks else 0

        self.draw_pose(frame, pose_result)
        self.draw_hands(frame, hand_result)
        self._update_fps()
        self._draw_hud(frame, pose_count, hand_count)

        return frame

    def run(self):
        """主循环：实时视频输出"""
        cap = self.open_camera()

        recorder = None
        if self.args.record:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            recorder = cv2.VideoWriter(self.args.record, fourcc, 30, (w, h))
            print(f"[INFO] 录制中 → {self.args.record}")

        window_name = "Body & Hand Tracker"
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        if self.args.fullscreen:
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        print("[INFO] 按 Q/ESC 退出 | S 截图 | F 全屏 | R 录制")
        print("[INFO] 实时视频输出中...")
        sys.stdout.flush()

        timestamp_ms = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("[WARN] 读取帧失败，重试...")
                    sys.stdout.flush()
                    time.sleep(0.1)
                    continue

                # 镜像翻转（自然交互）
                frame = cv2.flip(frame, 1)
                timestamp_ms = int(time.time() * 1000)

                # 检测 + 绘制
                frame = self.process_frame(frame, timestamp_ms)

                # 录制
                if recorder:
                    recorder.write(frame)

                # ★ 实时视频输出 ★
                cv2.imshow(window_name, frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    print("[INFO] 退出...")
                    break
                elif key == ord("s"):
                    ts = time.strftime("%Y%m%d_%H%M%S")
                    fname = f"screenshot_{ts}.png"
                    cv2.imwrite(fname, frame)
                    print(f"[INFO] 截图: {fname}")
                elif key == ord("f"):
                    prop = cv2.getWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN)
                    if prop == cv2.WINDOW_FULLSCREEN:
                        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                    else:
                        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                elif key == ord("r"):
                    if recorder is None:
                        ts = time.strftime("%Y%m%d_%H%M%S")
                        fname = f"record_{ts}.mp4"
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        recorder = cv2.VideoWriter(fname, fourcc, 30, (w, h))
                        print(f"[INFO] 开始录制: {fname}")
                    else:
                        recorder.release()
                        recorder = None
                        print("[INFO] 录制已停止")

                sys.stdout.flush()

        finally:
            cap.release()
            if recorder:
                recorder.release()
            cv2.destroyAllWindows()
            self.pose_detector.close()
            self.hand_detector.close()
            print("[INFO] 已释放所有资源")


def main():
    args = parse_args()
    print("=" * 50)
    print("  Body & Hand Tracker")
    print("  实时视频输出 | 黑色细线骨架")
    print("=" * 50)
    print(f"  摄像头   : {args.camera}")
    print(f"  分辨率   : {args.width}x{args.height}")
    print(f"  线条     : 黑色 {args.line_thickness}px")
    print(f"  GPU/CPU  : {args.delegate}")
    print("=" * 50)
    sys.stdout.flush()

    model_paths = ensure_models()
    tracker = BodyHandTracker(args, model_paths)
    tracker.run()


if __name__ == "__main__":
    main()
