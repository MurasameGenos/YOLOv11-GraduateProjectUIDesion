import time
import threading
import cv2
from PySide6.QtCore import QThread, Signal
from .utils import list_image_files
from .yolo_engine import YOLOEngine
from .stream_resolver import resolve_stream_url


class BaseWorker(QThread):
    log = Signal(str)
    finished = Signal()

    def __init__(self, parent=None, conf=0.25, iou=0.45):
        super().__init__(parent)
        self.conf = conf
        self.iou = iou
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()

    def update_params(self, conf, iou):
        """让 UI 线程可以远程修改运行中的参数"""
        self.conf = conf
        self.iou = iou

    def stop(self):
        self._stop_event.set()

    def pause(self):
        self._pause_event.set()

    def resume(self):
        self._pause_event.clear()

    def is_paused(self):
        return self._pause_event.is_set()

    def _wait_if_paused(self):
        while self._pause_event.is_set() and not self._stop_event.is_set():
            time.sleep(0.05)

    def stopped(self):
        return self._stop_event.is_set()


class ImageWorker(BaseWorker):
    frame_ready = Signal(object, object, list, float)

    # 【新增】save_dir 参数
    def __init__(self, engine: YOLOEngine, image_path: str, conf=0.25, iou=0.45, save_visual=False, save_dir="output", **kwargs):
        super().__init__()
        self.engine = engine
        self.image_path = image_path
        self.conf = conf
        self.iou = iou
        self.save_visual = save_visual
        self.save_dir = save_dir

    def run(self):
        import os
        if not os.path.exists(self.image_path):
            self.log.emit(f"图片不存在: {self.image_path}")
            self.finished.emit()
            return
        img = cv2.imread(self.image_path)
        if img is None:
            self.log.emit("读取图片失败")
            self.finished.emit()
            return
        try:
            t0 = time.perf_counter()
            result, details = self.engine.predict(img, conf=self.conf, iou=self.iou)
            t1 = time.perf_counter()
            fps = 1.0 / max(t1 - t0, 1e-6)

            # 【修改】使用动态路径保存
            if self.save_visual:
                os.makedirs(self.save_dir, exist_ok=True)
                out_name = f"res_{os.path.basename(self.image_path)}"
                cv2.imwrite(os.path.join(self.save_dir, out_name), result)

            self.frame_ready.emit(img, result, details, fps)
        except Exception as e:
            self.log.emit(f"检测失败: {e}")
        self.finished.emit()


class VideoWorker(BaseWorker):
    frame_ready = Signal(object, object, list, float)

    # 【新增】save_dir 参数
    def __init__(self, engine: YOLOEngine, source, conf=0.25, iou=0.45, save_visual=False, save_dir="output", use_tracking=False):
        super().__init__(conf=conf, iou=iou)
        from collections import defaultdict
        self.engine = engine
        self.source = source
        self.save_visual = save_visual
        self.save_dir = save_dir
        self.use_tracking = use_tracking
        # 【新增】字典，用于存储每个 ID 历史走过的中心点坐标
        self.track_history = defaultdict(lambda: [])

    def run(self):
        import os
        import math

        is_camera = isinstance(self.source, int)

        if not is_camera:
            headers = "Referer: https://www.bilibili.com/\r\nUser-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36\r\n"
            os.environ[
                "OPENCV_FFMPEG_CAPTURE_OPTIONS"] = f"protocol_whitelist;file,http,https,tcp,tls,crypto|headers;{headers}"

        source = resolve_stream_url(self.source)

        if source == "":
            self.log.emit(f"流媒体解析失败，无法获取直链: {self.source}")
            self.finished.emit()
            return

        if is_camera:
            cap = cv2.VideoCapture(source)
        else:
            cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)

        if not cap.isOpened():
            self.log.emit(f"无法打开视频源或被拒绝访问: {self.source}")
            self.finished.emit()
            return

        video_writer = None
        if self.save_visual:
            os.makedirs(self.save_dir, exist_ok=True)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0 or math.isnan(fps):
                fps = 25.0

            if is_camera:
                out_name = f"res_camera_{self.source}_{int(time.time())}.mp4"
            else:
                import urllib.parse
                parsed = urllib.parse.urlparse(str(self.source))
                if parsed.scheme in ('http', 'https', 'rtsp'):
                    out_name = f"res_stream_{int(time.time())}.mp4"
                else:
                    base_name = os.path.basename(str(self.source))
                    name, _ = os.path.splitext(base_name)
                    out_name = f"res_{name}.mp4"

            # 【修改】使用动态路径
            out_path = os.path.join(self.save_dir, out_name)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

        fps_ema = None
        while not self.stopped():
            self._wait_if_paused()
            if self.stopped():
                break

            ret, frame = cap.read()
            if not ret:
                break

            try:
                import numpy as np
                t0 = time.perf_counter()

                # 【新增核心】追踪分支与轨迹绘制
                if self.use_tracking:
                    result, details = self.engine.track(frame, conf=self.conf, iou=self.iou)
                    for det in details:
                        t_id = det.get("track_id")
                        if t_id is not None:
                            x1, y1, x2, y2 = det["xyxy"]
                            # 计算当前目标的底边中心点
                            center = (int((x1 + x2) / 2), int(y2))
                            self.track_history[t_id].append(center)
                            # 控制尾巴长度不超过 30 帧
                            if len(self.track_history[t_id]) > 30:
                                self.track_history[t_id].pop(0)

                            # 为不同 ID 生成随机但固定的专属颜色
                            color = ((t_id * 37) % 255, (t_id * 73) % 255, (t_id * 139) % 255)
                            # 绘制轨迹多边形连线
                            points = np.array(self.track_history[t_id], dtype=np.int32).reshape((-1, 1, 2))
                            cv2.polylines(result, [points], isClosed=False, color=color, thickness=3)
                else:
                    result, details = self.engine.predict(frame, conf=self.conf, iou=self.iou)

                t1 = time.perf_counter()

                inst_fps = 1.0 / max(t1 - t0, 1e-6)
                fps_ema = inst_fps if fps_ema is None else (0.9 * fps_ema + 0.1 * inst_fps)

                if video_writer is not None:
                    video_writer.write(result)

                self.frame_ready.emit(frame, result, details, fps_ema)
            except Exception as e:
                self.log.emit(f"检测失败: {e}")
                break

            time.sleep(0.001)

        if video_writer is not None:
            video_writer.release()

        cap.release()
        self.finished.emit()


class BatchWorker(BaseWorker):
    item_ready = Signal(object, object, list, str, int, int, float)

    # 【新增】save_dir 参数
    def __init__(self, engine: YOLOEngine, folder: str, conf=0.25, iou=0.45, save_visual=False, save_dir="output", **kwargs):
        super().__init__()
        self.engine = engine
        self.folder = folder
        self.conf = conf
        self.iou = iou
        self.save_visual = save_visual
        self.save_dir = save_dir

    def run(self):
        import os
        paths = list_image_files(self.folder)
        total = len(paths)
        if total == 0:
            self.log.emit("文件夹中没有图片")
            self.finished.emit()
            return

        # 【修改】使用动态路径
        if self.save_visual:
            os.makedirs(self.save_dir, exist_ok=True)

        for idx, path in enumerate(paths, start=1):
            if self.stopped():
                break
            self._wait_if_paused()
            if self.stopped():
                break

            img = cv2.imread(path)
            if img is None:
                self.log.emit(f"读取失败: {path}")
                continue

            try:
                t0 = time.perf_counter()
                result, details = self.engine.predict(img, conf=self.conf, iou=self.iou)
                t1 = time.perf_counter()
                fps = 1.0 / max(t1 - t0, 1e-6)

                if self.save_visual:
                    out_name = f"res_{os.path.basename(path)}"
                    cv2.imwrite(os.path.join(self.save_dir, out_name), result)

                self.item_ready.emit(img, result, details, path, idx, total, fps)
            except Exception as e:
                self.log.emit(f"检测失败: {path}, {e}")

        self.finished.emit()
