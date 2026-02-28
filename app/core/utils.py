from dataclasses import dataclass
from pathlib import Path
import numpy as np
import cv2
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import Qt, QSize

def bgr_to_qimage(bgr):
    if bgr is None:
        return QImage()
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    bytes_per_line = ch * w
    return QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()

def bgr_to_qpixmap(bgr, target_size: QSize | None = None):
    qimg = bgr_to_qimage(bgr)
    if qimg.isNull():
        return QPixmap()
    pix = QPixmap.fromImage(qimg)
    if target_size is not None and not target_size.isEmpty():
        pix = pix.scaled(target_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
    return pix

def list_image_files(folder: str):
    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
    p = Path(folder)
    if not p.exists():
        return []
    files = [str(f) for f in p.iterdir() if f.suffix.lower() in exts]
    files.sort()
    return files

@dataclass
class BatchItem:
    path: str
    source: np.ndarray
    result: np.ndarray
    details: list
    fps: float = 0.0  # 【新增】存储这张图片的检测耗时帧率
