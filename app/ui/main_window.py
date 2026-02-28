import os
import sys
import logging
import pandas as pd
from datetime import datetime
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QGridLayout, QGroupBox,
    QLineEdit, QPushButton, QLabel, QComboBox, QFileDialog, QTextEdit,
    QStackedWidget, QTableWidget, QTableWidgetItem, QSpinBox,
    QSizePolicy, QHeaderView, QSlider, QCheckBox, QApplication
)
from PySide6.QtCore import Qt, QSize, qInstallMessageHandler, QtMsgType
from app.core.yolo_engine import YOLOEngine
from app.core.workers import ImageWorker, VideoWorker, BatchWorker
from app.core.utils import bgr_to_qpixmap, BatchItem
from app.core.logger import LogEmitter, QtHandler
from app.core.stream_resolver import cleanup_cache


# 【终极修复方案】定义全局 Qt 消息拦截器
def _qt_message_handler(mode, context, message):
    """拦截并过滤掉无害的 pointSize 警告，防止 CMD 刷屏"""
    if "Point size <= 0" in message:
        return  # 直接丢弃这条警告
    # 其他真正的错误依然输出到控制台
    print(message, file=sys.stderr)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # 1. 挂载 Qt 消息拦截器
        qInstallMessageHandler(_qt_message_handler)

        # 2. 尝试修复 QApplication 全局字体
        app = QApplication.instance()
        if app:
            sys_font = app.font()
            if sys_font.pointSize() <= 0:
                sys_font.setPointSize(10)
                app.setFont(sys_font)

        self.setWindowTitle("YOLO检测前端v2 by Sonneto in 2026.2.27")
        self.resize(1280, 720)

        self.engine = YOLOEngine()
        self.worker = None

        # 批量图片专用的数据
        self.batch_items = []
        self.batch_total = 0
        self.current_batch_index = -1

        # 视频/实时流专用的逐帧纯数据记录 (防内存溢出设计)
        self.record_data = []
        self.frame_counter = 0

        self._build_ui()
        self._setup_logging()

        # 软件启动时的欢迎语
        self._log("本软件用于个人毕业设计展示")
        self._log("Ciallo～(∠・ω< )⌒★")

        self._on_source_changed(0)
        self._set_buttons_running(False)

    def closeEvent(self, event):
        """当软件关闭时，执行清理工作"""
        self._log("正在关闭软件，清理临时资源...")
        self._stop_worker()
        cleanup_cache()
        event.accept()

    # ---------- UI ----------
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        # 左侧
        left = QWidget()
        left.setFixedWidth(360)
        left_layout = QVBoxLayout(left)

        # ---------- 模型配置 ----------
        gb_model = QGroupBox("模型配置")
        grid = QGridLayout(gb_model)

        self.model_path = QLineEdit()
        self.model_path.setPlaceholderText("选择 .pt 文件")
        self.btn_browse_model = QPushButton("选择")
        self.btn_load_model = QPushButton("加载")
        self.model_status = QLabel("未加载")

        self.device_combo = QComboBox()
        self.device_combo.addItems(["Auto (自动探测)", "CPU (强制)", "GPU (CUDA)"])

        grid.addWidget(QLabel("模型文件:"), 0, 0)
        grid.addWidget(self.model_path, 0, 1)
        grid.addWidget(self.btn_browse_model, 0, 2)
        grid.addWidget(QLabel("计算设备:"), 1, 0)
        grid.addWidget(self.device_combo, 1, 1)
        grid.addWidget(self.btn_load_model, 2, 1)
        grid.addWidget(self.model_status, 2, 2)

        self.btn_browse_model.clicked.connect(self._browse_model)
        self.btn_load_model.clicked.connect(self._load_model)

        # ==================== 参数配置 ====================
        gb_params = QGroupBox("参数配置")
        v_params = QVBoxLayout(gb_params)

        # 1. 置信度 (Conf)
        h_conf = QHBoxLayout()
        h_conf.addWidget(QLabel("置信度 (Conf):"))
        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setRange(1, 100)
        self.conf_slider.setValue(25)  # 默认 0.25
        self.conf_label = QLabel("0.25")
        self.conf_label.setFixedWidth(35)
        h_conf.addWidget(self.conf_slider)
        h_conf.addWidget(self.conf_label)

        # 2. 交并比 (IoU)
        h_iou = QHBoxLayout()
        h_iou.addWidget(QLabel("交并比 (IoU):"))
        self.iou_slider = QSlider(Qt.Horizontal)
        self.iou_slider.setRange(1, 100)
        self.iou_slider.setValue(45)  # 默认 0.45
        self.iou_label = QLabel("0.45")
        self.iou_label.setFixedWidth(35)
        h_iou.addWidget(self.iou_slider)
        h_iou.addWidget(self.iou_label)

        # 3. 视觉结果保存开关
        self.cb_save_visual = QCheckBox("保存视觉结果 (图片/视频)")
        self.cb_save_visual.setStyleSheet("color: #4CC2FF; font-weight: bold; margin-top: 5px;")

        # 4. 目标追踪开关
        self.cb_tracking = QCheckBox("启用目标追踪与轨迹线 (Tracking)")
        self.cb_tracking.setStyleSheet("color: #FFB900; font-weight: bold; margin-top: 5px;")
        self.cb_tracking.setToolTip("开启后，将赋予每个目标固定编号，并绘制移动轨迹。仅对视频流生效。")

        v_params.addLayout(h_conf)
        v_params.addLayout(h_iou)
        v_params.addWidget(self.cb_save_visual)
        v_params.addWidget(self.cb_tracking)

        # 绑定滑动信号
        self.conf_slider.valueChanged.connect(self._on_params_changed)
        self.iou_slider.valueChanged.connect(self._on_params_changed)
        # ========================================================

        # ---------- 检测源配置 ----------
        gb_source = QGroupBox("检测源配置")
        v_source = QVBoxLayout(gb_source)
        self.source_combo = QComboBox()
        self.source_combo.addItems([
            "单张图片", "单个视频", "电脑摄像头", "文件夹批量图片", "网络视频流"
        ])
        v_source.addWidget(self.source_combo)

        self.source_stack = QStackedWidget()

        # 1. 图片
        self.image_path = QLineEdit()
        btn_img = QPushButton("浏览")
        btn_img.clicked.connect(lambda: self._browse_file(
            self.image_path, "图片文件 (*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp)"
        ))
        page_img = QWidget()
        h_img = QHBoxLayout(page_img)
        h_img.addWidget(self.image_path)
        h_img.addWidget(btn_img)

        # 2. 视频
        self.video_path = QLineEdit()
        btn_vid = QPushButton("浏览")
        btn_vid.clicked.connect(lambda: self._browse_file(
            self.video_path, "视频文件 (*.mp4 *.avi *.mov *.mkv)"
        ))
        page_vid = QWidget()
        h_vid = QHBoxLayout(page_vid)
        h_vid.addWidget(self.video_path)
        h_vid.addWidget(btn_vid)

        # 3. 摄像头
        page_cam = QWidget()
        h_cam = QHBoxLayout(page_cam)
        self.camera_index = QSpinBox()
        self.camera_index.setRange(0, 10)
        h_cam.addWidget(QLabel("摄像头编号:"))
        h_cam.addWidget(self.camera_index)
        h_cam.addStretch()

        # 4. 文件夹
        self.folder_path = QLineEdit()
        btn_folder = QPushButton("浏览")
        btn_folder.clicked.connect(self._browse_folder)
        page_folder = QWidget()
        h_folder = QHBoxLayout(page_folder)
        h_folder.addWidget(self.folder_path)
        h_folder.addWidget(btn_folder)

        # 5. 网络流
        self.stream_url = QLineEdit()
        self.stream_url.setPlaceholderText("rtsp:// 或 http://")
        page_stream = QWidget()
        h_stream = QHBoxLayout(page_stream)
        h_stream.addWidget(self.stream_url)

        self.source_stack.addWidget(page_img)
        self.source_stack.addWidget(page_vid)
        self.source_stack.addWidget(page_cam)
        self.source_stack.addWidget(page_folder)
        self.source_stack.addWidget(page_stream)

        v_source.addWidget(self.source_stack)

        # ---------- 检测控制 ----------
        gb_ctrl = QGroupBox("检测控制")
        h_ctrl = QHBoxLayout(gb_ctrl)
        self.btn_start = QPushButton("开始")
        self.btn_pause = QPushButton("暂停")
        self.btn_stop = QPushButton("结束")
        self.btn_start.setObjectName("btn_start")
        h_ctrl.addWidget(self.btn_start)
        h_ctrl.addWidget(self.btn_pause)
        h_ctrl.addWidget(self.btn_stop)

        # ---------- 运行日志 ----------
        gb_log = QGroupBox("运行日志")
        v_log = QVBoxLayout(gb_log)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet("font-size: 10pt;")

        # 修复日志文档默认字体
        doc_font = self.log_text.document().defaultFont()
        if doc_font.pointSize() <= 0:
            doc_font.setPointSize(10)
            self.log_text.document().setDefaultFont(doc_font)

        v_log.addWidget(self.log_text)

        left_layout.addWidget(gb_model)
        left_layout.addWidget(gb_params)
        left_layout.addWidget(gb_source)
        left_layout.addWidget(gb_ctrl)
        left_layout.addWidget(gb_log, 1)

        # 右侧
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)

        # 右上角全局保存视觉结果开关及路径选择器
        top_h = QHBoxLayout()
        self.cb_save_visual_top = QLabel("保存目录:")
        self.cb_save_visual_top.setStyleSheet("font-weight: bold; font-size: 11pt; color: #4CC2FF;")

        self.out_dir_edit = QLineEdit()
        self.out_dir_edit.setPlaceholderText("请选择保存目录...")
        self.out_dir_edit.setText(os.path.abspath("output"))
        self.out_dir_edit.setReadOnly(True)
        self.out_dir_edit.setFixedWidth(200)

        self.btn_browse_out = QPushButton("更改目录")

        top_h.addStretch()
        top_h.addWidget(self.cb_save_visual_top)
        top_h.addWidget(self.out_dir_edit)
        top_h.addWidget(self.btn_browse_out)

        self.right_stack = QStackedWidget()
        self.page_realtime = self._make_realtime_page()
        self.page_batch = self._make_batch_page()
        self.page_monitor = self._make_monitor_page()

        self.right_stack.addWidget(self.page_realtime)
        self.right_stack.addWidget(self.page_batch)
        self.right_stack.addWidget(self.page_monitor)

        right_layout.addLayout(top_h)
        right_layout.addWidget(self.right_stack, 1)

        main_layout.addWidget(left)
        main_layout.addWidget(right_widget, 1)

        # 连接信号
        self.source_combo.currentIndexChanged.connect(self._on_source_changed)
        self.btn_start.clicked.connect(self._on_start)
        self.btn_pause.clicked.connect(self._on_pause)
        self.btn_stop.clicked.connect(self._on_stop)

        self.btn_prev.clicked.connect(self._on_prev_batch)
        self.btn_next.clicked.connect(self._on_next_batch)
        self.btn_save_csv.clicked.connect(self._on_save_batch_csv)
        self.btn_clear_batch.clicked.connect(self._on_clear_batch)
        self.btn_browse_out.clicked.connect(self._browse_out_dir)

    # ---------- 右侧 UI 重构 ----------
    def _make_realtime_page(self):
        w = QWidget()
        v = QVBoxLayout(w)

        info = QHBoxLayout()
        self.rt_fps_label = QLabel("FPS: 0.0")
        self.rt_fps_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self.rt_fps_label.setMaximumHeight(24)
        info.addWidget(self.rt_fps_label)
        info.addStretch()

        # 上半部分：图像
        h_img = QHBoxLayout()
        self.rt_src_label = QLabel()
        self.rt_res_label = QLabel()
        self._init_image_label(self.rt_src_label, "检测源")
        self._init_image_label(self.rt_res_label, "检测结果")
        h_img.addWidget(self.rt_src_label, 1)
        h_img.addWidget(self.rt_res_label, 1)

        # 下半部分：表格和统计框
        self.detail_table = QTableWidget(0, 5)
        self.detail_table.setHorizontalHeaderLabels(["序号", "类别(ID)", "置信度", "坐标(x1,y1,x2,y2)", "尺寸(宽x高)"])
        self.detail_table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.detail_table.horizontalHeader().setStretchLastSection(True)

        self.rt_stats_label = QLabel("等待检测...")
        self.rt_stats_label.setStyleSheet(
            "background:#2D2D2D; padding:10px; border-radius:5px; font-weight:bold; font-size:11pt; color:#4CC2FF;")
        self.rt_stats_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.rt_stats_label.setMinimumHeight(60)

        # 导出当前记录按钮
        bottom_h = QHBoxLayout()
        self.btn_save_rt_csv = QPushButton("💾 导出逐帧检测记录(CSV)")
        self.btn_save_rt_csv.setStyleSheet(
            "padding: 8px 20px; font-weight: bold; font-size: 10pt; background-color: #2D2D2D;")
        self.btn_save_rt_csv.clicked.connect(self._on_save_record_csv)
        bottom_h.addStretch()
        bottom_h.addWidget(self.btn_save_rt_csv)

        v.addLayout(info, 0)
        v.addLayout(h_img, 3)
        v.addWidget(self.detail_table, 2)
        v.addWidget(self.rt_stats_label, 0)
        v.addLayout(bottom_h, 0)
        return w

    def _make_batch_page(self):
        w = QWidget()
        v = QVBoxLayout(w)

        info = QHBoxLayout()
        self.batch_fps_label = QLabel("整体处理速度: 0.0 img/s")
        info.addWidget(self.batch_fps_label)
        info.addStretch()

        h_img = QHBoxLayout()
        self.batch_src_label = QLabel()
        self.batch_res_label = QLabel()
        self._init_image_label(self.batch_src_label, "检测源")
        self._init_image_label(self.batch_res_label, "检测结果")
        h_img.addWidget(self.batch_src_label, 1)
        h_img.addWidget(self.batch_res_label, 1)

        self.batch_stats_label = QLabel("等待检测...")
        self.batch_stats_label.setStyleSheet(
            "background:#2D2D2D; padding:15px; border-radius:5px; font-weight:bold; font-size:11pt; color:#4CC2FF;")
        self.batch_stats_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.batch_stats_label.setMinimumHeight(80)

        nav = QHBoxLayout()
        self.btn_prev = QPushButton("上一个")
        self.btn_next = QPushButton("下一个")
        self.batch_index_label = QLabel("0/0")
        self.btn_save_csv = QPushButton("💾 保存批量结果(CSV)")
        self.btn_save_csv.setStyleSheet("padding: 8px; font-weight: bold; font-size: 10pt; background-color: #2D2D2D;")
        self.btn_clear_batch = QPushButton("清空结果")

        nav.addWidget(self.btn_prev)
        nav.addWidget(self.btn_next)
        nav.addWidget(self.batch_index_label)
        nav.addStretch()
        nav.addWidget(self.btn_save_csv)
        nav.addWidget(self.btn_clear_batch)

        v.addLayout(info, 0)
        v.addLayout(h_img, 1)
        v.addWidget(self.batch_stats_label, 0)
        v.addLayout(nav, 0)
        return w

    def _make_monitor_page(self):
        w = QWidget()
        v = QVBoxLayout(w)

        info = QHBoxLayout()
        self.mon_fps_label = QLabel("FPS: 0.0")
        self.mon_fps_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self.mon_fps_label.setMaximumHeight(24)
        info.addWidget(self.mon_fps_label)
        info.addStretch()

        h = QHBoxLayout()
        self.mon_src_label = QLabel()
        self.mon_res_label = QLabel()
        self._init_image_label(self.mon_src_label, "实时监控")
        self._init_image_label(self.mon_res_label, "实时检测结果")
        h.addWidget(self.mon_src_label, 1)
        h.addWidget(self.mon_res_label, 1)

        # 导出监控记录按钮
        bottom_h = QHBoxLayout()
        self.btn_save_mon_csv = QPushButton("💾 导出监控检测记录(CSV)")
        self.btn_save_mon_csv.setStyleSheet(
            "padding: 8px 20px; font-weight: bold; font-size: 10pt; background-color: #2D2D2D;")
        self.btn_save_mon_csv.clicked.connect(self._on_save_record_csv)
        bottom_h.addStretch()
        bottom_h.addWidget(self.btn_save_mon_csv)

        v.addLayout(info, 0)
        v.addLayout(h, 1)
        v.addLayout(bottom_h, 0)
        return w

    def _init_image_label(self, label: QLabel, text: str):
        label.setAlignment(Qt.AlignCenter)
        label.setText(text)
        label.setStyleSheet("QLabel{background:#222;color:#aaa;border:1px solid #444;}")
        label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        label.setMinimumSize(1, 1)

    # ---------- Logging ----------
    def _setup_logging(self):
        self.log_emitter = LogEmitter()
        self.log_emitter.message.connect(self._append_log)

        handler = QtHandler(self.log_emitter)
        formatter = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", "%H:%M:%S")
        handler.setFormatter(formatter)

        root = logging.getLogger()
        root.handlers.clear()
        root.setLevel(logging.INFO)
        root.addHandler(handler)

    def _append_log(self, msg: str):
        self.log_text.append(msg)

    def _log(self, msg: str):
        logging.info(msg)

    # ---------- Browse ----------
    def _browse_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择模型文件", "", "模型文件 (*.pt)")
        if path:
            self.model_path.setText(path)

    def _browse_file(self, line_edit: QLineEdit, filters: str):
        path, _ = QFileDialog.getOpenFileName(self, "选择文件", "", filters)
        if path:
            line_edit.setText(path)

    def _browse_folder(self):
        path = QFileDialog.getExistingDirectory(self, "选择文件夹")
        if path:
            self.folder_path.setText(path)

    def _browse_out_dir(self):
        """选择保存视觉结果的输出目录"""
        path = QFileDialog.getExistingDirectory(self, "选择保存视觉结果的文件夹")
        if path:
            self.out_dir_edit.setText(path)

    def _load_model(self):
        path = self.model_path.text().strip()
        if not path or not os.path.exists(path):
            self._log("模型路径无效")
            return

        device_idx = self.device_combo.currentIndex()
        if device_idx == 1:
            target_device = "cpu"
        elif device_idx == 2:
            target_device = "cuda:0"
        else:
            target_device = None

        try:
            used_device = self.engine.load_model(path, device=target_device)
            self.model_status.setText(f"已加载 ({used_device})")

            if used_device == "cpu":
                self._log(f"模型已加载: {os.path.basename(path)}")
                self._log("当前正在使用 CPU 进行计算。")
            else:
                self._log(f"模型已加载: {os.path.basename(path)}")
                self._log(f"成功调用 GPU 硬件加速: {used_device}，运行更高效。")

        except Exception as e:
            self.model_status.setText("加载失败")
            self._log(f"加载模型失败: {e}")

    def _on_params_changed(self):
        conf = self.conf_slider.value() / 100.0
        iou = self.iou_slider.value() / 100.0
        self.conf_label.setText(f"{conf:.2f}")
        self.iou_label.setText(f"{iou:.2f}")

        if self.worker:
            self.worker.update_params(conf, iou)

    # ---------- Source change ----------
    def _on_source_changed(self, idx: int):
        self.source_stack.setCurrentIndex(idx)
        if idx == 3:
            self.right_stack.setCurrentWidget(self.page_batch)
        elif idx == 2:
            self.right_stack.setCurrentWidget(self.page_monitor)
        else:
            self.right_stack.setCurrentWidget(self.page_realtime)

    # ---------- Control ----------
    def _on_start(self):
        if not self.engine.is_loaded():
            self._log("请先加载模型")
            return

        self._stop_worker()

        self.record_data.clear()
        self.frame_counter = 0

        conf = self.conf_slider.value() / 100.0
        iou = self.iou_slider.value() / 100.0

        save_visual = self.cb_save_visual.isChecked()
        save_dir = self.out_dir_edit.text().strip()
        use_tracking = self.cb_tracking.isChecked()

        if save_visual and not save_dir:
            self._log("请选择保存视觉结果的文件夹！")
            return

        idx = self.source_combo.currentIndex()
        common_kwargs = {
            "conf": conf, "iou": iou,
            "save_visual": save_visual, "save_dir": save_dir,
            "use_tracking": use_tracking
        }

        if idx == 0:
            path = self.image_path.text().strip()
            if not path:
                self._log("请选择图片")
                return
            self.worker = ImageWorker(self.engine, path, **common_kwargs)
            self.worker.frame_ready.connect(self._on_frame)
        elif idx == 1:
            path = self.video_path.text().strip()
            if not path:
                self._log("请选择视频")
                return
            self.worker = VideoWorker(self.engine, path, **common_kwargs)
            self.worker.frame_ready.connect(self._on_frame)
        elif idx == 2:
            cam_index = self.camera_index.value()
            self.worker = VideoWorker(self.engine, cam_index, **common_kwargs)
            self.worker.frame_ready.connect(self._on_monitor_frame)
        elif idx == 3:
            folder = self.folder_path.text().strip()
            if not folder:
                self._log("请选择文件夹")
                return
            self._clear_batch_internal()
            self.worker = BatchWorker(self.engine, folder, **common_kwargs)
            self.worker.item_ready.connect(self._on_batch_item)
        elif idx == 4:
            url = self.stream_url.text().strip()
            if not url:
                self._log("请输入网络视频流链接")
                return
            self.worker = VideoWorker(self.engine, url, **common_kwargs)
            self.worker.frame_ready.connect(self._on_frame)

        self.worker.log.connect(self._log)
        self.worker.finished.connect(self._on_worker_finished)

        self._set_buttons_running(True)
        self._log("开始检测")
        self.worker.start()

    def _on_pause(self):
        if not self.worker:
            return
        if self.worker.is_paused():
            self.worker.resume()
            self.btn_pause.setText("暂停")
            self._log("继续检测")
        else:
            self.worker.pause()
            self.btn_pause.setText("继续")
            self._log("已暂停")

    def _on_stop(self):
        if self.worker:
            self._stop_worker()
            self._log("检测已停止")

    def _on_worker_finished(self):
        self.worker = None
        self._set_buttons_running(False)
        self.btn_pause.setText("暂停")
        self._log("检测完成")

    def _stop_worker(self):
        if self.worker:
            self.worker.stop()
            self.worker.wait(1000)
            self.worker = None
        self._set_buttons_running(False)
        self.btn_pause.setText("暂停")

    def _set_buttons_running(self, running: bool):
        self.btn_start.setEnabled(not running)
        self.btn_pause.setEnabled(running)
        self.btn_stop.setEnabled(running)

    # ---------- 逐帧数据记录核心 ----------
    def _record_frame_data(self, details):
        self.frame_counter += 1

        if not details:
            self.record_data.append({
                "Frame (帧)": self.frame_counter,
                "Tracking ID (追踪ID)": "",
                "Class (类别)": "None",
                "Confidence (置信度)": "",
                "x1": "", "y1": "", "x2": "", "y2": ""
            })
        else:
            for det in details:
                x1, y1, x2, y2 = det["xyxy"]
                self.record_data.append({
                    "Frame (帧)": self.frame_counter,
                    "Tracking ID (追踪ID)": det.get("track_id", ""),
                    "Class (类别)": det["class_name"],
                    "Confidence (置信度)": round(det["confidence"], 4),
                    "x1": round(x1, 1), "y1": round(y1, 1),
                    "x2": round(x2, 1), "y2": round(y2, 1)
                })

    def _on_save_record_csv(self):
        if not self.record_data:
            self._log("当前没有可保存的检测记录！")
            return

        file_path, _ = QFileDialog.getSaveFileName(self, "保存逐帧记录CSV", "video_record.csv", "CSV Files (*.csv)")
        if not file_path:
            return

        df = pd.DataFrame(self.record_data)
        df.to_csv(file_path, index=False, encoding='utf-8-sig')
        self._log(f"成功导出逐帧检测记录: {file_path}")

    # ---------- UI Update & 数据统计 ----------
    def _set_label_image(self, label: QLabel, bgr):
        size = label.contentsRect().size()
        if size.width() < 2 or size.height() < 2:
            size = QSize(640, 480)
        pix = bgr_to_qpixmap(bgr, size)
        label.setPixmap(pix)

    def _update_detail_table_and_stats(self, details, fps):
        self.detail_table.setRowCount(0)
        if not details:
            self.rt_stats_label.setText("未检测到任何目标")
            return

        self.detail_table.setRowCount(len(details))
        sum_conf = 0.0
        class_counts = {}

        for row, det in enumerate(details):
            cls_name = det["class_name"]

            # 动态展示附带 ID 的名称
            t_id = det.get("track_id")
            if t_id is not None:
                cls_name_display = f"{cls_name} (ID:{t_id})"
            else:
                cls_name_display = cls_name

            conf = det["confidence"]
            sum_conf += conf

            class_counts[cls_name] = class_counts.get(cls_name, 0) + 1

            x1, y1, x2, y2 = det["xyxy"]
            w, h = x2 - x1, y2 - y1

            self.detail_table.setItem(row, 0, QTableWidgetItem(str(row + 1)))
            self.detail_table.setItem(row, 1, QTableWidgetItem(cls_name_display))
            self.detail_table.setItem(row, 2, QTableWidgetItem(f"{conf:.3f}"))
            self.detail_table.setItem(row, 3, QTableWidgetItem(f"{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}"))
            self.detail_table.setItem(row, 4, QTableWidgetItem(f"{w:.1f} x {h:.1f}"))

        self.detail_table.resizeColumnsToContents()

        total_objs = len(details)
        avg_conf = sum_conf / total_objs
        time_ms = (1.0 / fps) * 1000 if fps > 0 else 0.0

        line1 = f"🎯 目标总数: {total_objs}  |  📊 平均置信度: {avg_conf:.3f}  |  ⏱️ 推理耗时: {time_ms:.1f} ms\n"
        counts_str = "  ".join([f"[{k}]: {v}" for k, v in class_counts.items()])
        line2 = f"🏷️ 类别统计: {counts_str}"

        self.rt_stats_label.setText(line1 + line2)

    def _on_frame(self, src, res, details, fps):
        self._record_frame_data(details)
        self._set_label_image(self.rt_src_label, src)
        self._set_label_image(self.rt_res_label, res)
        self._update_detail_table_and_stats(details, fps)
        self.rt_fps_label.setText(f"FPS: {fps:.2f}")

    def _on_monitor_frame(self, src, res, details, fps):
        self._record_frame_data(details)
        self._set_label_image(self.mon_src_label, src)
        self._set_label_image(self.mon_res_label, res)
        self.mon_fps_label.setText(f"FPS: {fps:.2f}")

    # ---------- Batch 处理 ----------
    def _on_batch_item(self, src, res, details, path, idx, total, fps):
        self.batch_total = total
        item = BatchItem(path=path, source=src, result=res, details=details, fps=fps)
        self.batch_items.append(item)
        self.current_batch_index = len(self.batch_items) - 1
        self._show_batch_item(self.current_batch_index)
        self.batch_fps_label.setText(f"整体处理速度: {fps:.2f} img/s")

    def _show_batch_item(self, idx):
        if idx < 0 or idx >= len(self.batch_items):
            return
        item = self.batch_items[idx]
        self._set_label_image(self.batch_src_label, item.source)
        self._set_label_image(self.batch_res_label, item.result)
        total = self.batch_total if self.batch_total else len(self.batch_items)
        self.batch_index_label.setText(f"{idx + 1}/{total}")

        filename = os.path.basename(item.path)
        if not item.details:
            self.batch_stats_label.setText(f"📁 当前文件: {filename}\n未检测到任何目标")
            return

        total_objs = len(item.details)
        sum_conf = sum(d["confidence"] for d in item.details)
        avg_conf = sum_conf / total_objs
        time_ms = (1.0 / item.fps) * 1000 if hasattr(item, 'fps') and item.fps > 0 else 0.0

        class_counts = {}
        for d in item.details:
            c = d["class_name"]
            class_counts[c] = class_counts.get(c, 0) + 1

        line1 = f"📁 当前文件: {filename}  |  🎯 目标总数: {total_objs}  |  📊 平均置信度: {avg_conf:.3f}  |  ⏱️ 单张耗时: {time_ms:.1f} ms\n"
        counts_str = "  ".join([f"[{k}]: {v}" for k, v in class_counts.items()])
        line2 = f"🏷️ 类别统计: {counts_str}"

        self.batch_stats_label.setText(line1 + line2)

    def _on_prev_batch(self):
        if not self.batch_items:
            return
        self.current_batch_index = max(0, self.current_batch_index - 1)
        self._show_batch_item(self.current_batch_index)

    def _on_next_batch(self):
        if not self.batch_items:
            return
        self.current_batch_index = min(len(self.batch_items) - 1, self.current_batch_index + 1)
        self._show_batch_item(self.current_batch_index)

    def _on_save_batch_csv(self):
        if not self.batch_items:
            self._log("没有批量结果可保存")
            return
        file_path, _ = QFileDialog.getSaveFileName(self, "保存CSV", "result.csv", "CSV Files (*.csv)")
        if not file_path:
            return

        rows = []
        for item in self.batch_items:
            if not item.details:
                rows.append({
                    "Image Name (图片名)": os.path.basename(item.path),
                    "Class (类别)": "", "Confidence (置信度)": "",
                    "x1": "", "y1": "", "x2": "", "y2": ""
                })
            else:
                for det in item.details:
                    x1, y1, x2, y2 = det["xyxy"]
                    rows.append({
                        "Image Name (图片名)": os.path.basename(item.path),
                        "Class (类别)": det["class_name"],
                        "Confidence (置信度)": round(det["confidence"], 4),
                        "x1": round(x1, 1), "y1": round(y1, 1),
                        "x2": round(x2, 1), "y2": round(y2, 1)
                    })

        df = pd.DataFrame(rows)
        df.to_csv(file_path, index=False, encoding='utf-8-sig')
        self._log(f"已保存批量CSV: {file_path}")

    def _on_clear_batch(self):
        self._clear_batch_internal()
        self._log("已清空批量结果")

    def _clear_batch_internal(self):
        self.batch_items.clear()
        self.batch_total = 0
        self.current_batch_index = -1
        self.batch_index_label.setText("0/0")
        self.batch_fps_label.setText("整体处理速度: 0.0 img/s")
        self.batch_stats_label.setText("等待检测...")
        self._init_image_label(self.batch_src_label, "检测源")
        self._init_image_label(self.batch_res_label, "检测结果")