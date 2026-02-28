import torch
from ultralytics import YOLO


class YOLOEngine:
    def __init__(self):
        self.model = None
        self.names = None

    def load_model(self, pt_path: str, device: str | None = None):
        self.model = YOLO(pt_path)

        if device is None:
            # Auto 模式：有卡用卡，没卡用 CPU
            if torch.cuda.is_available():
                device = "cuda:0"
            else:
                device = "cpu"
        elif device.startswith("cuda") and not torch.cuda.is_available():
            # 【新增防御】用户强制选了 GPU，但系统检测不到显卡
            print("警告: 未检测到可用 GPU，已自动回退到 CPU")
            device = "cpu"

        self.model.to(device)
        self.names = self.model.names

        # 返回最终决定使用的设备名，方便 UI 显示
        return device

    def is_loaded(self):
        return self.model is not None

    def predict(self, image_bgr, conf=0.25, iou=0.45):
        if self.model is None:
            raise RuntimeError("Model not loaded")
        results = self.model.predict(image_bgr, conf=conf, iou=iou, verbose=False)
        r = results[0]
        annotated = r.plot()  # BGR
        details = []

        boxes = r.boxes
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            clss = boxes.cls.cpu().numpy()
            for i in range(len(boxes)):
                cls_id = int(clss[i])
                name = self.names[cls_id] if isinstance(self.names, (list, tuple)) else self.names.get(cls_id,
                                                                                                       str(cls_id))
                details.append({
                    "class_id": cls_id,
                    "class_name": name,
                    "confidence": float(confs[i]),
                    "xyxy": [float(x) for x in xyxy[i].tolist()],
                })
        return annotated, details

    def track(self, image_bgr, conf=0.25, iou=0.45):
        if self.model is None:
            raise RuntimeError("Model not loaded")

        # persist=True 告诉模型记住上一帧的信息，实现连贯追踪
        results = self.model.track(image_bgr, conf=conf, iou=iou, persist=True, verbose=False)
        r = results[0]
        annotated = r.plot()
        details = []

        boxes = r.boxes
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            clss = boxes.cls.cpu().numpy()
            # 提取追踪分配的唯一 ID (如果丢失或刚进入画面，可能为空)
            t_ids = boxes.id.int().cpu().tolist() if boxes.id is not None else [None] * len(boxes)

            for i in range(len(boxes)):
                cls_id = int(clss[i])
                name = self.names[cls_id] if isinstance(self.names, (list, tuple)) else self.names.get(cls_id,
                                                                                                       str(cls_id))
                details.append({
                    "class_id": cls_id,
                    "class_name": name,
                    "confidence": float(confs[i]),
                    "xyxy": [float(x) for x in xyxy[i].tolist()],
                    "track_id": t_ids[i]  # 【新增】存储轨迹 ID
                })
        return annotated, details