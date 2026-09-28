"""Chest disease detection service (Tab 2) — "Model Service 2".

Returns class + bounding box + confidence for each finding (BRD FR-4.3 - FR-4.6).
The model is NOT included yet. Drop it into `model_artifacts/cv/` together with `config.json`
(copy `config.example.json`); no code changes are needed. Supported frameworks:

  ultralytics     YOLOv8/11 `.pt` weights           (needs `pip install ultralytics`)
  yolo_onnx       YOLOv8/11 exported to ONNX        (`yolo export model=best.pt format=onnx`)
  classifier_onnx plain image classifier in ONNX    (no bounding box, class + confidence only)
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

CV_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model_artifacts", "cv")

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_DEFAULT_NORMAL = ("normal", "no finding", "no_finding", "no abnormality")


@dataclass
class Detection:
    label: str
    confidence: float
    box: Optional[Tuple[float, float, float, float]] = None  # x1, y1, x2, y2 in ORIGINAL image pixels


@dataclass
class DetectionResult:
    findings: List[Detection] = field(default_factory=list)  # empty => no abnormal finding
    has_boxes: bool = True
    scores: Optional[List[Tuple[str, float]]] = None  # classifier mode: top-k class probabilities


# --------------------------------------------------------------------- helpers
def _letterbox(img: Image.Image, size_hw):
    H, W = size_hw
    w, h = img.size
    r = min(W / w, H / h)
    nw, nh = max(1, round(w * r)), max(1, round(h * r))
    canvas = Image.new("RGB", (W, H), (114, 114, 114))
    px, py = (W - nw) // 2, (H - nh) // 2
    canvas.paste(img.resize((nw, nh), Image.BILINEAR), (px, py))
    return canvas, r, px, py


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> List[int]:
    order, keep = scores.argsort()[::-1], []
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    while order.size:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        rest = order[1:]
        xx1, yy1 = np.maximum(boxes[i, 0], boxes[rest, 0]), np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2, yy2 = np.minimum(boxes[i, 2], boxes[rest, 2]), np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        iou = inter / (areas[i] + areas[rest] - inter + 1e-9)
        order = rest[iou <= iou_thr]
    return keep


def describe_location(box, image_size) -> str:
    """Coarse zone of a box. Assumes standard PA/AP orientation (patient's right = image left)."""
    W, H = image_size
    cx, cy = (box[0] + box[2]) / 2 / W, (box[1] + box[3]) / 2 / H
    zone = "Upper" if cy < 1 / 3 else "Middle" if cy < 2 / 3 else "Lower"
    side = "patient's right" if cx < 0.5 else "patient's left"
    return f"{zone} zone, {side} side of the chest"


def draw_detections(image: Image.Image, findings: List[Detection]) -> Image.Image:
    img = image.convert("RGB").copy()
    d = ImageDraw.Draw(img)
    thick = max(2, round(min(img.size) / 250))
    font = ImageFont.load_default(size=max(12, round(min(img.size) / 32)))
    for f in findings:
        if f.box is None:
            continue
        x1, y1, x2, y2 = f.box
        d.rectangle([x1, y1, x2, y2], outline=(235, 64, 52), width=thick)
        text = f"{f.label} {f.confidence:.2f}"
        tb = d.textbbox((0, 0), text, font=font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        ty = y1 - th - 6 if y1 - th - 6 > 0 else y1 + 2
        d.rectangle([x1, ty, x1 + tw + 8, ty + th + 6], fill=(235, 64, 52))
        d.text((x1 + 4, ty + 2), text, fill="white", font=font)
    return img


# --------------------------------------------------------------------- service
class DetectionService:
    def __init__(self, cv_dir: str = CV_DIR):
        self.cv_dir = cv_dir
        self.config: dict = {}
        self.ready = False
        self.message = ""
        self._model = None
        self._input_name = None
        self._labels: List[str] = []
        self._load()

    def _load(self) -> None:
        cfg_path = os.path.join(self.cv_dir, "config.json")
        if not os.path.isfile(cfg_path):
            self.message = "No `config.json` found in `model_artifacts/cv/`."
            return
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                self.config = json.load(f)
        except Exception as e:
            self.message = f"`config.json` could not be read: {e}"
            return

        fw = self.config.get("framework", "yolo_onnx").lower()
        model_file = self.config.get("model_file")
        path = os.path.join(self.cv_dir, model_file or "")
        if not model_file or not os.path.isfile(path):
            self.message = f"Model file `{model_file}` was not found in `model_artifacts/cv/`."
            return
        self._labels = list(self.config.get("labels", []))
        if fw in ("yolo_onnx", "classifier_onnx") and not self._labels:
            self.message = "`config.json` must contain a non-empty `labels` list (same order as training)."
            return
        try:
            if fw in ("yolo_onnx", "classifier_onnx"):
                import onnxruntime as ort

                self._model = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
                self._input_name = self._model.get_inputs()[0].name
            elif fw == "ultralytics":
                from ultralytics import YOLO

                self._model = YOLO(path)
                if not self._labels:
                    self._labels = [v for _, v in sorted(self._model.names.items())]
            else:
                self.message = f"Unsupported framework '{fw}' (use ultralytics, yolo_onnx or classifier_onnx)."
                return
        except ImportError as e:
            self.message = f"Missing dependency for framework '{fw}': {e}. Add it to requirements.txt."
            return
        except Exception as e:
            self.message = f"Failed to load the model: {e}"
            return
        self.ready, self.message = True, "Model loaded."

    # ----------------------------------------------------------- properties
    @property
    def framework(self) -> str:
        return self.config.get("framework", "yolo_onnx").lower()

    @property
    def default_conf(self) -> float:
        return float(self.config.get("conf_threshold", 0.25))

    @property
    def labels(self) -> List[str]:
        return list(self._labels)

    # ------------------------------------------------------------ inference
    def predict(self, image: Image.Image, conf: Optional[float] = None) -> DetectionResult:
        if not self.ready:
            raise RuntimeError(self.message)
        conf = self.default_conf if conf is None else conf
        image = image.convert("RGB")
        if self.framework == "ultralytics":
            return self._predict_ultralytics(image, conf)
        if self.framework == "yolo_onnx":
            return self._predict_yolo_onnx(image, conf)
        return self._predict_classifier(image, conf)

    def _predict_ultralytics(self, image, conf) -> DetectionResult:
        res = self._model.predict(np.asarray(image), conf=conf, verbose=False)[0]
        findings = []
        for b in res.boxes:
            cid = int(b.cls[0])
            label = self._labels[cid] if cid < len(self._labels) else str(cid)
            findings.append(Detection(label, float(b.conf[0]), tuple(float(v) for v in b.xyxy[0].tolist())))
        return DetectionResult(sorted(findings, key=lambda f: -f.confidence))

    def _predict_yolo_onnx(self, image, conf) -> DetectionResult:
        shape = self._model.get_inputs()[0].shape  # e.g. [1, 3, 640, 640]
        size = self.config.get("input_size") or [
            shape[2] if isinstance(shape[2], int) else 640,
            shape[3] if isinstance(shape[3], int) else 640,
        ]
        canvas, r, px, py = _letterbox(image, size)
        x = (np.asarray(canvas, dtype=np.float32) / 255.0).transpose(2, 0, 1)[None]
        raw = np.squeeze(self._model.run(None, {self._input_name: x})[0], axis=0)
        if raw.shape[0] < raw.shape[1]:  # (4+nc, N) -> (N, 4+nc)
            raw = raw.T
        nc = len(self._labels)
        if raw.shape[1] != 4 + nc:
            raise ValueError(
                f"Model output has {raw.shape[1] - 4} classes but config lists {nc} labels "
                "(YOLOv8/11 export expected: 4 box values + class scores)."
            )
        scores_all = raw[:, 4:]
        cls_ids, confs = scores_all.argmax(1), scores_all.max(1)
        keep = confs >= conf
        raw, cls_ids, confs = raw[keep], cls_ids[keep], confs[keep]
        if len(raw) == 0:
            return DetectionResult([])
        cx, cy, w, h = raw[:, 0], raw[:, 1], raw[:, 2], raw[:, 3]
        boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - px) / r
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - py) / r
        W, H = image.size
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, W)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, H)
        offset = cls_ids[:, None] * (max(W, H) + 1)  # class-aware NMS
        kept = _nms(boxes + offset, confs, float(self.config.get("iou_threshold", 0.45)))
        return DetectionResult(
            [Detection(self._labels[int(cls_ids[i])], float(confs[i]), tuple(float(v) for v in boxes[i])) for i in kept]
        )

    def _predict_classifier(self, image, conf) -> DetectionResult:
        h, w = self.config.get("input_size", [224, 224])
        gray = self.config.get("color_mode", "RGB").upper() in ("L", "GRAY", "GRAYSCALE")
        arr = np.asarray(image.convert("L" if gray else "RGB").resize((w, h)), dtype=np.float32)
        if arr.ndim == 2:
            arr = arr[..., None]
        norm = self.config.get("normalize", "zero_one")
        if norm == "zero_one":
            arr = arr / 255.0
        elif norm == "minus_one_one":
            arr = arr / 127.5 - 1.0
        elif norm == "imagenet":
            arr = (arr / 255.0 - _IMAGENET_MEAN[: arr.shape[-1]]) / _IMAGENET_STD[: arr.shape[-1]]
        if self.config.get("channels_first", False):
            arr = arr.transpose(2, 0, 1)
        out = np.asarray(self._model.run(None, {self._input_name: arr[None].astype(np.float32)})[0], dtype=np.float64).reshape(-1)
        if out.size == 1:
            p = out[0] if 0 <= out[0] <= 1 else 1 / (1 + np.exp(-out[0]))
            out = np.array([1 - p, p])
        elif not (np.all(out >= 0) and np.isclose(out.sum(), 1.0, atol=1e-3)):
            e = np.exp(out - out.max())
            out = e / e.sum()
        if len(out) != len(self._labels):
            raise ValueError(f"Model returned {len(out)} outputs but config lists {len(self._labels)} labels.")
        order = np.argsort(out)[::-1]
        scores = [(self._labels[i], float(out[i])) for i in order[:5]]
        normal = {n.lower() for n in self.config.get("normal_labels", _DEFAULT_NORMAL)}
        top_label, top_p = scores[0]
        findings = [] if (top_label.lower() in normal or top_p < conf) else [Detection(top_label, top_p, None)]
        return DetectionResult(findings, has_boxes=False, scores=scores)
