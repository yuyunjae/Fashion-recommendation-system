from pathlib import Path
from typing import Optional, Tuple

import torch
from PIL import Image
from ultralytics import YOLO

PERSON_CLASS_ID = 0
TARGET_WIDTH = 224
TARGET_HEIGHT = 448


def clamp(val: float, low: float, high: float) -> float:
    return max(low, min(high, val))


def load_detector(model_name: str = "yolov8s.pt") -> YOLO:
    return YOLO(model_name)


def select_largest_person_box(result, conf_thres: float = 0.25) -> Optional[Tuple[float, float, float, float]]:
    if result.boxes is None or len(result.boxes) == 0:
        return None

    best_box = None
    best_area = -1.0
    boxes_xyxy = result.boxes.xyxy.cpu().numpy()
    boxes_cls = result.boxes.cls.cpu().numpy()
    boxes_conf = result.boxes.conf.cpu().numpy()

    for xyxy, cls_id, conf in zip(boxes_xyxy, boxes_cls, boxes_conf):
        if int(cls_id) != PERSON_CLASS_ID:
            continue
        if float(conf) < conf_thres:
            continue
        x1, y1, x2, y2 = map(float, xyxy)
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        if area > best_area:
            best_area = area
            best_box = (x1, y1, x2, y2)
    return best_box


def expand_box_conservatively(
    box: Tuple[float, float, float, float],
    image_w: int,
    image_h: int,
    pad_x_ratio: float = 0.18,
    pad_y_ratio_top: float = 0.15,
    pad_y_ratio_bottom: float = 0.12,
) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    bw = x2 - x1
    bh = y2 - y1

    pad_x = bw * pad_x_ratio
    pad_top = bh * pad_y_ratio_top
    pad_bottom = bh * pad_y_ratio_bottom

    nx1 = int(clamp(x1 - pad_x, 0, image_w - 1))
    ny1 = int(clamp(y1 - pad_top, 0, image_h - 1))
    nx2 = int(clamp(x2 + pad_x, 1, image_w))
    ny2 = int(clamp(y2 + pad_bottom, 1, image_h))

    if nx2 <= nx1:
        nx2 = min(image_w, nx1 + 1)
    if ny2 <= ny1:
        ny2 = min(image_h, ny1 + 1)
    return nx1, ny1, nx2, ny2


def adjust_box_to_target_aspect(
    box: Tuple[int, int, int, int],
    image_w: int,
    image_h: int,
    target_w: int = TARGET_WIDTH,
    target_h: int = TARGET_HEIGHT,
) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    bw = x2 - x1
    bh = y2 - y1

    target_ratio = target_w / target_h
    current_ratio = bw / bh
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    if current_ratio > target_ratio:
        new_bw = bw
        new_bh = bw / target_ratio
    else:
        new_bh = bh
        new_bw = bh * target_ratio

    nx1 = int(round(cx - new_bw / 2))
    nx2 = int(round(cx + new_bw / 2))
    ny1 = int(round(cy - new_bh / 2))
    ny2 = int(round(cy + new_bh / 2))

    if nx1 < 0:
        shift = -nx1
        nx1 += shift
        nx2 += shift
    if ny1 < 0:
        shift = -ny1
        ny1 += shift
        ny2 += shift
    if nx2 > image_w:
        shift = nx2 - image_w
        nx1 -= shift
        nx2 -= shift
    if ny2 > image_h:
        shift = ny2 - image_h
        ny1 -= shift
        ny2 -= shift

    nx1 = int(clamp(nx1, 0, image_w - 1))
    ny1 = int(clamp(ny1, 0, image_h - 1))
    nx2 = int(clamp(nx2, 1, image_w))
    ny2 = int(clamp(ny2, 1, image_h))

    if nx2 <= nx1 or ny2 <= ny1:
        return x1, y1, x2, y2
    return nx1, ny1, nx2, ny2


def crop_person_from_result(image: Image.Image, result, conf_thres: float = 0.25) -> Tuple[Image.Image, bool]:
    image_rgb = image.convert("RGB")
    w, h = image_rgb.size
    person_box = select_largest_person_box(result, conf_thres=conf_thres)
    if person_box is None:
        return image_rgb.resize((TARGET_WIDTH, TARGET_HEIGHT), Image.BILINEAR), False

    box = expand_box_conservatively(person_box, w, h)
    box = adjust_box_to_target_aspect(box, w, h, TARGET_WIDTH, TARGET_HEIGHT)
    cropped = image_rgb.crop(box)
    return cropped.resize((TARGET_WIDTH, TARGET_HEIGHT), Image.BILINEAR), True


def preprocess_query_image_with_yolo(
    image: Image.Image,
    detector: YOLO,
    conf_thres: float = 0.25,
):
    device = 0 if torch.cuda.is_available() else "cpu"
    results = detector.predict(
        source=[image.convert("RGB")],
        conf=conf_thres,
        verbose=False,
        classes=[PERSON_CLASS_ID],
        device=device,
    )
    return crop_person_from_result(image, results[0], conf_thres=conf_thres)
