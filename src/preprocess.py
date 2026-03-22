import argparse
from pathlib import Path
from typing import List, Optional, Tuple
import random
import shutil
import torch
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO


PERSON_CLASS_ID = 0  # COCO person
TARGET_WIDTH = 224
TARGET_HEIGHT = 448


def collect_image_paths(root_dir: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_paths = []
    for p in root_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            image_paths.append(p)
    return sorted(image_paths)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def clamp(val: float, low: float, high: float) -> float:
    return max(low, min(high, val))


def chunk_list(items: List[Path], batch_size: int):
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


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


def crop_person_from_result(
    image: Image.Image,
    result,
    image_path: Path,
    conf_thres: float = 0.25,
) -> Image.Image:
    image_rgb = image.convert("RGB")
    w, h = image_rgb.size

    person_box = select_largest_person_box(result, conf_thres=conf_thres)

    if person_box is None:
        return image_rgb.resize((TARGET_WIDTH, TARGET_HEIGHT), Image.BILINEAR)

    box = expand_box_conservatively(person_box, w, h)
    box = adjust_box_to_target_aspect(box, w, h, TARGET_WIDTH, TARGET_HEIGHT)

    cropped = image_rgb.crop(box)
    return cropped.resize((TARGET_WIDTH, TARGET_HEIGHT), Image.BILINEAR)


def build_existing_name_set(dirs: List[Path]) -> set[str]:
    existing = set()
    for d in dirs:
        if not d.exists():
            continue
        for p in d.iterdir():
            if p.is_file():
                existing.add(p.name)
    return existing

def process_split(
    input_root: Path,
    output_root: Path,
    detector: YOLO,
    conf_thres: float = 0.25,
    batch_size: int = 128,
    extra_skip_dirs: Optional[List[Path]] = None    
) -> None:
    ensure_dir(output_root)

    if not input_root.exists():
        raise FileNotFoundError(f"Input directory not found: {input_root}")

    all_image_paths = collect_image_paths(input_root)
    print(f"[INFO] Found {len(all_image_paths)} images in: {input_root}")


    skip_dirs = [output_root]
    if extra_skip_dirs:
        skip_dirs.extend(extra_skip_dirs)

    existing_names = build_existing_name_set(skip_dirs)

    # 이미 생성된 파일은 제외
    image_paths = []
    skipped_existing = 0
    for img_path in all_image_paths:
        if img_path.name in existing_names:
            skipped_existing += 1
            continue
        image_paths.append(img_path)


    print(f"[INFO] Skip existing files: {skipped_existing}")
    print(f"[INFO] Remaining files to process: {len(image_paths)}")

    success = 0
    failed = 0

    device = 0 if torch.cuda.is_available() else "cpu"
    total_batches = (len(image_paths) + batch_size - 1) // batch_size

    for batch_paths in tqdm(chunk_list(image_paths, batch_size), total=total_batches, desc=f"Processing {input_root.name}"):
        batch_images = []
        valid_paths = []

        # 이미지 로드
        for img_path in batch_paths:
            try:
                img = Image.open(img_path).convert("RGB")
                batch_images.append(img)
                valid_paths.append(img_path)
            except Exception as e:
                failed += 1
                print(f"[WARN] Failed to open: {img_path} | {e}")

        if not batch_images:
            continue

        try:
            # batch inference
            results = detector.predict(
                source=batch_images,
                conf=conf_thres,
                verbose=False,
                classes=[PERSON_CLASS_ID],
                device=device,
            )

            for img_path, img, result in zip(valid_paths, batch_images, results):
                out_path = output_root / img_path.name

                # train/valid 재실행 대응
                if img_path.name in existing_names or out_path.exists():
                    continue

                try:
                    processed = crop_person_from_result(
                        image=img,
                        result=result,
                        image_path=img_path,
                        conf_thres=conf_thres,
                    )
                    processed.save(out_path, quality=90)
                    success += 1
                except Exception as e:
                    failed += 1
                    print(f"[WARN] Failed to process/save: {img_path} | {e}")

        except Exception as e:
            failed += len(valid_paths)
            print(f"[WARN] Batch inference failed | {e}")

        finally:
            for img in batch_images:
                img.close()

    print(f"[DONE] {input_root}")
    print(f"  success         : {success}")
    print(f"  failed          : {failed}")
    print(f"  skipped_existing: {skipped_existing}")
    print(f"  output          : {output_root}")



def split_preprocessed_train_valid(
    train_dir: Path,
    valid_dir: Path,
    valid_ratio: float = 0.3,
    seed: int = 42,
) -> None:
    ensure_dir(valid_dir)

    valid_existing = collect_image_paths(valid_dir)
    if len(valid_existing) > 0:
        print(f"[INFO] Valid directory already has {len(valid_existing)} files. Skip splitting.")
        return

    train_images = collect_image_paths(train_dir)
    if len(train_images) == 0:
        print("[WARN] No images found in preprocess/train. Skip splitting.")
        return

    num_valid = int(len(train_images) * valid_ratio)
    if num_valid <= 0:
        print("[WARN] valid_ratio too small or train set too small. Skip splitting.")
        return

    rng = random.Random(seed)
    shuffled = train_images[:]
    rng.shuffle(shuffled)

    valid_images = shuffled[:num_valid]

    print(f"[INFO] Splitting preprocess/train -> valid")
    print(f"[INFO] Total train images before split: {len(train_images)}")
    print(f"[INFO] Move to valid: {len(valid_images)}")
    print(f"[INFO] Keep in train: {len(train_images) - len(valid_images)}")

    moved = 0
    failed = 0

    for src_path in tqdm(valid_images, desc="Moving valid images"):
        dst_path = valid_dir / src_path.name

        try:
            if dst_path.exists():
                continue
            shutil.move(str(src_path), str(dst_path))
            moved += 1
        except Exception as e:
            failed += 1
            print(f"[WARN] Failed to move: {src_path} -> {dst_path} | {e}")

    print("[DONE] Train/Valid split finished.")
    print(f"  moved : {moved}")
    print(f"  failed: {failed}")
    print(f"  train : {train_dir}")
    print(f"  valid : {valid_dir}")

def main():
    project_root = Path(__file__).resolve().parent.parent
    default_dataset_root = project_root / "패션 데이터셋"

    parser = argparse.ArgumentParser(description="Preprocess fashion dataset with YOLO person detection")
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=str(default_dataset_root),
        help="패션 데이터셋 루트 디렉토리",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8s.pt",
        help="YOLO model name. ex) yolov8n.pt, yolov8s.pt, yolov8m.pt",
    )
    parser.add_argument(
        "--conf-thres",
        type=float,
        default=0.25,
        help="person detection confidence threshold",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="YOLO batch inference size",
    )

    parser.add_argument(
    "--valid-ratio",
    type=float,
    default=0.3,
    help="preprocess/train 에서 preprocess/valid 로 이동할 비율",
    )

    parser.add_argument(
        "--split-seed",
        type=int,
        default=42,
        help="train/valid split random seed",
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).resolve()

    train_input = dataset_root / "Training" / "01.원천데이터"
    test_input = dataset_root / "Validation" / "01.원천데이터"

    train_output = dataset_root / "preprocess" / "train"
    valid_output = dataset_root / "preprocess" / "valid"
    test_output = dataset_root / "preprocess" / "test"

    print(f"[INFO] Project root : {project_root}")
    print(f"[INFO] Dataset root : {dataset_root}")
    print(f"[INFO] Train input  : {train_input}")
    print(f"[INFO] Test input   : {test_input}")
    print(f"[INFO] Train output : {train_output}")
    print(f"[INFO] Valid output : {valid_output}")
    print(f"[INFO] Test output  : {test_output}")
    print(f"[INFO] Batch size   : {args.batch_size}")
    print(f"[INFO] CUDA avail   : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[INFO] GPU name     : {torch.cuda.get_device_name(0)}")

    print("[INFO] Loading detector...")
    detector = load_detector(args.model)

    process_split(
        input_root=train_input,
        output_root=train_output,
        detector=detector,
        conf_thres=args.conf_thres,
        batch_size=args.batch_size,
        extra_skip_dirs=[valid_output]
    )

    process_split(
        input_root=test_input,
        output_root=test_output,
        detector=detector,
        conf_thres=args.conf_thres,
        batch_size=args.batch_size,
    )

    split_preprocessed_train_valid(train_dir=train_output, valid_dir=valid_output)

    print("\nAll preprocessing done.")


if __name__ == "__main__":
    main()