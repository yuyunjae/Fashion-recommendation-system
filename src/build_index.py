import argparse
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import faiss
import numpy as np
import pandas as pd
import psycopg2
import torch
from PIL import Image
from psycopg2.extras import execute_batch
from torch.utils.data import DataLoader, Dataset

from resnet_model import ResNet50Classifier, create_transforms


VALID_SPLIT = "valid"
TEST_SPLIT = "test"
ALLOWED_SPLITS = {VALID_SPLIT, TEST_SPLIT}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class ItemRecord:
    item_id: int
    split: str
    source_root_name: str
    preprocessed_path: str
    original_path: str
    filename: str
    image_id: str
    year_code: str
    style: str
    gender: str
    label: str


class InferenceDataset(Dataset):
    def __init__(self, records: Sequence[ItemRecord], transform):
        self.records = list(records)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        record = self.records[idx]
        image = Image.open(record.preprocessed_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, record.item_id


def parse_filename_tokens(file_name: str) -> Dict[str, str]:
    stem = Path(file_name).stem
    parts = stem.split("_")
    if len(parts) < 5:
        raise ValueError(f"Unexpected filename format: {file_name}")

    prefix = parts[0]
    image_id = parts[1]
    year_code = parts[2]
    style = parts[3]
    gender_token = parts[-1].upper()
    if gender_token not in {"M", "W"}:
        raise ValueError(f"Unexpected gender token in filename: {file_name}")

    return {
        "prefix": prefix,
        "image_id": image_id,
        "year_code": year_code,
        "style": style,
        "gender": "male" if gender_token == "M" else "female",
        "gender_token": gender_token,
    }


def get_source_root_name(split: str) -> str:
    if split == VALID_SPLIT:
        return "Training"
    if split == TEST_SPLIT:
        return "Validation"
    raise ValueError(f"Unsupported split: {split}")


def collect_original_name_map(root_dir: Path) -> Dict[str, str]:
    if not root_dir.exists():
        raise FileNotFoundError(f"Original root does not exist: {root_dir}")

    name_map: Dict[str, str] = {}
    duplicates: List[str] = []
    for path in sorted(root_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
            continue
        if path.name in name_map:
            duplicates.append(path.name)
            continue
        name_map[path.name] = str(path.resolve())

    if duplicates:
        sample = ", ".join(sorted(set(duplicates))[:10])
        raise ValueError(
            "Duplicate filenames found under original root. "
            f"Current design requires unique filenames per source root. Examples: {sample}"
        )
    return name_map


def load_manifest_records(
    manifest_path: Path,
    split: str,
    original_name_map: Dict[str, str],
    start_item_id: int,
) -> List[ItemRecord]:
    if split not in ALLOWED_SPLITS:
        raise ValueError(f"Unsupported split: {split}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    df = pd.read_csv(manifest_path)
    required_cols = {"path", "style", "gender", "label", "image_id"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Manifest missing columns {sorted(missing_cols)}: {manifest_path}")

    records: List[ItemRecord] = []
    source_root_name = get_source_root_name(split)
    next_item_id = start_item_id

    for row in df.itertuples(index=False):
        preprocessed_path = Path(getattr(row, "path")).resolve()
        filename = preprocessed_path.name
        if filename not in original_name_map:
            raise FileNotFoundError(
                f"Could not map preprocessed file '{filename}' back to original image in split={split}."
            )

        tokens = parse_filename_tokens(filename)
        style = str(getattr(row, "style"))
        gender = str(getattr(row, "gender"))
        label = str(getattr(row, "label"))
        image_id = str(getattr(row, "image_id"))

        records.append(
            ItemRecord(
                item_id=next_item_id,
                split=split,
                source_root_name=source_root_name,
                preprocessed_path=str(preprocessed_path),
                original_path=original_name_map[filename],
                filename=filename,
                image_id=image_id,
                year_code=tokens["year_code"],
                style=style,
                gender=gender,
                label=label,
            )
        )
        next_item_id += 1

    return records


def load_model(checkpoint_path: Path, label_map_path: Path, device: torch.device) -> ResNet50Classifier:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not label_map_path.exists():
        raise FileNotFoundError(f"label_to_index.json not found: {label_map_path}")

    with open(label_map_path, "r", encoding="utf-8") as f:
        label_to_index = json.load(f)

    model = ResNet50Classifier(num_classes=len(label_to_index), dropout=0.2, pretrained=False)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def extract_embeddings(
    model: ResNet50Classifier,
    records: Sequence[ItemRecord],
    image_size: int,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    _, val_transform = create_transforms(image_size)
    dataset = InferenceDataset(records, transform=val_transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    all_ids: List[np.ndarray] = []
    all_vecs: List[np.ndarray] = []

    with torch.no_grad():
        for images, item_ids in loader:
            images = images.to(device)
            feats = model.extract_feature_vector(images, normalize=True)
            all_vecs.append(feats.cpu().numpy().astype("float32"))
            all_ids.append(item_ids.numpy().astype("int64"))

    if not all_vecs:
        raise ValueError("No embeddings were extracted. Check the manifests and image paths.")

    vectors = np.concatenate(all_vecs, axis=0)
    ids = np.concatenate(all_ids, axis=0)
    return ids, vectors


def build_faiss_index(item_ids: np.ndarray, vectors: np.ndarray) -> faiss.IndexIDMap:
    if vectors.ndim != 2:
        raise ValueError(f"Expected 2D vectors, got shape={vectors.shape}")
    dim = vectors.shape[1]
    index = faiss.IndexIDMap(faiss.IndexFlatIP(dim))
    index.add_with_ids(vectors, item_ids)
    return index


def save_faiss_index(index: faiss.Index, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(output_path))


def connect_postgres(dsn: str):
    return psycopg2.connect(dsn)


def upsert_metadata_postgres(conn, records: Sequence[ItemRecord]) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS fashion_items (
                item_id BIGINT PRIMARY KEY,
                split VARCHAR(16) NOT NULL,
                source_root_name VARCHAR(32) NOT NULL,
                preprocessed_path TEXT NOT NULL,
                original_path TEXT NOT NULL,
                filename TEXT NOT NULL,
                image_id VARCHAR(64) NOT NULL,
                year_code VARCHAR(8) NOT NULL,
                style VARCHAR(128) NOT NULL,
                gender VARCHAR(16) NOT NULL,
                label VARCHAR(256) NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
            CREATE INDEX IF NOT EXISTS idx_fashion_items_split ON fashion_items(split);
            CREATE INDEX IF NOT EXISTS idx_fashion_items_label ON fashion_items(label);
            CREATE INDEX IF NOT EXISTS idx_fashion_items_filename ON fashion_items(filename);
            """
        )
        execute_batch(
            cur,
            """
            INSERT INTO fashion_items (
                item_id, split, source_root_name, preprocessed_path, original_path,
                filename, image_id, year_code, style, gender, label
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (item_id) DO UPDATE SET
                split = EXCLUDED.split,
                source_root_name = EXCLUDED.source_root_name,
                preprocessed_path = EXCLUDED.preprocessed_path,
                original_path = EXCLUDED.original_path,
                filename = EXCLUDED.filename,
                image_id = EXCLUDED.image_id,
                year_code = EXCLUDED.year_code,
                style = EXCLUDED.style,
                gender = EXCLUDED.gender,
                label = EXCLUDED.label;
            """,
            [
                (
                    r.item_id,
                    r.split,
                    r.source_root_name,
                    r.preprocessed_path,
                    r.original_path,
                    r.filename,
                    r.image_id,
                    r.year_code,
                    r.style,
                    r.gender,
                    r.label,
                )
                for r in records
            ],
            page_size=1000,
        )
    conn.commit()


def export_metadata_sqlite(sqlite_path: Path, records: Sequence[ItemRecord]) -> None:
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(sqlite_path))
    try:
        cur = conn.cursor()
        cur.executescript(
            """
            CREATE TABLE IF NOT EXISTS fashion_items (
                item_id INTEGER PRIMARY KEY,
                split TEXT NOT NULL,
                source_root_name TEXT NOT NULL,
                preprocessed_path TEXT NOT NULL,
                original_path TEXT NOT NULL,
                filename TEXT NOT NULL,
                image_id TEXT NOT NULL,
                year_code TEXT NOT NULL,
                style TEXT NOT NULL,
                gender TEXT NOT NULL,
                label TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE INDEX IF NOT EXISTS idx_fashion_items_split ON fashion_items(split);
            CREATE INDEX IF NOT EXISTS idx_fashion_items_label ON fashion_items(label);
            CREATE INDEX IF NOT EXISTS idx_fashion_items_filename ON fashion_items(filename);
            """
        )
        cur.executemany(
            """
            INSERT OR REPLACE INTO fashion_items (
                item_id, split, source_root_name, preprocessed_path, original_path,
                filename, image_id, year_code, style, gender, label
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    r.item_id,
                    r.split,
                    r.source_root_name,
                    r.preprocessed_path,
                    r.original_path,
                    r.filename,
                    r.image_id,
                    r.year_code,
                    r.style,
                    r.gender,
                    r.label,
                )
                for r in records
            ],
        )
        conn.commit()
    finally:
        conn.close()



def compute_label_centroids(records, item_ids, vectors):
    id_to_label = {r.item_id: r.label for r in records}
    buckets = {}

    for item_id, vec in zip(item_ids, vectors):
        label = id_to_label[int(item_id)]
        buckets.setdefault(label, []).append(vec)

    centroids = {}
    for label, vecs in buckets.items():
        mat = np.stack(vecs, axis=0).astype("float32")
        centroid = mat.mean(axis=0)
        centroid /= (np.linalg.norm(centroid) + 1e-12)
        centroids[label] = centroid
    return centroids


def save_label_centroids(centroids, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    labels = sorted(centroids.keys())
    matrix = np.stack([centroids[label] for label in labels], axis=0).astype("float32")

    np.savez(
        out_path,
        labels=np.array(labels, dtype=object),
        centroids=matrix,
    )

    meta_path = out_path.with_suffix(".json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "num_labels": len(labels),
                "dim": int(matrix.shape[1]),
                "labels": labels,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

def build_records_from_manifests(
    valid_manifest: Path,
    test_manifest: Path,
    training_original_root: Path,
    validation_original_root: Path,
) -> List[ItemRecord]:
    training_name_map = collect_original_name_map(training_original_root)
    validation_name_map = collect_original_name_map(validation_original_root)

    valid_records = load_manifest_records(
        manifest_path=valid_manifest,
        split=VALID_SPLIT,
        original_name_map=training_name_map,
        start_item_id=1,
    )
    test_records = load_manifest_records(
        manifest_path=test_manifest,
        split=TEST_SPLIT,
        original_name_map=validation_name_map,
        start_item_id=len(valid_records) + 1,
    )
    return valid_records + test_records


def save_build_summary(output_path: Path, records: Sequence[ItemRecord], index_path: Path, sqlite_export_path: Optional[Path]) -> None:
    summary = {
        "num_items": len(records),
        "num_valid_items": sum(1 for r in records if r.split == VALID_SPLIT),
        "num_test_items": sum(1 for r in records if r.split == TEST_SPLIT),
        "feature_dim": 2048,
        "faiss_metric": "inner_product_on_l2_normalized_vectors",
        "index_path": str(index_path.resolve()),
        "sqlite_export_path": str(sqlite_export_path.resolve()) if sqlite_export_path else None,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Faiss index and metadata DB for fashion recommendation")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best_model_state.pth")
    parser.add_argument("--label-map", type=str, required=True, help="Path to label_to_index.json")
    parser.add_argument("--valid-manifest", type=str, required=True, help="Path to val_manifest.csv")
    parser.add_argument("--test-manifest", type=str, required=True, help="Path to test_manifest.csv")
    parser.add_argument("--training-original-root", type=str, required=True, help="Training/01.원천데이터 root")
    parser.add_argument("--validation-original-root", type=str, required=True, help="Validation/01.원천데이터 root")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--faiss-index-out", type=str, required=True, help="Output path for .faiss index")
    parser.add_argument("--postgres-dsn", type=str, default="", help="PostgreSQL DSN for metadata upsert")
    parser.add_argument(
        "--sqlite-export",
        type=str,
        default="",
        help="Optional SQLite export path for local inspection or demo",
    )
    parser.add_argument("--summary-out", type=str, default="build_index_summary.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    records = build_records_from_manifests(
        valid_manifest=Path(args.valid_manifest),
        test_manifest=Path(args.test_manifest),
        training_original_root=Path(args.training_original_root),
        validation_original_root=Path(args.validation_original_root),
    )
    if not records:
        raise ValueError("No valid/test records found for indexing.")

    model = load_model(
        checkpoint_path=Path(args.checkpoint),
        label_map_path=Path(args.label_map),
        device=device,
    )
    item_ids, vectors = extract_embeddings(
        model=model,
        records=records,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
    )
    index_out = Path(args.faiss_index_out)
    centroid_out_path = index_out.with_name("label_centroids.npz")

    # label별 평균 저장
    label_centroids = compute_label_centroids(records, item_ids, vectors)
    save_label_centroids(label_centroids, centroid_out_path)

    index = build_faiss_index(item_ids, vectors)
    save_faiss_index(index, index_out)

    if args.postgres_dsn:
        conn = connect_postgres(args.postgres_dsn)
        try:
            upsert_metadata_postgres(conn, records)
        finally:
            conn.close()

    sqlite_export_path: Optional[Path] = None
    if args.sqlite_export:
        sqlite_export_path = Path(args.sqlite_export)
        export_metadata_sqlite(sqlite_export_path, records)

    save_build_summary(
        output_path=Path(args.summary_out),
        records=records,
        index_path=index_out,
        sqlite_export_path=sqlite_export_path,
    )

    print("[DONE] build_index completed")
    print(f"- indexed_items: {len(records)}")
    print(f"- faiss_index: {index_out.resolve()}")
    if args.postgres_dsn:
        print("- postgres: metadata upserted")
    if sqlite_export_path is not None:
        print(f"- sqlite_export: {sqlite_export_path.resolve()}")


if __name__ == "__main__":
    main()
