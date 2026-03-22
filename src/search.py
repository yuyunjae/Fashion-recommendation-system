import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import faiss
import numpy as np
import psycopg2
import torch
from PIL import Image

from resnet_model import ResNet50Classifier, create_transforms


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


def load_faiss_index(index_path: Path) -> faiss.Index:
    if not index_path.exists():
        raise FileNotFoundError(f"Faiss index not found: {index_path}")
    return faiss.read_index(str(index_path))


def encode_query_image(
    model: ResNet50Classifier,
    image_path: Path,
    image_size: int,
    device: torch.device,
) -> np.ndarray:
    if not image_path.exists():
        raise FileNotFoundError(f"Query image not found: {image_path}")

    _, val_transform = create_transforms(image_size)
    image = Image.open(image_path).convert("RGB")
    image_tensor = val_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        feat = model.extract_feature_vector(image_tensor, normalize=True)

    return feat.cpu().numpy().astype("float32")


def fetch_metadata(conn, item_ids: List[int]) -> Dict[int, Dict[str, Any]]:
    if not item_ids:
        return {}

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT
                item_id,
                split,
                source_root_name,
                preprocessed_path,
                original_path,
                filename,
                image_id,
                year_code,
                style,
                gender,
                label
            FROM fashion_items
            WHERE item_id = ANY(%s)
            """,
            (item_ids,),
        )
        rows = cur.fetchall()

    result: Dict[int, Dict[str, Any]] = {}
    for row in rows:
        result[int(row[0])] = {
            "item_id": int(row[0]),
            "split": row[1],
            "source_root_name": row[2],
            "preprocessed_path": row[3],
            "original_path": row[4],
            "filename": row[5],
            "image_id": row[6],
            "year_code": row[7],
            "style": row[8],
            "gender": row[9],
            "label": row[10],
        }
    return result


def search_top_k(index: faiss.Index, query_vector: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
    scores, ids = index.search(query_vector, top_k)
    return scores[0], ids[0]


def run_search(
    checkpoint_path: Path,
    label_map_path: Path,
    faiss_index_path: Path,
    postgres_dsn: str,
    query_image_path: Path,
    image_size: int,
    top_k: int,
) -> List[Dict[str, Any]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(checkpoint_path, label_map_path, device)
    index = load_faiss_index(faiss_index_path)
    query_vector = encode_query_image(model, query_image_path, image_size, device)

    scores, ids = search_top_k(index, query_vector, top_k)
    valid_item_ids = [int(item_id) for item_id in ids.tolist() if int(item_id) != -1]

    conn = psycopg2.connect(postgres_dsn)
    try:
        metadata_by_id = fetch_metadata(conn, valid_item_ids)
    finally:
        conn.close()

    results: List[Dict[str, Any]] = []
    for rank, (score, item_id) in enumerate(zip(scores.tolist(), ids.tolist()), start=1):
        item_id = int(item_id)
        if item_id == -1:
            continue
        metadata = metadata_by_id.get(item_id)
        if metadata is None:
            continue

        results.append(
            {
                "rank": rank,
                "score": float(score),
                **metadata,
            }
        )
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search top-k similar fashion images with Faiss + PostgreSQL")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best_model_state.pth")
    parser.add_argument("--label-map", type=str, required=True, help="Path to label_to_index.json")
    parser.add_argument("--faiss-index", type=str, required=True, help="Path to .faiss index")
    parser.add_argument("--postgres-dsn", type=str, required=True, help="PostgreSQL DSN")
    parser.add_argument("--query-image", type=str, required=True, help="Query image path")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output-json", type=str, default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = run_search(
        checkpoint_path=Path(args.checkpoint),
        label_map_path=Path(args.label_map),
        faiss_index_path=Path(args.faiss_index),
        postgres_dsn=args.postgres_dsn,
        query_image_path=Path(args.query_image),
        image_size=args.image_size,
        top_k=args.top_k,
    )

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
