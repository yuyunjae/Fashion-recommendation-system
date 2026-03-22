import argparse
import io
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import faiss
import numpy as np
import psycopg2
import torch
from PIL import Image

from person_crop import load_detector, preprocess_query_image_with_yolo
from resnet_model import ResNet50Classifier, create_transforms

GENDER_ALIASES = {
    "all": None,
    "male": "male",
    "man": "male",
    "m": "male",
    "female": "female",
    "woman": "female",
    "w": "female",
}


class FashionSearchEngine:
    def __init__(
        self,
        checkpoint_path: Path,
        label_map_path: Path,
        faiss_index_path: Path,
        postgres_dsn: str,
        image_size: int = 224,
        yolo_model: str = "yolov8s.pt",
        yolo_conf: float = 0.25,
        label_centroids_path: Optional[Path] = None,
        preferred_weight: float = 0.15,
        disliked_weight: float = 0.15,
        search_pool_factor: int = 5,
        max_search_pool: int = 200,
        device: Optional[str] = None,
    ):
        self.device = self._resolve_device(device)
        self.model = self._load_model(checkpoint_path, label_map_path)
        self.index = self._load_faiss_index(faiss_index_path)
        self.postgres_dsn = postgres_dsn
        self.image_size = image_size
        self.yolo_conf = yolo_conf
        self.detector = load_detector(yolo_model)
        self.label_centroids = self._load_label_centroids(label_centroids_path)
        self.preferred_weight = preferred_weight
        self.disliked_weight = disliked_weight
        self.search_pool_factor = max(1, int(search_pool_factor))
        self.max_search_pool = max(1, int(max_search_pool))
        _, self.val_transform = create_transforms(image_size)

    @staticmethod
    def _resolve_device(device: Optional[str]) -> torch.device:
        if device:
            return torch.device(device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _load_model(self, checkpoint_path: Path, label_map_path: Path) -> ResNet50Classifier:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        if not label_map_path.exists():
            raise FileNotFoundError(f"Label map not found: {label_map_path}")
        with open(label_map_path, "r", encoding="utf-8") as f:
            label_to_index = json.load(f)
        model = ResNet50Classifier(num_classes=len(label_to_index), dropout=0.2, pretrained=False)
        state_dict = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        return model

    @staticmethod
    def _load_faiss_index(index_path: Path) -> faiss.Index:
        if not index_path.exists():
            raise FileNotFoundError(f"Faiss index not found: {index_path}")
        return faiss.read_index(str(index_path))

    @staticmethod
    def _load_label_centroids(path: Optional[Path]) -> Dict[str, np.ndarray]:
        if path is None:
            return {}
        if not path.exists():
            raise FileNotFoundError(f"Label centroids file not found: {path}")
        data = np.load(path, allow_pickle=True)
        labels = data["labels"].tolist()
        centroids = data["centroids"].astype("float32")
        return {str(label): centroids[i] for i, label in enumerate(labels)}

    @staticmethod
    def normalize_gender_filter(value: str) -> Optional[str]:
        key = value.strip().lower()
        if key not in GENDER_ALIASES:
            raise ValueError(f"Unsupported gender filter: {value}. Use one of all/man/woman/male/female.")
        return GENDER_ALIASES[key]

    @staticmethod
    def parse_csv_list(value: Optional[str]) -> List[str]:
        if not value:
            return []
        return [v.strip() for v in value.split(",") if v.strip()]

    @staticmethod
    def l2_normalize(vector: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vector, axis=1, keepdims=True)
        norm = np.maximum(norm, 1e-12)
        return vector / norm

    def _encode_query_image(self, image: Image.Image) -> Tuple[np.ndarray, Dict[str, Any]]:
        cropped, detected_person = preprocess_query_image_with_yolo(image, self.detector, conf_thres=self.yolo_conf)
        tensor = self.val_transform(cropped).unsqueeze(0).to(self.device)
        with torch.no_grad():
            vec = self.model.extract_feature_vector(tensor, normalize=True)
        return vec.cpu().numpy().astype("float32"), {
            "person_detected": bool(detected_person),
            "preprocessed_size": list(cropped.size),
        }

    def _fetch_metadata(self, item_ids: Sequence[int]) -> Dict[int, Dict[str, Any]]:
        clean_ids = [int(x) for x in item_ids if int(x) != -1]
        if not clean_ids:
            return {}
        sql = """
            SELECT item_id, split, source_root_name, preprocessed_path, original_path,
                   filename, image_id, year_code, style, gender, label
            FROM fashion_items
            WHERE item_id = ANY(%s)
        """
        with psycopg2.connect(self.postgres_dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (clean_ids,))
                rows = cur.fetchall()
        out: Dict[int, Dict[str, Any]] = {}
        for row in rows:
            out[int(row[0])] = {
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
        return out

    def _fetch_labels_for_styles(self, styles: Sequence[str], gender_filter: Optional[str]) -> List[str]:
        styles = [s.strip() for s in styles if s and s.strip()]
        if not styles:
            return []
        sql = """
            SELECT DISTINCT label
            FROM fashion_items
            WHERE style = ANY(%s)
              AND (%s IS NULL OR gender = %s)
            ORDER BY label
        """
        with psycopg2.connect(self.postgres_dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (styles, gender_filter, gender_filter))
                rows = cur.fetchall()
        return [str(r[0]) for r in rows]

    def _apply_preference_bias(
        self,
        query_vector: np.ndarray,
        preferred_labels: Sequence[str],
        disliked_labels: Sequence[str],
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        adjusted = query_vector.copy()
        used_preferred = [label for label in preferred_labels if label in self.label_centroids]
        used_disliked = [label for label in disliked_labels if label in self.label_centroids]
        missing_preferred = [label for label in preferred_labels if label not in self.label_centroids]
        missing_disliked = [label for label in disliked_labels if label not in self.label_centroids]

        if used_preferred:
            pref_centroid = np.stack([self.label_centroids[l] for l in used_preferred], axis=0).mean(axis=0)
            adjusted = adjusted + self.preferred_weight * pref_centroid.reshape(1, -1)
        if used_disliked:
            dis_centroid = np.stack([self.label_centroids[l] for l in used_disliked], axis=0).mean(axis=0)
            adjusted = adjusted - self.disliked_weight * dis_centroid.reshape(1, -1)

        adjusted = self.l2_normalize(adjusted)
        debug = {
            "used_preferred_labels": used_preferred,
            "used_disliked_labels": used_disliked,
            "missing_preferred_labels": missing_preferred,
            "missing_disliked_labels": missing_disliked,
            "preferred_weight": self.preferred_weight,
            "disliked_weight": self.disliked_weight,
        }
        return adjusted, debug

    def _search_ids(self, query_vector: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        scores, ids = self.index.search(query_vector, k)
        return scores[0], ids[0]

    @staticmethod
    def _partition_results(
        scores: np.ndarray,
        ids: np.ndarray,
        metadata_by_id: Dict[int, Dict[str, Any]],
        gender_filter: Optional[str],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        strict: List[Dict[str, Any]] = []
        fallback: List[Dict[str, Any]] = []
        seen = set()
        for score, item_id in zip(scores.tolist(), ids.tolist()):
            item_id = int(item_id)
            if item_id == -1 or item_id in seen:
                continue
            seen.add(item_id)
            metadata = metadata_by_id.get(item_id)
            if metadata is None:
                continue
            row = {"score": float(score), **metadata}
            if gender_filter is None or str(metadata["gender"]).lower() == gender_filter:
                strict.append(row)
            else:
                fallback.append(row)
        return strict, fallback

    def search(
        self,
        image: Image.Image,
        top_k: int = 5,
        gender: str = "all",
        preferred_styles: Optional[Sequence[str]] = None,
        disliked_styles: Optional[Sequence[str]] = None,
        fallback_fill: bool = True,
    ) -> Dict[str, Any]:
        preferred_styles = list(preferred_styles or [])
        disliked_styles = list(disliked_styles or [])
        gender_filter = self.normalize_gender_filter(gender)

        query_vector, preprocess_info = self._encode_query_image(image)

        matched_preferred_labels = self._fetch_labels_for_styles(preferred_styles, gender_filter)
        matched_disliked_labels = self._fetch_labels_for_styles(disliked_styles, gender_filter)

        preference_debug = {
            "used_preferred_labels": [],
            "used_disliked_labels": [],
            "missing_preferred_labels": [],
            "missing_disliked_labels": [],
            "preferred_weight": self.preferred_weight,
            "disliked_weight": self.disliked_weight,
        }

        if self.label_centroids and (matched_preferred_labels or matched_disliked_labels):
            query_vector, preference_debug = self._apply_preference_bias(
                query_vector,
                matched_preferred_labels,
                matched_disliked_labels,
            )

        faiss_search_k = min(max(top_k * self.search_pool_factor, top_k), self.max_search_pool)
        scores, ids = self._search_ids(query_vector, faiss_search_k)
        metadata_by_id = self._fetch_metadata(ids.tolist())
        strict_results, fallback_results = self._partition_results(scores, ids, metadata_by_id, gender_filter)

        final_rows: List[Dict[str, Any]] = []
        final_rows.extend(strict_results[:top_k])
        fallback_used = False
        if len(final_rows) < top_k and fallback_fill:
            need = top_k - len(final_rows)
            final_rows.extend(fallback_results[:need])
            fallback_used = need > 0 and len(fallback_results) > 0

        results: List[Dict[str, Any]] = []
        for rank, row in enumerate(final_rows, start=1):
            row = dict(row)
            row["rank"] = rank
            results.append(row)

        return {
            "top_k": int(top_k),
            "requested_gender_filter": gender,
            "normalized_gender_filter": gender_filter,
            "preferred_styles": preferred_styles,
            "disliked_styles": disliked_styles,
            "matched_preferred_labels": matched_preferred_labels,
            "matched_disliked_labels": matched_disliked_labels,
            "preprocess": preprocess_info,
            "preference_bias": preference_debug,
            "faiss_search_k": int(faiss_search_k),
            "strict_filtered_count": len(strict_results),
            "fallback_candidate_count": len(fallback_results),
            "fallback_used": bool(fallback_used),
            "returned_count": len(results),
            "results": results,
        }


def _parse_args():
    parser = argparse.ArgumentParser(description="Fashion similarity search with YOLO preprocessing + Faiss + PostgreSQL")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--label-map", type=str, required=True)
    parser.add_argument("--faiss-index", type=str, required=True)
    parser.add_argument("--postgres-dsn", type=str, required=True)
    parser.add_argument("--query-image", type=str, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--gender", type=str, default="all")
    parser.add_argument("--preferred-styles", type=str, default="")
    parser.add_argument("--disliked-styles", type=str, default="")
    parser.add_argument("--label-centroids", type=str, default=None)
    parser.add_argument("--preferred-weight", type=float, default=0.15)
    parser.add_argument("--disliked-weight", type=float, default=0.15)
    parser.add_argument("--yolo-model", type=str, default="yolov8s.pt")
    parser.add_argument("--yolo-conf", type=float, default=0.25)
    parser.add_argument("--search-pool-factor", type=int, default=5)
    parser.add_argument("--max-search-pool", type=int, default=200)
    parser.add_argument("--fallback-fill", action="store_true")
    parser.add_argument("--output-json", type=str, default=None)
    return parser.parse_args()


def main():
    args = _parse_args()
    engine = FashionSearchEngine(
        checkpoint_path=Path(args.checkpoint),
        label_map_path=Path(args.label_map),
        faiss_index_path=Path(args.faiss_index),
        postgres_dsn=args.postgres_dsn,
        yolo_model=args.yolo_model,
        yolo_conf=args.yolo_conf,
        label_centroids_path=Path(args.label_centroids) if args.label_centroids else None,
        preferred_weight=args.preferred_weight,
        disliked_weight=args.disliked_weight,
        search_pool_factor=args.search_pool_factor,
        max_search_pool=args.max_search_pool,
    )

    image = Image.open(args.query_image).convert("RGB")
    results = engine.search(
        image=image,
        top_k=args.top_k,
        gender=args.gender,
        preferred_styles=engine.parse_csv_list(args.preferred_styles),
        disliked_styles=engine.parse_csv_list(args.disliked_styles),
        fallback_fill=args.fallback_fill,
    )

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[DONE] search results saved to {out_path.resolve()}")
    else:
        print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
