#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

from PIL import Image

from src.search import SearchEngine


VALID_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def collect_query_paths(query_dir: Path, limit: int | None) -> list[Path]:
    paths = sorted([p for p in query_dir.rglob("*") if p.suffix.lower() in VALID_EXTS])
    if limit is not None:
        paths = paths[:limit]
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multiple query images and summarize gender bias in top-k search results.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--label-map", type=Path, required=True)
    parser.add_argument("--faiss-index", type=Path, required=True)
    parser.add_argument("--postgres-dsn", type=str, required=True)
    parser.add_argument("--query-dir", type=Path, required=True, help="Directory containing query images of a known gender.")
    parser.add_argument("--query-gender", type=str, required=True, choices=["male", "female"], help="Ground-truth gender for all images in --query-dir.")
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--limit", type=int, default=20, help="How many query images to evaluate.")
    parser.add_argument("--label-centroids", type=Path, default=None)
    parser.add_argument("--yolo-model", type=str, default="yolov8s.pt")
    parser.add_argument("--out-json", type=Path, default=None)
    args = parser.parse_args()

    query_paths = collect_query_paths(args.query_dir, args.limit)
    if not query_paths:
        raise FileNotFoundError(f"No query images found under: {args.query_dir}")

    engine = SearchEngine(
        checkpoint_path=args.checkpoint,
        label_map_path=args.label_map,
        faiss_index_path=args.faiss_index,
        postgres_dsn=args.postgres_dsn,
        yolo_model=args.yolo_model,
        label_centroids_path=args.label_centroids,
    )

    all_gender_counts: Counter[str] = Counter()
    same_gender_ratios: list[float] = []
    top1_same_gender_flags: list[int] = []
    per_query: list[dict[str, Any]] = []

    for idx, path in enumerate(query_paths, start=1):
        image = Image.open(path).convert("RGB")
        result = engine.search(
            image=image,
            top_k=args.top_k,
            gender="all",
            preferred_styles=None,
            disliked_styles=None,
            fallback_fill=False,
        )
        genders = [r["gender"] for r in result["results"]]
        scores = [float(r["score"]) for r in result["results"]]
        counter = Counter(genders)
        all_gender_counts.update(counter)

        same_count = counter.get(args.query_gender, 0)
        ratio = same_count / max(len(genders), 1)
        same_gender_ratios.append(ratio)
        top1_same_gender_flags.append(int(bool(genders) and genders[0] == args.query_gender))

        per_query.append(
            {
                "query_path": str(path),
                "ground_truth_gender": args.query_gender,
                "returned_count": len(genders),
                "gender_counts": dict(counter),
                "same_gender_ratio": ratio,
                "top1_gender": genders[0] if genders else None,
                "top1_score": scores[0] if scores else None,
                "top5": result["results"][:5],
            }
        )
        print(
            f"[{idx:03d}/{len(query_paths):03d}] {path.name} | "
            f"top1={per_query[-1]['top1_gender']} | same_gender_ratio={ratio:.3f} | counts={dict(counter)}"
        )

    summary = {
        "query_dir": str(args.query_dir),
        "query_gender": args.query_gender,
        "num_queries": len(query_paths),
        "top_k": args.top_k,
        "aggregate_result_gender_counts": dict(all_gender_counts),
        "mean_same_gender_ratio": mean(same_gender_ratios) if same_gender_ratios else None,
        "top1_same_gender_rate": mean(top1_same_gender_flags) if top1_same_gender_flags else None,
        "per_query": per_query,
    }

    print("\n=== SUMMARY ===")
    print(json.dumps({k: v for k, v in summary.items() if k != "per_query"}, ensure_ascii=False, indent=2))

    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"[DONE] Wrote report to: {args.out_json}")


if __name__ == "__main__":
    main()
