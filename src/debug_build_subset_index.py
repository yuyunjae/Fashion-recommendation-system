#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import faiss
import numpy as np
import psycopg


VALID_GENDERS = {"male", "female"}


def fetch_item_ids(postgres_dsn: str, gender: str) -> list[int]:
    with psycopg.connect(postgres_dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT item_id
                FROM fashion_items
                WHERE gender = %s
                ORDER BY item_id
                """,
                (gender,),
            )
            return [int(row[0]) for row in cur.fetchall()]


def reconstruct_vectors(index, target_ids):
    # IndexIDMap / IndexIDMap2 에서 외부 ID 목록 추출
    id_map = faiss.vector_to_array(index.id_map)

    # external item_id -> internal position
    id_to_pos = {int(ext_id): pos for pos, ext_id in enumerate(id_map)}

    vectors = []
    valid_ids = []

    base_index = index.index  # 내부 실제 IndexFlatIP

    for item_id in target_ids:
        pos = id_to_pos.get(int(item_id))
        if pos is None:
            continue

        try:
            vec = base_index.reconstruct(pos)
            vectors.append(vec)
            valid_ids.append(int(item_id))
        except Exception:
            continue

    if not vectors:
        raise RuntimeError("No vectors could be reconstructed from the source index.")

    vectors = np.stack(vectors).astype("float32")
    valid_ids = np.array(valid_ids, dtype=np.int64)
    return vectors, valid_ids


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a gender-specific Faiss subset index from the full fashion index.")
    parser.add_argument("--faiss-index", type=Path, required=True, help="Path to the full Faiss index.")
    parser.add_argument("--postgres-dsn", type=str, required=True, help="PostgreSQL DSN for fashion_items.")
    parser.add_argument("--gender", type=str, required=True, choices=sorted(VALID_GENDERS), help="Subset gender to export.")
    parser.add_argument("--out-index", type=Path, required=True, help="Output path for the subset Faiss index.")
    args = parser.parse_args()

    print(f"[INFO] Loading full index: {args.faiss_index}")
    index = faiss.read_index(str(args.faiss_index))

    print(f"[INFO] Fetching item_ids for gender={args.gender!r} from PostgreSQL")
    item_ids = fetch_item_ids(args.postgres_dsn, args.gender)
    print(f"[INFO] Retrieved {len(item_ids)} ids from fashion_items")

    vectors, valid_ids = reconstruct_vectors(index, item_ids)
    print(f"[INFO] Reconstructed {len(valid_ids)} vectors from source index")

    dim = int(vectors.shape[1])
    subset_index = faiss.IndexIDMap(faiss.IndexFlatIP(dim))
    subset_index.add_with_ids(vectors, valid_ids)

    args.out_index.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(subset_index, str(args.out_index))
    print(f"[DONE] Wrote subset index to: {args.out_index}")
    print(f"- dim: {dim}")
    print(f"- ntotal: {subset_index.ntotal}")


if __name__ == "__main__":
    main()
