#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import faiss
import matplotlib.pyplot as plt
import numpy as np
import psycopg
from sklearn.decomposition import PCA


def plot_label_centroids(centroid_path: Path, out_path: Path | None) -> None:
    data = np.load(centroid_path, allow_pickle=True)
    labels = data["labels"].tolist()
    centroids = np.asarray(data["centroids"], dtype="float32")

    pca = PCA(n_components=2)
    xy = pca.fit_transform(centroids)

    plt.figure(figsize=(14, 11))
    for i, label in enumerate(labels):
        x, y = xy[i]
        color = "tab:red" if label.endswith("_female") else "tab:blue"
        plt.scatter(x, y, c=color, s=35)
        plt.text(x, y, label, fontsize=8)
    plt.title("Label centroid PCA (blue=male, red=female)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()

    if out_path is None:
        plt.show()
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=180)
        print(f"[DONE] Saved centroid PCA to: {out_path}")


def plot_item_embeddings(index_path: Path, postgres_dsn: str, sample_per_gender: int, out_path: Path | None) -> None:
    index = faiss.read_index(str(index_path))

    with psycopg.connect(postgres_dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                (
                    SELECT item_id, gender
                    FROM fashion_items
                    WHERE gender = 'male'
                    ORDER BY random()
                    LIMIT %s
                )
                UNION ALL
                (
                    SELECT item_id, gender
                    FROM fashion_items
                    WHERE gender = 'female'
                    ORDER BY random()
                    LIMIT %s
                )
                """,
                (sample_per_gender, sample_per_gender),
            )
            rows = cur.fetchall()

    vectors = []
    colors = []
    for item_id, gender in rows:
        try:
            vec = index.reconstruct(int(item_id))
        except RuntimeError:
            continue
        vectors.append(np.asarray(vec, dtype="float32"))
        colors.append("tab:blue" if gender == "male" else "tab:red")

    if not vectors:
        raise RuntimeError("No item vectors reconstructed; cannot plot item embeddings.")

    vectors_np = np.stack(vectors)
    xy = PCA(n_components=2).fit_transform(vectors_np)

    plt.figure(figsize=(12, 10))
    plt.scatter(xy[:, 0], xy[:, 1], c=colors, s=10, alpha=0.65)
    plt.title(f"Item embedding PCA (sample_per_gender={sample_per_gender})")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()

    if out_path is None:
        plt.show()
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=180)
        print(f"[DONE] Saved item PCA to: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize label centroid and item embedding distributions.")
    parser.add_argument("--centroids", type=Path, default=None, help="Path to label_centroids.npz")
    parser.add_argument("--faiss-index", type=Path, default=None, help="Path to full Faiss index")
    parser.add_argument("--postgres-dsn", type=str, default=None, help="PostgreSQL DSN; required for item embedding plot")
    parser.add_argument("--sample-per-gender", type=int, default=1000)
    parser.add_argument("--centroid-out", type=Path, default=None)
    parser.add_argument("--item-out", type=Path, default=None)
    args = parser.parse_args()

    if args.centroids is not None:
        plot_label_centroids(args.centroids, args.centroid_out)

    if args.faiss_index is not None:
        if args.postgres_dsn is None:
            raise ValueError("--postgres-dsn is required when --faiss-index is provided")
        plot_item_embeddings(args.faiss_index, args.postgres_dsn, args.sample_per_gender, args.item_out)


if __name__ == "__main__":
    main()
