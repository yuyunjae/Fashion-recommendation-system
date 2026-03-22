import io
import os
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from starlette.concurrency import run_in_threadpool

from search import FashionSearchEngine

APP_TITLE = "Fashion Retrieval API"


def parse_csv_list(value: Optional[str]) -> List[str]:
    if not value:
        return []
    return [v.strip() for v in value.split(",") if v.strip()]


CHECKPOINT = Path(os.getenv("CHECKPOINT_PATH", "outputs_resnet50/best_model_state.pth"))
LABEL_MAP = Path(os.getenv("LABEL_MAP_PATH", "outputs_resnet50/label_to_index.json"))
FAISS_INDEX = Path(os.getenv("FAISS_INDEX_PATH", "artifacts/fashion_index.faiss"))
LABEL_CENTROIDS = os.getenv("LABEL_CENTROIDS_PATH", "")
POSTGRES_DSN = os.getenv("POSTGRES_DSN", "postgresql://postgres:postgres@localhost:5432/fashion")
YOLO_MODEL = os.getenv("YOLO_MODEL", "yolov8s.pt")
YOLO_CONF = float(os.getenv("YOLO_CONF", "0.25"))
PREFERRED_WEIGHT = float(os.getenv("PREFERRED_WEIGHT", "0.15"))
DISLIKED_WEIGHT = float(os.getenv("DISLIKED_WEIGHT", "0.15"))
SEARCH_POOL_FACTOR = int(os.getenv("SEARCH_POOL_FACTOR", "5"))
MAX_SEARCH_POOL = int(os.getenv("MAX_SEARCH_POOL", "200"))

app = FastAPI(title=APP_TITLE)
engine: Optional[FashionSearchEngine] = None


@app.on_event("startup")
def startup_event() -> None:
    global engine
    engine = FashionSearchEngine(
        checkpoint_path=CHECKPOINT,
        label_map_path=LABEL_MAP,
        faiss_index_path=FAISS_INDEX,
        postgres_dsn=POSTGRES_DSN,
        yolo_model=YOLO_MODEL,
        yolo_conf=YOLO_CONF,
        label_centroids_path=Path(LABEL_CENTROIDS) if LABEL_CENTROIDS else None,
        preferred_weight=PREFERRED_WEIGHT,
        disliked_weight=DISLIKED_WEIGHT,
        search_pool_factor=SEARCH_POOL_FACTOR,
        max_search_pool=MAX_SEARCH_POOL,
    )


@app.get("/health")
def health():
    return {"status": "ok", "engine_loaded": engine is not None}


@app.post("/search")
async def search_fashion(
    image: UploadFile = File(...),
    gender: str = Form("all"),
    top_k: int = Form(5),
    preferred_styles: str = Form(""),
    disliked_styles: str = Form(""),
    fallback_fill: bool = Form(True),
):
    if engine is None:
        raise HTTPException(status_code=503, detail="Search engine is not initialized")
    if top_k <= 0:
        raise HTTPException(status_code=400, detail="top_k must be >= 1")

    raw = await image.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty image upload")

    try:
        pil_image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {e}") from e

    try:
        result = await run_in_threadpool(
            engine.search,
            pil_image,
            top_k,
            gender,
            parse_csv_list(preferred_styles),
            parse_csv_list(disliked_styles),
            fallback_fill,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {e}") from e

    return JSONResponse(result)
