from __future__ import annotations

from io import BytesIO
from pathlib import Path
from PIL import Image
import os
from functools import lru_cache
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict
from starlette.concurrency import run_in_threadpool

from src.search import SearchEngine, SearchRequest, SearchResponse, normalize_gender_filter


class Settings(BaseSettings):
    checkpoint_path: str = "outputs_resnet50/best_model_state.pth"
    label_map_path: str = "outputs_resnet50/label_to_index.json"
    faiss_index_path: str = "artifacts/fashion_index.faiss"
    label_centroids_path: Optional[str] = "artifacts/label_centroids.npz"
    postgres_dsn: str = "postgresql://postgres:postgres@localhost:5432/fashion"
    yolo_model_path: str = "yolov8s.pt"

    default_top_k: int = 5
    default_search_pool_factor: int = 5
    preferred_weight: float = 0.15
    disliked_weight: float = 0.15
    fallback_fill: bool = True

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


# env setting
@lru_cache
def get_settings() -> Settings:
    try:
        return Settings()
    except ValidationError as e:
        raise RuntimeError(
            "Failed to load configuration. Check your .env values and types."
        ) from e


app = FastAPI(
    title="Fashion Retrieval API",
    version="1.1.0",
    summary="YOLO + ResNet50 + Faiss 기반 패션 유사 이미지 검색 API",
    description=(
        "업로드된 이미지에 대해 YOLO로 사람 영역을 crop하고, ResNet50 backbone으로 임베딩을 추출한 뒤, "
        "Faiss에서 유사 이미지를 찾는 API입니다. PostgreSQL에서 원본 이미지 경로와 메타데이터를 조회합니다.\n\n"
        "주요 기능:\n"
        "- YOLO 기반 사람 영역 전처리\n"
        "- 성별 필터(all/man/woman)\n"
        "- top-k 검색\n"
        "- preferred/disliked styles 반영\n"
        "- 결과 부족 시 fallback 채우기 옵션"
    ),
    contact={"name": "Fashion Retrieval API"},
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_tags=[
        {"name": "health", "description": "서버 상태 및 현재 설정 확인"},
        {"name": "search", "description": "이미지 업로드 기반 유사 패션 검색"},
    ],
)



@app.on_event("startup")
def startup_event() -> None:
    settings = get_settings()
    app.state.settings = settings
    app.state.search_engine = SearchEngine(
        checkpoint_path=Path(settings.checkpoint_path),
        label_map_path=Path(settings.label_map_path),
        faiss_index_path=Path(settings.faiss_index_path),
        postgres_dsn=settings.postgres_dsn,
        yolo_model=settings.yolo_model_path,
        label_centroids_path=Path(settings.label_centroids_path) if settings.label_centroids_path else None,
        preferred_weight=settings.preferred_weight,
        disliked_weight=settings.disliked_weight,
        search_pool_factor=settings.default_search_pool_factor,
    )


# 서버 status, 관련 경로 체크 (서비스 떄는 status빼고는 없애야 할 듯)
@app.get(
    "/health",
    tags=["health"],
    summary="check server status",
    description="현재 서버가 정상 구동 중인지와 로드된 기본 설정값을 반환합니다.",
)
def health() -> dict:
    settings = app.state.settings
    return {
        "status": "ok",
        "checkpoint_path": settings.checkpoint_path,
        "label_map_path": settings.label_map_path,
        "faiss_index_path": settings.faiss_index_path,
        "label_centroids_path": settings.label_centroids_path,
        "postgres_dsn": settings.postgres_dsn,
        "yolo_model_path": settings.yolo_model_path,
        "default_top_k": settings.default_top_k,
        "default_search_pool_factor": settings.default_search_pool_factor,
        "preferred_weight": settings.preferred_weight,
        "disliked_weight": settings.disliked_weight,
        "fallback_fill": settings.fallback_fill,
        "env_file_exists": os.path.exists(".env"),
    }




@app.post(
    "/search",
    tags=["search"],
    response_model=SearchResponse,
    summary="유사 패션 이미지 검색",
    description=(
        "이미지 파일과 검색 옵션을 받아 유사한 패션 이미지를 반환합니다. "
        "입력 이미지는 YOLO 사람 crop 전처리를 거친 뒤 검색됩니다."
    ),
    responses={
        200: {
            "description": "검색 성공",
            "content": {
                "application/json": {
                    "example": {
                        "query_image": "example.jpg",
                        "top_k": 5,
                        "requested_gender_filter": "woman",
                        "normalized_gender_filter": "female",
                        "preferred_styles": ["street", "casual"],
                        "disliked_styles": ["formal"],
                        "results": [
                            {
                                "rank": 1,
                                "score": 0.9123,
                                "item_id": 1034,
                                "split": "test",
                                "source_root_name": "Validation",
                                "original_path": ".../W_12345_21_street_W.jpg",
                                "style": "street",
                                "gender": "female",
                                "label": "street_female",
                            }
                        ],
                    }
                }
            },
        },
        400: {"description": "잘못된 요청 또는 이미지 형식 오류"},
        500: {"description": "모델/인덱스/DB 로드 실패 또는 검색 내부 오류"},
    },
)
async def search(
    image: UploadFile = File(
        ...,
        description="검색할 이미지 파일. 임의 크기 가능. 서버에서 YOLO 전처리 후 검색합니다.",
    ),
    gender: str = Form(
        "all",
        description="성별 필터. all, man, woman, male, female 중 하나.",
        examples=["all"],
    ),
    top_k: Optional[int] = Form(
        None,
        description="반환할 최종 결과 개수. 미입력 시 서버 기본값 사용.",
        examples=[5],
    ),
    preferred_styles: Optional[str] = Form(
        None,
        description="선호 스타일 목록. 쉼표로 구분합니다. 예: street,casual,sporty",
        examples=["street,casual"],
    ),
    disliked_styles: Optional[str] = Form(
        None,
        description="비선호 스타일 목록. 쉼표로 구분합니다. 예: formal,classic",
        examples=["formal"],
    ),
    fallback_fill: Optional[bool] = Form(
        None,
        description="성별 필터 등으로 결과가 부족할 때 fallback 결과로 채울지 여부.",
        examples=[True],
    ),
) -> SearchResponse:
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    settings = app.state.settings
    engine: SearchEngine = app.state.search_engine

    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded image is empty.")

    try:
        pil_image = Image.open(BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to decode image: {e}") from e

    preferred_list = engine.parse_csv_list(preferred_styles)
    disliked_list = engine.parse_csv_list(disliked_styles)

    try:
        result = await run_in_threadpool(
            engine.search,
            pil_image,
            top_k if top_k is not None else settings.default_top_k,
            gender,
            preferred_list,
            disliked_list,
            settings.fallback_fill if fallback_fill is None else fallback_fill,
        )
        return SearchResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {e}") from e



if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
