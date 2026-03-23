# Fashion Recommendation System

ResNet50 backbone + Faiss + PostgreSQL 기반의 **패션 추천 시스템**입니다.  
사용자가 이미지를 업로드하면, YOLO 기반 전처리를 통해 사람 영역을 추출한 뒤 feature vector를 생성하고,  
Faiss에서 유사한 패션 이미지를 검색하여 추천 결과를 반환합니다.

---

## 프로젝트 개요

이 프로젝트는 다음 흐름으로 동작합니다.

1. AI Hub 패션 데이터셋 다운로드
2. YOLO 기반 사람 영역 전처리
3. ResNet50 학습
4. 학습된 모델의 마지막 FC layer 제거 후 backbone으로 사용
5. backbone feature를 Faiss + PostgreSQL에 저장
6. 사용자 입력 이미지에 대해 유사 패션 검색
7. FastAPI 서버를 통해 검색 API 제공

---

## 디렉토리 구조 예시

```bash
.
├── app.py
├── .env
├── artifacts/
│   ├── fashion_index.faiss
│   └── label_centroids.npz
├── outputs_resnet50/
│   ├── best_model_state.pth
│   ├── label_to_index.json
│   ├── val_manifest.csv
│   └── test_manifest.csv
├── sample_image/
├── src/
│   ├── preprocess.py
│   ├── resnet_model.py
│   ├── build_index.py
│   ├── search.py
│   ├── person_crop.py
│   └── schema_postgres.sql
└── 패션 데이터셋/
```

---

## 1. 데이터셋 다운로드

AI Hub의 **연도별 패션 선호도 파악 및 추천 데이터**를 사용합니다.  
다운로드 용량은 약 **330GB 이상**입니다.

- 데이터셋 다운로드 URL:  
  https://www.aihub.or.kr/aihubdata/data/view.do?pageIndex=2&currMenu=115&topMenu=100&srchOptnCnd=OPTNCND003&searchKeyword=%ED%8C%A8%EC%85%98&srchDetailCnd=DETAILCND001&srchOrder=ORDER001&srchPagePer=20&aihubDataSe=data&dataSetSn=71446


데이터를 다운로드한 뒤, `.zip` 파일들을 모두 압축 해제하고  
프로젝트 루트에 아래와 같이 `패션 데이터셋` 디렉토리를 위치시켜 주세요.

```bash
./패션 데이터셋
```

---

## 2. 전처리 파일 / 모델 weight / 학습 결과 파일 다운로드 (선택)

추론만 빠르게 실행하고 싶다면 아래 파일들을 별도로 다운로드해서 사용할 수 있습니다.

- 전처리 결과 (`./패션 데이터/preprocess/` 형태로 저장)
- `artifacts/`
- `outputs_resnet50/`

다운로드 링크:  
Google Drive 폴더
- 다운로드 URL:  
  https://drive.google.com/drive/folders/1dFYvq9yQkJ1Hs97dU7XDony6ZcK7C2-W?usp=sharing

---

## 3. 전처리

YOLO를 이용해 사람 영역만 crop한 뒤, `224 x 448` 크기로 맞춰 저장합니다.

```bash
python src/preprocess.py \
    --dataset-root "./패션 데이터셋" \
    --batch-size 128
```

---

## 4. 모델 학습 (ResNet50)

전처리된 데이터셋을 이용해 ResNet50 기반 분류 모델을 학습합니다.

```bash
python src/resnet_model.py \
  --train-dir "./패션 데이터셋/preprocess/train" \
  --val-dir "./패션 데이터셋/preprocess/valid" \
  --test-dir "./패션 데이터셋/preprocess/test" \
  --image-size 224 \
  --batch-size 32 \
  --num-epochs 30 \
  --patience 5 \
  --lr 1e-4 \
  --num-workers 4 \
  --output-dir "./outputs_resnet50"
```

학습이 완료되면 다음과 같은 파일들이 생성됩니다.

- `best_model_state.pth`
- `label_to_index.json`
- `val_manifest.csv`
- `test_manifest.csv`

---

## 5. PostgreSQL 설정

### 5-1. Docker로 PostgreSQL 실행

```bash
docker pull postgres:18

docker run --name fashion-postgres \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=fashion \
  -p 5432:5432 \
  -d postgres:18
```


### 5-2. PostgreSQL 접속

```bash
docker exec -it fashion-postgres psql -U postgres -d fashion
```

### 5-3. 테이블 / 인덱스 생성

`src/schema_postgres.sql`을 이용해 테이블과 인덱스를 생성합니다.

```bash
docker exec -i fashion-postgres psql -U postgres -d fashion < src/schema_postgres.sql
```


### 5-4. 컨테이너 정지/시작

```bash
docker stop fashion-postgres

docker start fashion-postgres
```


---

## 6. 벡터 인덱스 및 DB 구축

`valid`, `test` 데이터셋을 기반으로 Faiss 인덱스와 PostgreSQL metadata를 생성합니다.

```bash
python src/build_index.py \
  --checkpoint "outputs_resnet50/best_model_state.pth" \
  --label-map "outputs_resnet50/label_to_index.json" \
  --valid-manifest "outputs_resnet50/val_manifest.csv" \
  --test-manifest "outputs_resnet50/test_manifest.csv" \
  --training-original-root "./패션 데이터셋/Training/01.원천데이터" \
  --validation-original-root "./패션 데이터셋/Validation/01.원천데이터" \
  --faiss-index-out "artifacts/fashion_index.faiss" \
  --postgres-dsn "postgresql://postgres:postgres@localhost:5432/fashion"
```

이 단계가 끝나면:

- `artifacts/fashion_index.faiss`
- `artifacts/label_centroids.npz`
- PostgreSQL `fashion_items` 테이블

이 준비됩니다.

---

## 7. 단일 이미지 검색

CLI 환경에서 직접 검색을 테스트할 수 있습니다.

```bash
python src/search_local.py \
  --checkpoint "outputs_resnet50/best_model_state.pth" \
  --label-map "outputs_resnet50/label_to_index.json" \
  --faiss-index "artifacts/fashion_index.faiss" \
  --postgres-dsn "postgresql://postgres:postgres@localhost:5432/fashion" \
  --query-image "sample_image/T_00006_50_ivy_M.jpg" \
  --top-k 10 \
  --gender "woman" \
  --preferred-styles "street,casual" \
  --disliked-styles "formal" \
  --label-centroids "artifacts/label_centroids.npz" \
  --fallback-fill
```

### 지원 옵션

- `--top-k`  
  최종 반환할 추천 이미지 개수

- `--gender`  
  `all`, `man`, `woman`

- `--preferred-styles`  
  선호 스타일 목록 (쉼표 구분)

- `--disliked-styles`  
  비선호 스타일 목록 (쉼표 구분)

- `--label-centroids`  
  스타일별 centroid 파일 경로

- `--fallback-fill`  
  필터링 후 결과가 부족할 때 fallback 허용

---

## 8. FastAPI 서버 실행

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

실행 후 Swagger 문서는 아래 주소에서 확인할 수 있습니다.

```text
http://localhost:8000/docs
```

---

## 9. 환경 변수 설정 (`.env`)

기본값이 코드에 포함되어 있지만, 필요하다면 `.env` 파일로 설정을 덮어쓸 수 있습니다.

```env
CHECKPOINT_PATH=outputs_resnet50/best_model_state.pth
LABEL_MAP_PATH=outputs_resnet50/label_to_index.json
FAISS_INDEX_PATH=artifacts/fashion_index.faiss
LABEL_CENTROIDS_PATH=artifacts/label_centroids.npz
POSTGRES_DSN=postgresql://postgres:postgres@localhost:5432/fashion
YOLO_MODEL_PATH=yolov8s.pt
DEFAULT_TOP_K=5
DEFAULT_SEARCH_POOL_FACTOR=5
PREFERRED_WEIGHT=0.15
DISLIKED_WEIGHT=0.15
FALLBACK_FILL=true
```

---

## 10. 검색 API 개요

### `POST /search`

사용자 이미지를 업로드하면 유사한 패션 이미지를 검색합니다.

#### 입력값

- `image`: 업로드 이미지
- `gender`: `all`, `man`, `woman`
- `top_k`: 반환 개수
- `preferred_styles`: 선호 스타일 목록
- `disliked_styles`: 비선호 스타일 목록

#### 반환값 예시

- 전처리 정보
- 사용된 스타일 bias 정보
- Faiss 검색 개수
- 최종 반환 결과 수
- 추천 이미지 목록
- 각 결과의 score, style, gender, original_path 등

---

## 11. 주의사항

- 데이터셋 용량이 매우 큽니다. 저장 공간을 충분히 확보하세요.
- PostgreSQL은 로컬 또는 Docker 환경에서 실행되어 있어야 합니다.
- `build_index.py` 실행 전 반드시 PostgreSQL 테이블을 생성해야 합니다.
- FastAPI 외부 공개 시 포트포워딩 및 방화벽 설정이 필요합니다.
- 추천 품질을 위해 query 이미지에도 YOLO 기반 전처리가 적용됩니다.


---

## 12. 실행 순서 요약

### 전체 파이프라인
```bash
# 1. 전처리
python src/preprocess.py --dataset-root "./패션 데이터셋" --batch-size 128

# 2. 학습
python src/resnet_model.py ...

# 3. PostgreSQL 실행
docker run --name fashion-postgres ...

# 4. 테이블 생성
docker exec -i fashion-postgres psql -U postgres -d fashion < src/schema_postgres.sql

# 5. 인덱스 구축
python src/build_index.py ...

# 6. 검색 테스트
python src/search_local.py ...

# 7. 서버 실행
uvicorn app:app --host 0.0.0.0 --port 8000

# 8. streamlit 실행
python -m streamlit run src/frontend/app_api_ver.py --server.port 8501 --server.address 0.0.0.0

```


---

## 13. 향후 개선 방향

- 선호/비선호 스타일 weighting 고도화
- 스타일 centroid를 활용한 re-ranking 강화
- 인증이 포함된 API 서버 배포
- 프론트엔드 UI 연동
- 대규모 벡터 검색을 위한 구조 확장

---
<!-- 
## 14. 기술 스택

- **PyTorch**
- **ResNet50**
- **YOLO**
- **Faiss**
- **PostgreSQL**
- **FastAPI**
- **Docker** -->

