# 데이터셋 다운로드
AI Hub의 연도별 패션 선호도 파악 및 추천 데이터 (>= 330GB(다운로드 기준))
![dataset 다운로드](https://www.aihub.or.kr/aihubdata/data/view.do?pageIndex=2&currMenu=115&topMenu=100&srchOptnCnd=OPTNCND003&searchKeyword=%ED%8C%A8%EC%85%98&srchDetailCnd=DETAILCND001&srchOrder=ORDER001&srchPagePer=20&aihubDataSe=data&dataSetSn=71446)

'패션 데이터셋' 디렉토리 내부 .zip 파일들을 압축 해제 후, 프로젝트 루트에 '패션 데이터셋' 디렉토리를 위치시키세요. 

# 전처리 파일 실행
python src/preprocess.py \
    --dataset-root "./패션 데이터셋" \
    --batch-size 128

# 모델 학습 (resnet50)
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

# PostgreSQL 기초 설정 
## (docker로 다운로드 및 run)
docker pull postgres:18
docker run --name fashion-postgres \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=fashion \
  -p 5432:5432 \
  -d postgres:18

## 컨테이너 시작 (껐다 키거나 했을 떄)
docker start fashion-postgres

## 터미널로 연결
docker exec -it fashion-postgres psql -U postgres -d fashion

## src/schema_postgres 에 있는 스키마를 이용해 table, index생성
docker exec -i fashion-postgres psql -U postgres -d fashion < src/schema_postgres.sql

# build_index.py 실행
time python src/build_index.py \
  --checkpoint "outputs_resnet50/best_model_state.pth" \
  --label-map "outputs_resnet50/label_to_index.json" \
  --valid-manifest "outputs_resnet50/val_manifest.csv" \
  --test-manifest "outputs_resnet50/test_manifest.csv" \
  --training-original-root "./패션 데이터셋/Training/01.원천데이터" \
  --validation-original-root "./패션 데이터셋/Validation/01.원천데이터" \
  --faiss-index-out "artifacts/fashion_index.faiss" \
  --postgres-dsn "postgresql://postgres:postgres@localhost:5432/fashion"