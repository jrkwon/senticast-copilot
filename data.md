# 데이터 설명 (Data Description)

## 개요

SentiCast 시스템은 세 가지 광물(금, 은, 구리)의 가격 예측을 위해 두 종류의 데이터를 사용합니다.
실제 데이터는 `src/data/dataset/` 디렉토리에 있으며 `dataset.md`에 자세한 설명이 있습니다.

---

## 1. 광물 가격 시계열 데이터 (`trading-data-{mineral}.csv`)

### 파일 위치
```
src/data/dataset/trading-data-gold.csv
src/data/dataset/trading-data-silver.csv
src/data/dataset/trading-data-copper.csv
```

### 형식

| 컬럼명   | 타입   | 설명             |
|---------|--------|-----------------|
| `time`  | string | 날짜 (YYYY-MM-DD) |
| `open`  | float  | 시가              |
| `high`  | float  | 고가              |
| `low`   | float  | 저가              |
| `close` | float  | **종가** (모델 입력에 사용) |
| `EMA`   | float  | 지수이동평균      |
| `Volume`| int    | 거래량            |

### 가격 단위 및 범위

| 광물   | 단위         | 대략적 범위       | 특성 |
|--------|-------------|-----------------|------|
| 금     | USD/troy oz | ~1,000 – 5,400  | 안전자산, 낮은 변동성 |
| 은     | USD/troy oz | ~12 – 116       | 귀금속 + 산업재, 중간 변동성 |
| 구리   | USD/lb      | ~1.9 – 6.2      | 순수 산업재, 경기 선행지표 |

### 데이터 범위
- **금**: 2013-10-09 ~ 2026-02-26 (약 3,116 거래일)
- **은/구리**: 2013-12-05 ~ 2026-02-26 (약 3,076 거래일)
- 세 광물이 공통으로 존재하는 날짜: **3,076일** (2013-12-05 ~ 2026-02-26)

### 상호 연관성

```
금 ↔ 은:   높은 양의 상관관계 (둘 다 귀금속)
은 ↔ 구리:  중간 양의 상관관계 (둘 다 산업재)
금 ↔ 구리:  낮은 상관관계 (금=안전자산, 구리=경기민감)
```

---

## 2. 뉴스 요약 데이터 (`news-summary-{mineral}.csv`)

### 파일 위치
```
src/data/dataset/news-summary-gold.csv
src/data/dataset/news-summary-silver.csv
src/data/dataset/news-summary-copper.csv
```

### 형식

| 컬럼명       | 타입   | 설명                                         |
|------------|--------|---------------------------------------------|
| `text`     | string | 6개월 기간의 뉴스 요약 텍스트                  |
| `start_date` | string | 요약 시작일 (YYYY-MM-DD)                     |
| `end_date`   | string | 요약 종료일 (YYYY-MM-DD, ≈ start_date + 6개월) |

### 뉴스 텍스트 구조

**은/구리**: 예측 지평별 섹션이 명시적으로 구분됨
```
### Summary
- Short-term:  [단기 전망 - 30일 예측에 사용]
- Medium-term: [중기 전망 - 90일 예측에 사용]
- Long-term:   [장기 전망 - 180일 예측에 사용]

{"signals": [...]}
```

**금**: Overview 형식 (Short/Medium/Long 구분 없음 → 세 지평에 동일하게 사용)
```
### Summary
- Overview: [전반적인 전망 - 30/90/180일 예측 모두에 사용]

{"signals": [...]}
```

### 데이터 범위
- **금/은/구리**: 2017년 말 ~ 2025년 중순 (약 1,933–1,939 행)
- 각 행은 약 6개월 기간의 뉴스를 요약
- 행 간 겹침 있음 (sliding window 방식으로 생성됨)

### 기준일 T에 대한 뉴스 조회 방법
- `start_date <= T` 조건을 만족하는 행 중 `start_date`가 가장 최근인 행을 사용
- T < 2017-10-04 인 경우: 해당 광물의 뉴스 없음 → 영벡터 임베딩 사용

---

## 3. 입력/출력 형식

### 입력 (기준 날짜 T 기준)

```
price_series : [T-179 : T]  → shape (180, 3)  # 180일 × 3개 광물 (정규화된 종가)
news_embeds  : 기준일 T에 해당하는 최신 뉴스 → shape (3, 3, embed_dim)
                # 3개 광물 × 3개 지평(short/medium/long) × embed_dim
```

### 출력

```
predictions : 30일, 90일, 180일 후 가격 → shape (3, 3)  # 3개 지평 × 3개 광물
intervals   : 예측 신뢰구간 (90% CI)    → shape (3, 3, 2)  # lower/upper bounds
```

---

## 4. 데이터 전처리

### 정규화
- 각 광물별 독립적으로 z-score 정규화 (`config.yaml`에서 `minmax`로 변경 가능)
- **학습 데이터 통계만** 사용 (시험 데이터 사전 관찰 방지)
- 가격 범위가 광물별로 매우 다른 문제 (금 ~3000, 은 ~30, 구리 ~4) 해소

### 뉴스 임베딩
- **1순위**: `sentence-transformers` (all-MiniLM-L6-v2, 384차원)
- **폴백**: TF-IDF + Truncated SVD (scikit-learn, `news_embed_dim` 차원)
- 임베딩 결과는 `data/cache/news_tensor.npy`에 캐시됨 (재실행 시 재계산 불필요)

### 결측치 처리
- 주말/공휴일: 선형 보간

---

## 5. 데이터 분리 (Rolling Forward)

```
전체 기간: 2013-12-05 ~ 2026-02-26 (3,076 거래일)

학습: 60% = 약 1,845일 (2013-12-05 ~ 2019-02)
검증: 20% = 약 615일  (2019-02 ~ 2021-07)
시험: 20% = 약 615일  (2021-07 ~ 2026-02)

롤링 포워드: step=30일 단위로 윈도를 앞으로 이동하며 평가
```

---

## 6. 설정 (`config.yaml`)

```yaml
data:
  data_dir: "src/data/dataset"    # 실제 데이터 디렉토리
  news_cache_path: "data/cache/news_tensor.npy"  # 임베딩 캐시
  minerals: ["gold", "silver", "copper"]
  lookback: 180
  horizons: [30, 90, 180]
  news_embed_dim: 384
```
