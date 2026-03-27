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
news_mask    : 뉴스 가용성 마스크 → shape (3,)  bool
               # True = 해당 광물의 뉴스 존재, False = 뉴스 없음 (2017-10 이전)
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

### 뉴스 임베딩 (`news_encoder` 설정)

뉴스 텍스트는 학습 전에 한 번 임베딩되며, 결과는 `data/cache/news_tensor.npy`에 캐시됩니다.

| 옵션 | 설명 | 차원 | 설치 |
|------|------|------|------|
| `"finbert"` | **권장** – `ProsusAI/finbert` (금융 특화 BERT), CLS 토큰 | 768 | `pip install transformers` |
| `"sentence-transformers"` | `all-MiniLM-L6-v2`, 범용 의미 유사도 모델 | 384 | `pip install sentence-transformers` |
| `"tfidf-svd"` | TF-IDF + Truncated SVD, CPU 경량 폴백 | `news_embed_dim` | sklearn |
| `"auto"` | finBERT → sentence-transformers → TF-IDF+SVD 순서로 시도 | 자동 감지 | — |

> **finBERT를 권장하는 이유**: 금융 뉴스로 사전 학습(fine-tuned)되어 금리, 달러, 지정학적 리스크
> 같은 금융 용어를 더 정확하게 이해합니다. 범용 모델(sentence-transformers)보다 금융 텍스트의
> 의미론적 구분 능력이 뛰어나 모델 예측 성능에 직접적으로 기여합니다.

> **캐시 주의**: `news_encoder`를 변경할 경우 `news_cache_path`도 함께 변경하거나
> 기존 캐시 파일을 삭제하여 stale 캐시를 방지하세요.

### 뉴스 가용성 마스크 (`news_mask`)

뉴스 데이터는 2017-10-04부터만 존재합니다. 이전 날짜(전체의 31.3%, 학습 분할의 52%)에는
뉴스가 없으므로, 학습 시 두 가지 처리가 적용됩니다:

1. **NewsEncoder 게이팅**: `news_mask=False`인 광물의 context 벡터를 **명시적으로 0으로 설정**.
   Linear projection의 bias 항이 "뉴스 없음"을 실제 신호처럼 보이게 만드는 문제를 방지합니다.

2. **CrossAttention 마스킹**: `key_padding_mask`를 사용해 뉴스 없는 광물을 attention에서 제외.
   전체 광물이 마스크된 배치 행은 attention을 건너뛰어 NaN 발생을 방지합니다.

```
뉴스 없는 구간 (2013-12 ~ 2017-10):
  news_embeds → 0 벡터
  news_mask   → [False, False, False]
  → NewsEncoder output: [0, 0, 0]   (게이팅)
  → CrossAttention: ts_feat 그대로 통과 (마스킹)

뉴스 있는 구간 (2017-10 이후):
  news_embeds → 실제 임베딩
  news_mask   → [True, True, True]
  → NewsEncoder output: 실제 context 벡터
  → CrossAttention: 뉴스 정보가 시계열 특징과 융합
```

### 결측치 처리
- 주말/공휴일: 선형 보간

---

## 5. 데이터 분리 (Rolling Forward)

```
전체 기간: 2013-12-05 ~ 2026-02-26 (3,076 거래일)

학습: 60% = 약 1,845일 (2013-12-05 ~ 2019-02)  ← 52%가 뉴스 없음 → news_mask로 처리
검증: 20% = 약 615일  (2019-02 ~ 2021-07)
시험: 20% = 약 615일  (2021-07 ~ 2026-02)       ← 전부 뉴스 존재

롤링 포워드: step=30일 단위로 윈도를 앞으로 이동하며 평가
```

---

## 6. 설정 (`config.yaml`)

```yaml
data:
  data_dir: "src/data/dataset"          # 실제 데이터 디렉토리
  news_cache_path: "data/cache/news_tensor.npy"  # 임베딩 캐시
  news_encoder: "finbert"               # 뉴스 인코더 선택
  news_embed_dim: 768                   # finBERT=768, sentence-transformers=384
  minerals: ["gold", "silver", "copper"]
  lookback: 180
  horizons: [30, 90, 180]
```
