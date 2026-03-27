# SentiCast — 뉴스 반응형 광물가격 예측 시스템

SentiCast는 세 가지 광물(금·은·구리)의 가격을 **시계열 자료 + 뉴스 요약**을 결합하여
30일·90일·180일 지평으로 예측하는 딥러닝 시스템입니다.

---

## 목차
1. [시스템 개요](#시스템-개요)
2. [아키텍처](#아키텍처)
3. [설치](#설치)
4. [데이터 준비](#데이터-준비)
5. [학습](#학습)
6. [평가](#평가)
7. [시각화](#시각화)
8. [프로젝트 구조](#프로젝트-구조)
9. [성능 지표](#성능-지표)

---

## 시스템 개요

| 항목 | 내용 |
|------|------|
| **입력** | 기준일 T 기준 이전 180일 가격 시계열 + 단기·중기·장기 뉴스 요약 임베딩 |
| **출력** | T+30, T+90, T+180 일 후 가격 예측 + 90% 신뢰구간 |
| **광물** | 금(Gold), 은(Silver), 구리(Copper) |
| **평가 지표** | ICP, MIW, Pearson Correlation (예측 지평별) |
| **학습 방식** | 롤링 포워드 Train/Val/Test 윈도 |

---

## 아키텍처

```
입력: price_series (B, 180, 3)  +  news_embeds (B, 3, 3, 384)
         |                                |
    +----v------------------+    +--------v--------+
    |  GLAFF                |    |  NewsEncoder    |
    |  (Global-Local        |    |  Short/Mid/Long |
    |   Adaptive FFT        |    |  Self-Attention |
    |   + Multi-Scale Conv) |    +--------+--------+
    +--------+--------------+             | news_context (B,3,128)
             | (B, 180, 128)              |
    +--------v--------------+    +--------v--------+
    |  Positional Enc.      |    | Cross-Attention |
    |  + Transformer        +---►| (ts x news)     |
    |  Encoder (4L)         |    +--------+--------+
    +-----------------------+             |
                                 +--------v--------+
                                 | MoE Layer       |
                                 | (4 experts,     |
                                 |  Top-2 routing) |
                                 +--------+--------+
                                          | context (B, 180, 128)
                               +----------+-----------+
                               |                      |
                    +----------v------+   +-----------v-----+
                    |  Diffusion      |   |  Quantile Head  |
                    |  Backbone       |   |  (Pinball loss) |
                    |  (DDPM/DDIM,    |   +-----------+-----+
                    |   T=100 steps)  |               |
                    +----------+------+               |
                               | (B, H, M) mean       | (B,H,M,Q)
                               +-----------+----------+
                                           |
                                  출력: 예측값 + CI
```

### 모델 구성요소

| 구성요소 | 역할 |
|----------|------|
| **GLAFF** | FFT 기반 전역 주기성 + 다중 스케일 합성곱 국소 패턴 융합 |
| **Transformer Encoder** | 시계열 자기-어텐션 (n_layers=4, n_heads=8) |
| **NewsEncoder** | 단기/중기/장기 요약 임베딩 → 광물별 컨텍스트 벡터 |
| **Cross-Attention** | 시계열 특징 × 뉴스 컨텍스트 융합 |
| **MoE (4 experts)** | 스파스 전문가 네트워크로 예측 용량 증대 |
| **Diffusion (DDPM/DDIM)** | 확산 모델로 불확실성 정량화 + 앙상블 CI 생성 |
| **Quantile Head** | 분위수 회귀 직접 예측 헤드 (학습 안정성) |

---

## 설치

### 방법 1: uv 사용 (권장)

[uv](https://docs.astral.sh/uv/)는 Rust로 작성된 빠른 Python 패키지 관리자입니다.

```bash
# uv 설치 (처음 한 번)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 가상환경 생성 및 의존성 설치
uv sync

# CPU-only PyTorch 사용 시 (GPU 없는 환경)
uv sync --extra-index-url https://download.pytorch.org/whl/cpu
```

### 방법 2: pip 사용

```bash
pip install -r requirements.txt
```

---

## 웹 인터페이스 (Web UI)

학습과 평가를 위한 Gradio 기반 웹 인터페이스가 제공됩니다.

```bash
# uv로 실행 (권장)
uv run python app.py

# 또는
python app.py
```

브라우저에서 `http://localhost:7860` 접속

**탭 구성:**
- **🚀 Training** — 하이퍼파라미터 설정 및 학습 시작/중지, 실시간 로그 및 손실 곡선
- **🔍 Evaluation** — 체크포인트 선택, 평가 실행, 메트릭 테이블 및 예측 차트
- **📄 Config** — 현재 `config.yaml` 내용 확인

---

## 데이터 준비

데이터 형식 상세는 [`data.md`](data.md)를 참조하세요.

### 실제 데이터 사용
`config.yaml`에서 경로 설정:
```yaml
data:
  prices_path: "data/your_prices.csv"
  news_path:   "data/your_news.csv"
```

### 샘플 데이터 생성 (테스트용)
```bash
python src/data/generate_sample.py --n_days 2000 --out_dir data/sample
```

생성 파일:
- `data/sample/prices.csv` — GBM + 계절성 시뮬레이션 광물 가격
- `data/sample/news.csv`   — 무작위 뉴스 임베딩 (384차원)

---

## 학습

### CLI
```bash
# 첫 번째 롤링 윈도 학습 (uv)
uv run python -m src.train --config config.yaml --split_idx 0

# 전체 롤링 윈도 학습
uv run python -m src.train --config config.yaml --split_idx -1
```

### 웹 UI
`python app.py` 실행 후 브라우저에서 **🚀 Training** 탭을 사용합니다.

주요 하이퍼파라미터는 `config.yaml`에서 조정하거나 웹 UI에서 실시간으로 변경할 수 있습니다.

---

## 평가

### CLI
```bash
uv run python -m src.evaluate \
  --config config.yaml \
  --checkpoint checkpoints/best_split0.pt \
  --split_idx 0
```

### 웹 UI
`python app.py` 실행 후 **🔍 Evaluation** 탭에서 체크포인트를 선택하고 실행합니다.

출력 예시:
```
── Horizon: 30d ──
  Mineral       ICP        MIW   Pearson        MAE      MAPE
──────────────────────────────────────────────────────────────
     gold    0.9120    85.3421    0.8742    32.1500    0.0168
   silver    0.8980    3.2100     0.8210     1.2300    0.0521
   copper    0.9050   412.3000    0.8560   155.4000    0.0198
```

---

## 시각화

```bash
python src/visualize.py \
  --config config.yaml \
  --eval_file results/eval_split0.npz
```

생성 파일 (figures/ 디렉토리):
- `prediction_horizon30d.png`  — 30일 예측 (전체 광물)
- `prediction_horizon90d.png`  — 90일 예측
- `prediction_horizon180d.png` — 180일 예측
- `mineral_gold.png`           — 금 전체 지평 비교
- `mineral_silver.png`         — 은 전체 지평 비교
- `mineral_copper.png`         — 구리 전체 지평 비교

---

## 프로젝트 구조

```
senticast-copilot/
├── data.md                          # 데이터 형식 및 특성 설명
├── config.yaml                      # 전체 하이퍼파라미터 설정
├── requirements.txt
├── data/
│   └── sample/                      # 생성된 샘플 데이터
├── src/
│   ├── data/
│   │   ├── dataset.py               # PyTorch Dataset (슬라이딩 윈도)
│   │   ├── preprocessing.py         # 정규화, 롤링 분리, 뉴스 텐서 빌드
│   │   └── generate_sample.py       # 샘플 데이터 생성기
│   ├── models/
│   │   ├── glaff.py                 # GLAFF: 전역-국소 적응 특징 융합
│   │   ├── news_encoder.py          # 뉴스 임베딩 인코더 + 교차 어텐션
│   │   ├── diffusion.py             # DDPM/DDIM 확산 백본
│   │   ├── moe.py                   # Mixture of Experts
│   │   └── senticast.py             # 메인 모델 (통합)
│   ├── utils/
│   │   └── metrics.py               # ICP, MIW, Pearson, MAE, MAPE
│   ├── train.py                     # 학습 파이프라인
│   ├── evaluate.py                  # 평가 스크립트
│   └── visualize.py                 # 시각화 스크립트
├── checkpoints/                     # 학습된 모델 저장
├── results/                         # 평가 결과 (npz, json)
└── figures/                         # 시각화 결과 (png)
```

---

## 성능 지표

### ICP (Interval Coverage Probability)
예측 신뢰구간이 실제값을 포함하는 비율 (목표: >= 90%)
```
ICP = E[1(y_lower <= y_true <= y_upper)]
```

### MIW (Mean Interval Width)
신뢰구간의 평균 폭 (좁을수록 정밀한 예측)
```
MIW = E[y_upper - y_lower]
```

### Pearson Correlation
예측값과 실제값 간의 선형 상관계수 (1에 가까울수록 우수)
```
r = sum((y_true - mean_true)(y_pred - mean_pred)) / (std_true * std_pred * N)
```

---

## 모델 선택 근거

| 모델 요소 | 선택 이유 |
|----------|-----------|
| **GLAFF** | 광물 가격의 장단기 주기성 (계절성, 경기 사이클) 효과적 포착 |
| **Transformer** | 장기 의존성 모델링 (180일 맥락) |
| **뉴스 교차-어텐션** | 뉴스 내용에 따른 가격 방향성 학습 |
| **MoE** | 광물별·지평별 이질적 특성을 별도 전문가로 처리 |
| **Diffusion** | 가격 예측의 불확실성 정량화 (확률적 CI 생성) |
| **Quantile head** | 확산 학습 보완 -> 안정적인 분위수 직접 예측 |
