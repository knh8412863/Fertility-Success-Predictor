# 🧬 Fertility Success Predictor 🧬

난임 환자 시술 데이터를 기반으로 **임신 성공 여부를 예측하는 AI 모델**을 개발한 프로젝트입니다.

- 프로젝트 기간: 26.04.23 ~ 26.04.29 (5일)
- 주제: 난임 환자 대상 임신 성공 여부 예측
- 평가 지표: ROC-AUC
- 팀원: 김나현, 이정훈, 이지민
- 최종 제출 파일: `submission/final_submission.csv`

## 📍 프로젝트 개요

본 프로젝트는 난임 환자의 시술 정보, 과거 시술 이력, 배아/난자 관련 수치, 불임 원인 정보를 활용하여 임신 성공 확률을 예측합니다.

대회 규칙상 test 데이터가 모델 학습, 인코딩, 결측치 처리, feature selection 등에 활용되면 데이터 누수로 간주됩니다. 따라서 전처리와 모델링 전 과정에서 train 기준 처리와 fold 기준 검증을 적용했습니다.

---
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


## 🗂️ 프로젝트 구조

```text
.
├── README.md
├── requirements.txt
├── data/                         # 원본 데이터 위치, GitHub 제외
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_preprocessing_final.ipynb
│   ├── 03_modeling_final.ipynb
│   └── 04_review.ipynb
├── src/
│   ├── 01_EDA.py
│   ├── 02_preprocessing.py
│   └── 03_modeling.py
└── submission/
    └── final_submission.csv
```

심사용 제출 코드는 전처리와 모델링을 하나로 합친 `notebooks/10조_해커톤_최종제출코드.ipynb` 파일로 별도 생성했습니다. 해당 파일은 GitHub 업로드 대상에서는 제외했습니다.

## 🗂️ 데이터

원본 데이터는 GitHub에 포함하지 않습니다. 실행 전 아래 파일을 `data/` 폴더에 위치시켜야 합니다.

```text
data/
  train.csv
  test.csv
  sample_submission.csv
```

전처리 실행 후 아래 파일이 생성됩니다.

```text
data/
  X_train.csv
  y_train.csv
  X_test.csv
  test_ids.csv
```

## 🛠️ 사용 기술

- Python
- pandas, numpy
- scikit-learn
- LightGBM
- CatBoost
- matplotlib

## ⚙️ 실행 방법

```bash
pip install -r requirements.txt
```

노트북 실행 순서:

```text
01_EDA.ipynb
02_preprocessing_final.ipynb
03_modeling_final.ipynb
04_review.ipynb
```

`03_modeling_final.ipynb`는 기본값 기준으로 모델별 OOF/test prediction을 재학습하도록 구성되어 있어 실행 시간이 오래 걸릴 수 있습니다.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


## 📍 데이터 탐색 EDA

`01_EDA.ipynb`에서 데이터 구조, target 분포, 결측치 패턴, 주요 변수별 성공률을 확인했습니다.

- 성공 클래스 비율은 약 25.8%로 불균형 데이터
- 나이 구간이 높아질수록 임신 성공률이 낮아지는 경향 확인
- 배아 이식 경과일에 따라 성공률 차이가 크게 나타남
- 배아 수, 이식 배아 수, 저장 배아 수 등 배아 관련 변수가 중요한 신호를 가짐
- 일부 결측치는 단순 누락이 아니라 미시행/해당 없음의 의미를 가질 수 있음

EDA 결과는 이후 파생변수 생성의 근거로 사용했습니다.

## 📍 데이터 전처리

주요 전처리 내용은 다음과 같습니다.

- DI 시술에서 배아/난자 관련 결측치를 0으로 처리
- PGD/PGS 및 착상 전 유전 검사 관련 결측치를 미시행으로 간주해 0 처리
- 결측 여부 자체가 의미 있는 경과일 변수에 결측 flag 생성
- numeric 결측치는 train median으로 대체
- categorical 결측치는 train mode로 대체
- 범주형 인코딩은 train에만 fit 후 test에 transform
- test에서 처음 등장한 label은 `-1` 처리
- 배아 활용률, 수정률, 난자-배아 전환율, 반복 실패 여부, 불임 원인 count 등 파생변수 생성

## ❗️ 누수 방지 원칙

- test 데이터로 encoder/imputer/scaler를 fit하지 않음
- test 데이터 통계값으로 결측치를 대체하지 않음
- test 데이터에 별도 `get_dummies()`를 적용하지 않음
- target encoding은 fold 내부 train split으로만 생성
- feature selection과 blend weight 탐색은 train OOF 기준으로만 수행
- test 데이터는 최종 예측과 동일한 blend 가중치 적용에만 사용

## 📍 모델링

최종 제출은 단일 모델이 아니라 여러 모델의 OOF/test prediction을 조합한 ensemble입니다.

사용한 주요 모델 후보:

| 모델 후보 | 수행 내용 |
|---|---|
| Compact LGBM | 노이즈가 큰 변수를 제거한 LGBM 모델 |
| LGBM 72 Feature | feature importance 기준 상위 72개 변수를 사용한 LGBM |
| Feature-count LGBM | 중요 변수 개수를 바꿔가며 탐색한 LGBM |
| Raw CatBoost depth7 | 원본 범주형 정보를 최대한 유지한 CatBoost |
| Raw CatBoost depth8 | tree depth를 높이고 regularization을 강화한 CatBoost |
| OOF Blend | fold별 validation 예측값 기준 weighted ensemble |
| Segment 보정 | OOF 분석에서 취약했던 일부 구간의 예측 보정 |

최종 blend 가중치는 별도 파일로 저장하지 않고 코드 안에 고정값으로 명시했습니다.

## 📍 주요 실험 결과

| 실험 | OOF AUC | 비고 |
|---|---:|---|
| LGBM fold-safe encoder baseline | 0.74053 | fold-safe target/frequency encoding |
| Compact LGBM Optuna blend | 0.74081 | compact LGBM 기반 튜닝 |
| Raw CatBoost blend | 0.74084 | raw categorical CatBoost 활용 |
| Raw CatBoost variants | 0.74085 | depth7/depth8 후보 비교 |
| LGBM review 72 feature reproduction | 0.74087 | feature importance 기반 72개 feature |
| Final OOF blend | 0.74088 | compact + lgbm72 + raw CatBoost |
| Feature-count sweep blend | 0.74089 | feature 수 탐색 |
| Conservative segment blend | 0.74092 | OOF 취약 segment 보정 |
| Final blend with feature-count | 0.74093 | 최종 선택 |

CatBoost 단일 모델은 LGBM보다 OOF AUC가 소폭 낮았지만, LGBM과 blend했을 때 OOF AUC가 상승했습니다. 따라서 최종 모델에서는 단독 최고 모델보다 서로 다른 예측 패턴을 가진 모델들을 조합하는 전략을 선택했습니다.

## ✨ 평가 분석

`04_review.ipynb`에서 최종 제출 모델의 OOF prediction을 기준으로 발표 및 검토에 사용할 시각화 자료를 생성했습니다.

- 모델별 OOF AUC 비교 bar chart
- 최종 blend ROC Curve
- fold별 OOF AUC 안정성
- segment별 취약 구간 분석
- LGBM feature importance Top 20
- 최종 blend weight 구성

---

## ✨ 최종 성능

| 구분 | AUC |
|---|---:|
| Local OOF AUC | 0.74093 |
| Public Leaderboard AUC | 0.742116375 |

최종 모델은 OOF 기준으로 선택했으며, public score를 이용해 학습하거나 blend weight를 탐색하지 않았습니다.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


## 최종 제출 모델 요약

최종 제출은 다음 예측값들을 조합했습니다.

- compact LGBM OOF/test prediction
- LGBM 72-feature OOF/test prediction
- raw CatBoost depth8/depth7 OOF/test prediction
- feature-count sweep LGBM OOF/test prediction
- OOF 기준 segment 보정 blend

최종 결과 파일은 `submission/final_submission.csv`입니다.

---

## 한계 및 개선 방향

- 일부 배아 이식 경과일 및 시술 조건 segment에서 상대적으로 낮은 AUC 확인
- 모델 간 예측 상관이 높아 더 다양한 모델 구조 탐색 여지 존재
- segment별 전용 모델 또는 calibration을 적용하면 확률 예측 안정성 개선 가능
- 시술 유형, 나이, 배아 상태 간 interaction feature를 추가로 탐색할 수 있음

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
