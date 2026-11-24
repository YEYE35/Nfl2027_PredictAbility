# NFL Big Data Bowl 2027 — 루키 선수 운동능력 · 역할 예측 프로젝트

드래프트 **전(Pre-Draft)** 시점에 알 수 있는 컴바인 측정치, 10Hz 컴바인 트래킹 데이터, 인적·대학 정보만으로 NFL 루키 시즌의 **실전 경기 운동능력과 역할/생산성**을 예측하여 **드래프트 실패(Bust)를 예방하고 핵심 신입을 발굴**하기 위한 머신러닝 프로젝트입니다.

---

## 1. 프로젝트 핵심 전략 및 주요 발견

### 1.1 2중 신입 평가 축 (드래프트 의사결정 반영)
드래프트 스카우팅의 실제 의사결정 구조를 반영해 두 가지 기준을 함께 예측·평가합니다.
1. **절대적으로 폭발적인 신입 (Absolute Standout)**: 전체 NFL 신입 풀(포지션 내)에서 절대적 운동능력·스냅 비중이 최상위권(상위 20% / 주전 상위 1/3)인 선수.
2. **해당 시즌 코호트 상대 우위 (Cohort-Relative Standout)**: 매년 드래프트 풀(`draft_year`: 2023/2024/2025)의 수준이 다르므로, **같은 `draft_year × combine_position` 코호트 안에서 상대적으로 두드러지는 선수**.

> ⚠️ 두 축 모두 "다른 선수와 비교한" **상대 라벨**입니다. 비교 기준(백분위 참조분포, 코호트 평균·표준편차, 상위 20% / 1/3 컷)은 **Train 선수에게서만 계산**하고 Test에는 그 기준을 적용만 합니다 (`RelativeTargetEncoder`, 3.5절). 510명 전체로 기준을 잡으면 Test 선수의 실제 성적이 Train 라벨의 컷과 순위에 섞여 들어가는 타깃 누수가 됩니다.

### 1.2 원본 데이터 정밀 검증을 통한 4가지 핵심 전략
1. **`combine_tracking.csv`의 `dis` 오염 폐기 및 `(x, y)` 재계산**:
   - 전 연도(2023~2025) 검증 결과, 원본 `dis` 컬럼은 2프레임씩 복제되며 실제 `(x, y)` 이동거리의 **중앙값 53.2%**만 기록되는 오류가 확인되었습니다.
   - `dis`를 전면 폐기하고 `hypot(dx, dy)`로 재계산하여 `path_efficiency <= 1.0` 물리 불변성과 40야드 스플릿 정합성(`r = 0.921`)을 확보했습니다.
2. **부호 없는 가속도 `a`의 3축 생체역학 분해**:
   - 트래킹의 `a`는 추진·제동 구분이 없는 총 벡터 가속도 크기($\sqrt{a_\parallel^2 + a_\perp^2}$)로, 실제 감속 구간(`ds/dt < -1`)의 99.7%에서도 `a > 1.0`으로 기록됩니다.
   - 이를 **추진 가속($a_\parallel^+$), 제동 감속($a_\parallel^-$), 구심/횡가속($a_\perp = s|\omega|$)** 3축으로 분리해 컷·코너링·급제동 민첩성을 정밀 추출했습니다.
3. **컴바인 측정 불참(결측) 선수 구제 전략**:
   - 510명 검증 결과, `forty` 불참자(86명)의 1~3R 비율은 **30.2%**(측정자 30.7%, Pro Bowl 3명 배출), `three_cone` 불참자(330명)의 루키 스냅 중앙값은 **117**(측정자 94), 7개 드릴 전부 불참자(27명) 중에서도 Pro Bowl 2명이 나왔습니다.
   - 즉 불참자는 "실력 부족/부상"과 "전략적 불참 상위 유망주"가 혼재하므로 삭제하지 않고, **510명 전원(100%)이 보유한 포지션 스킬 드릴 10Hz 트래킹 운동학 + 결측 지시자 플래그**로 유망주를 구제합니다 (7개 전부 불참 27명 **100% 구제**).
4. **포지션 교란(Simpson's Paradox) 및 데이터 누수 원천 차단**:
   - `draft_round`, `draft_overall_pick`, `draft_tier`는 드래프트 이후 정보이므로 **입력 피처 $X$에서 완전히 배제**합니다 (층화 분할 및 사후 버스트/스틸 오류 분석 전용).
   - 전체 $R^2$는 포지션 체격 차이가 만든 착시가 크므로, 모든 비교에서 **포지션 평균을 제거한 `within-position R²`**를 핵심 지표로 사용합니다.
   - 대치·스케일링·포지션/코호트 Z-score·드릴별 Z-score·Isolation Forest는 모두 **`scikit-learn` `Pipeline` 내부 Train 폴드에서만 `fit`**합니다.
   - 같은 원칙을 **타깃(y)에도** 적용합니다. 포지션/코호트 백분위·코호트 Z·폭발 플래그·`starter_flag`·버스트/스틸 진단은 parquet에 저장하지 않고, Train(CV에서는 매 폴드의 Train 폴드)에서만 기준을 학습한 `RelativeTargetEncoder`로 그때그때 만듭니다. `check_relative_labels_ignore_holdout()`이 "Test 값을 마구 바꿔도 Train 라벨이 그대로인지"를 매 실행마다 검사합니다.

---

## 2. 디렉터리 및 파일별 역할

```text
nfl-bdb2027/
├── src/
│   ├── config.py                    # 경로(로컬/Colab 자동 탐색)·시드(42)·도메인 상수·환경 점검
│   └── features/
│       ├── combine_fe.py            # [Stage 2: 컴바인 단독] 정제·이상치·3축 운동학·40yd F-V·결측 구제
│       ├── game_targets.py          # [Stage 3: 경기 단독] 2GB 청크 스트리밍·절대/코호트 다중 타깃
│       └── preprocessing.py         # [Stage 4: 통합] 마스터 테이블·층화 분할·CV 비교·누수 방지 Pipeline
├── notebooks/
│   ├── 00_local_setup.ipynb         # 실행 환경·패키지·data/raw 9개 CSV 연결 검증 (로컬/Colab 겸용)
│   ├── 01_eda_combine.ipynb         # [Stage 1a] 컴바인 측정치·결측·상관·트라이얼 키·글리치 EDA
│   ├── 01_eda_game.ipynb            # [Stage 1b] 경기 시즌 함정·2GB 트래킹·노출량·player_play·타깃 EDA
│   ├── 02_combine_preprocessing_fe.ipynb  # [Stage 2] 컴바인 단독 정제·이상치·결측 구제·피처 추출 실행/확인
│   ├── 03_game_targets.ipynb              # [Stage 3] 경기 단독 1회 스트리밍·절대/코호트 타깃 생성/확인
│   └── 04_integrate_and_preprocess.ipynb  # [Stage 4] 통합·분할·결측/스케일러/피처세트 CV 비교·PCA/Lasso 점검
├── data/
│   ├── raw/                         # 원본 CSV 9종 (~2.2GB, Git 제외)
│   ├── processed/                   # 중간·전처리 산출물 Parquet (T1~T5, Git 제외)
│   └── splits/test_ids.csv          # 포지션×드래프트 티어 층화 20% 고정 테스트셋 (102명, Git 추적)
├── reports/
│   ├── eda.md                       # Stage 1 EDA 확정 사실 및 결정사항(D1~D10) 요약
│   ├── tables/*.csv                 # Stage 1·Stage 4 비교 요약표 (Git 추적)
│   └── figures/                     # 로컬 시각화 이미지 (Git 제외)
└── requirements.txt                 # 팀 공통 Python 3.11.9 기준 고정(==) 버전 의존성
```

### 파이썬 모듈 (`src/`) 상세 설명
| 파일 | 데이터 범위 | 핵심 역할 및 산출물 |
|---|---|---|
| `src/config.py` | 공통 | 로컬 `data/raw` 및 Colab Drive 자동 경로 탐색(`StopIteration` 방지), `SEED=42`, `TRIAL_KEY`, `SUB_POSITION_MAP`(8개 서브포지션), 글리치/잘림 임계값 단일 관리 |
| `src/features/combine_fe.py` | 컴바인 단독 | `combine_tracking.csv` 정제(`dis` 폐기→`(x, y)` 재계산), 3축 가속 분해, 40야드 Samozino–Morin F-V 피팅, `draft_age`/물리 파생, 스킬 드릴 구제 피처 생성 → `combine_trials.parquet`, `combine_trial_flags.parquet`, `combine_forty_curves.parquet`, `combine_player_features.parquet`(510×81) |
| `src/features/game_targets.py` | 경기 단독 | 2GB `game_tracking_2023~25.csv` 1회 청크 스트리밍(`games.csv` 시즌 조인, 루키 조기 필터링, 글리치 제거) + `player_play.csv` 집계 → `game_play_peaks.parquet`, `game_rookie_athletic.parquet`, `game_rookie_role.parquet` (**원값 타깃만 저장**). 상대 라벨은 `RelativeTargetEncoder`(Train 전용 `fit`)로 생성하고 `check_relative_labels_ignore_holdout()`으로 누수 검사 |
| `src/features/preprocessing.py` | 컴바인+경기 통합 | T2+T3+T4 Left Join → `master_player_table.parquet`(510×123), `test_ids.csv` 생성, 누수 방지 커스텀 Transformer 4종, 선형/트리 듀얼 모드 `build_preprocessor()`, 5×5 층화 CV **쌍대 비교**(기준 대비 Δ·승수·보정 SE·판정) 및 PCA/Lasso 점검 |

### 노트북 (`notebooks/`) 상세 설명
- **`00_local_setup.ipynb`**: 로컬(`.venv`)과 Colab 양쪽에서 `src/config.py` 탐색, `data/raw` 9개 CSV 용량 확인, `pyarrow`/`scikit-learn` 동작 점검.
- **`01_eda_combine.ipynb`**: 컴바인 포지션별 결측률·분포·전체 vs 그룹 내 상관(교란 검증), `event_id` 포함 트라이얼 키(`D1`), 분위수 잘림 오탐(`D7`) 도출.
- **`01_eda_game.ipynb`**: `game_id[:4]` 연도 함정(`D3`), 2GB 1회 스트리밍, 키 정합성(510→497→492), `player_play` 채움률 vs TRUE 비율, `starter_flag` 미리보기.
- **`02_combine_preprocessing_fe.ipynb`**: `[데이터 범위 / 작업 단계 / 목적 / 확인 결과]` 표준 형식으로 Stage 2 전처리·생체역학 피처 추출을 단계별 실행·확인.
- **`03_game_targets.ipynb`**: 루키 경기 3축 운동학 타깃 원값(REG+POST 및 PRE 민감도), 포지션별 정규시즌 스냅 분포, Test 102명을 봉인한 채 Train으로만 학습한 `RelativeTargetEncoder`로 절대/코호트 상대 라벨·사후 버스트/스틸 진단 생성 및 Test 섭동 누수 검사.
- **`04_integrate_and_preprocess.ipynb`**: 마스터 조인, 80/20 테스트 봉인, 결측 대치 3종(`group_median`/`knn`/`mice`)·스케일러 3종(`standard`/`robust`/`yeo_johnson`)·피처 세트 4종(`FS-Trad`~`FS-Full`) 5×5 CV 쌍대 비교(기준 대비 Δ·승수·판정), 상대 타깃 누수 검사, PCA·Lasso/ElasticNet 생존 피처 확인.

---

## 3. 적용된 전처리 · 피처추출 기법의 개념 · 수식 · 원리

### 3.1 트래킹 정제 및 3축 생체역학 운동학 (`combine_fe.py`, `game_targets.py`)
1. **실제 이동거리 및 경로 효율 (Path Efficiency)**:
   $$\Delta d_t = \sqrt{(x_t - x_{t-1})^2 + (y_t - y_{t-1})^2}, \quad \text{Path Efficiency} = \frac{\sqrt{(x_T - x_0)^2 + (y_T - y_0)^2}}{\sum_{t=1}^{T} \Delta d_t} \in [0, 1]$$
   - 오염된 `dis` 대신 `(x, y)` 차분으로 재계산해 직선 주행은 1에 가깝고, 컷·방향전환이 많을수록 작아지는 물리 정합성을 보장합니다.
2. **추진 가속 · 제동 감속 분리**:
   $$\frac{\Delta s}{\Delta t} \implies a_\parallel^+ = \max\!\left(\frac{\Delta s}{\Delta t},\ 0\right), \quad a_\parallel^- = \max\!\left(-\frac{\Delta s}{\Delta t},\ 0\right)$$
   - `|Δs/Δt| > 12.0 yd/s²`인 센서 튐 프레임 쌍은 제외합니다.
3. **각도 Wrap-Around 보정 및 구심/횡가속 ($a_\perp$, 컷·코너링 민첩성)**:
   $$\Delta\theta_t = \big((\theta_t - \theta_{t-1} + 180^\circ) \bmod 360^\circ\big) - 180^\circ \in [-180^\circ, 180^\circ)$$
   $$\omega_t = \frac{|\Delta\theta_t| \cdot \frac{\pi}{180}}{\Delta t}, \quad a_{\perp, t} = s_t \cdot \omega_t \quad (s_t \ge 1.5\ \text{yd/s})$$
   - 정지 상태의 방향각 떨림을 막기 위해 이동 중($s \ge 1.5$)에만 계산하며, 0.3초(3프레임) 누적 회전각 $\ge 60^\circ$로 방향전환(COD) 구간을 탐지합니다.
   - 경기 트래킹에서는 신체 방향(`o`)과 이동 방향(`dir`)의 차이 변화율($\frac{\Delta|o - \text{dir}|}{\Delta t}$, `hip_flip`)도 추가 추출합니다.

### 3.2 40야드 Samozino–Morin Force-Velocity 물리 곡선 피팅
50개 시계열 점을 그대로 넣으면 소표본($N=510$)에서 차원 폭발이 발생하므로, 스포츠 생체역학의 2-파라미터 지수 모델로 압축합니다 (실측 피팅 중앙 $R^2 = 0.9939$):
$$v(t) = v_{\max}\left(1 - e^{-t/\tau}\right)$$
- $v_{\max}$: 이론 최고속도, $\tau$: 가속 시상수(작을수록 초반 폭발형)
- $a_0 = \dfrac{v_{\max}}{\tau}$: $t=0$ 이론 초기 폭발 가속도
- $P_{\max,\text{abs}} = \dfrac{1}{4}\, m_{\text{kg}}\, v_{\max}\, a_0 \cdot (0.9144)^2$ (SI 단위 Watts, 이론 최대 기계적 스프린트 파워)

### 3.3 도메인 물리 파생 & 드래프트 전 메타데이터
- **`draft_age`**: 드래프트 기준일(`YYYY-04-28`) 대비 만 나이 (20.65~26.38세)
- **`sub_position`**: 거친 5개 포지션 내부 교란(예: `DL` 안의 330 lb `NT/DT` vs 245 lb `DE/OLB`)을 분리하기 위해 8개 서브포지션(`EDGE`, `IDL`, `CB`, `S`, `OT`, `IOL`, `WR`, `TE`)으로 세분화
- **물리 지수**:
  - $\text{BMI} = \dfrac{\text{weight}}{\text{height}^2} \times 703$, `wing_to_height`, `arm_to_height`, `hand_to_height`
  - $\text{Speed Score} = \dfrac{\text{weight} \times 200}{\text{forty}^4}$, $\text{HaSS} = \text{Speed Score} \times \dfrac{\text{height}}{73}$
  - $\text{Burst Score} = \text{vertical} + \text{broad\_jump}$
  - $\text{Sayers Peak Anaerobic Power (W)} = 60.7 \cdot \text{vertical}_{\text{cm}} + 45.3 \cdot \text{weight}_{\text{kg}} - 2055$

### 3.4 누수 방지 커스텀 Transformer (`preprocessing.py`)
모든 통계량은 반드시 **Train 폴드에서만 `fit`**하고 Validation/Test에는 `transform`만 적용합니다.
1. **`WithinGroupZScoreAdder` (포지션 내 + 시즌 코호트 내 2중 Z-score)**:
   - 포지션 내 $z_{\text{pos}} = \dfrac{x - \mu_p}{\sigma_p}$와 함께, 소표본 코호트(`draft_year × position`)의 분산 폭발을 막기 위해 포지션 통계로 부드럽게 수축(Shrinkage, $w = \frac{n}{n+10}$)한 코호트 Z-score $z_{\text{coh}}$를 추가합니다.
2. **`DrillZScoreFeaturizer` (동일 `drill_name` 대비 상대 운동능력, 510명 전원 구제)**:
   - 어떤 드릴을 수행했느냐에 따른 스케일 편향을 없애기 위해, Train 트라이얼만으로 각 `drill_name`의 $(\mu_d, \sigma_d)$를 학습한 뒤 선수별 평균 상대 실력(`drill_z_composite` 등)을 산출합니다.
3. **결측치 대치 3종 비교**:
   - **`WithinGroupMedianImputer`**: 포지션별 Train 중앙값 대치 (전체 중앙값의 포지션 왜곡 방지).
   - **`KNNImputer` (포지션 가중, $K=7$)**: `RobustScaler` 적용 후 포지션 원핫 가중치($\times 2$)를 결합해 같은 포지션 내 가장 유사한 7명의 거리 역수 가중 평균으로 대치.
   - **`MICE` (`IterativeImputer`)**: 스케일 격차에 따른 수렴 불안정을 막기 위해 `RobustScaler` 사전 적용 후, 결측이 있는 각 컬럼을 임시 타깃으로 두고 가장 상관 높은 12개 피처로 `Ridge(alpha=10)` 회귀를 반복 수행해 대치.
4. **스케일링 · 분포 변환 3종 비교**:
   - **`StandardScaler`**: $z = \dfrac{x - \mu}{\sigma}$
   - **`RobustScaler`**: $x_{\text{rob}} = \dfrac{x - Q_2}{Q_3 - Q_1}$ (극단값에 강건)
   - **`Yeo-Johnson PowerTransformer`**: 0과 음수에도 적용 가능한 거듭제곱 변환으로 치우친 분포의 정규성을 높이는 최적 $\lambda$를 Train에서 최대우도추정:
     $$\psi(\lambda, x) = \begin{cases} \frac{(x+1)^\lambda - 1}{\lambda} & (x \ge 0,\ \lambda \ne 0) \\ \ln(x+1) & (x \ge 0,\ \lambda = 0) \\ -\frac{(-x+1)^{2-\lambda} - 1}{2-\lambda} & (x < 0,\ \lambda \ne 2) \\ -\ln(-x+1) & (x < 0,\ \lambda = 2) \end{cases}$$
5. **다변량 이상치 점수 (`IsolationForest`) & 듀얼 모드 파이프라인**:
   - 무작위 분할 트리에서 고립되는 경로 길이가 짧을수록 이상치 점수를 높게 부여해 보조 피처로 활용 가능.
   - `build_preprocessor(mode="linear")`는 NaN 0건의 정규화 행렬을, `mode="tree"`는 추후 XGBoost/CatBoost용으로 원본 결측(`NaN`)을 네이티브 보존해 반환합니다.

### 3.5 견고한 타깃 및 포지션 내부 평가 지표 (`game_targets.py`, `preprocessing.py`)
- **견고한 속도 타깃 (`g_smax_top5pct_mean`)**: 루키 `REG+POST` 플레이 $n$개의 최고속도 중 **상위 5%** ($k = \max(1, \lceil 0.05\,n \rceil)$)의 평균입니다. 처음에는 고정 상위 5개 평균을 썼지만, 플레이가 많을수록 운 좋게 튄 값이 쌓여 저절로 커지는 출전량 편향이 컸습니다(Train 포지션 내 $\log n$과의 상관 0.73). 평균 낼 개수를 $n$에 비례시키면 많이 뛸수록 튄 값이 묽어져 상관이 0.12로 줄어듭니다. 여기에 저노출 편향을 완화하기 위해 신뢰도 가중치 $w_i = \dfrac{n_i}{n_i + 20}$ 및 프리시즌 포함 민감도 버전(`_with_pre`)을 함께 저장합니다.
- **상대 타깃 (`RelativeTargetEncoder`, Train 전용 `fit`)**: 아래 라벨은 저장하지 않고, `fit(train)`이 학습한 기준으로 `transform`해서 만듭니다.
  - `{col}_pos_pct`, `{col}_cohort_pct`: Train 참조분포 안에서의 평균 순위 백분위 $\big((lo + hi + 1)/2\big)/n$. Train 선수 자신에게는 기존 `rank(pct=True)`와 같은 값이 나옵니다.
  - `{col}_cohort_z` $= (x - \mu_{\text{coh}}) / \sigma_{\text{coh}}$ ($\mu, \sigma$는 Train 코호트 통계). 코호트 표본이 5명 미만이면 포지션 통계로 대체합니다.
  - `is_explosive_speed_abs` / `_cohort`: 해당 백분위 $\ge 0.8$ (상위 20%).
  - `starter_flag` / `starter_flag_cohort`: Train 포지션(코호트) 정규시즌 스냅의 상위 1/3 컷 이상.
  - `diag_draft_bust` (1~3R인데 스냅 < Train 포지션 중앙값), `diag_draft_steal` (4R 이후인데 주전): 사후 오류 분석 전용, 입력 피처로 쓰지 않음.
  - 속도 계열 기준은 저노출(`low_exposure`) 선수를 빼고, 스냅 기준은 전원으로 학습합니다. CV에서는 **폴드마다 Train 폴드로 다시 `fit`**합니다.
- **포지션 내부 잔차 $R^2$ (`r2_within_pos`)**: 실제값 $y$와 예측값 $\hat{y}$ 각각에서 소속 포지션 평균을 뺀 잔차 $(y - \bar{y}_p,\ \hat{y} - \bar{\hat{y}}_p)$ 사이의 $R^2$를 측정해 포지션 교란을 제거한 순수 변별력을 평가합니다.

---

## 4. 전처리 · 피처추출 5×5 층화 CV 검증 요약 (Train 408명 중 평가 가능 273명, RidgeCV)

테스트셋(`data/splits/test_ids.csv`, 102명)은 완전히 봉인하고, Train 408명 중 경기 노출이 충분한(`low_exposure == False`) **273명**으로 5×5 Repeated Stratified K-Fold CV를 돌린 결과입니다. 요약표는 `reports/tables/4_*.csv`, 25개 폴드별 원점수는 `reports/tables/4_cv_fold_scores.csv`에 있습니다.

### 4.0 판정 방법: 같은 폴드 위의 쌍대 비교
모든 후보는 **똑같은 25개 폴드**에서 평가되므로, 평균끼리 비교하지 않고 폴드마다 차이 $d_i = \text{후보}_i - \text{기준}_i$ (지표: within-position $R^2$)를 구해 비교합니다.
- **승수 (`wins`)**: 25개 폴드 중 후보가 기준보다 높았던 폴드 수.
- **보정 표준오차 (`delta_se_corrected`)**: 반복 K-Fold의 폴드들은 학습 데이터를 공유하므로 서로 독립이 아닙니다. 그래서 단순 $\sigma_d/\sqrt{J}$ 대신 Nadeau–Bengio 보정을 씁니다.
  $$\widehat{SE}_{\text{corr}} = \sqrt{\left(\frac{1}{J} + \frac{n_{\text{test}}}{n_{\text{train}}}\right)\hat\sigma_d^{2}}, \qquad J = 25,\ \frac{n_{\text{test}}}{n_{\text{train}}} = \frac{1}{4}$$
- **판정 (`verdict`)**: $\bar d > 2\,\widehat{SE}$ → `우위`, $\bar d < -2\,\widehat{SE}$ → `열위`, 그 사이 → `차이 없음`. 기준 행은 `기준`.

### 4.1 피처 세트 (기준 `FS-Trad`)
| 타깃 | 피처 세트 | 전체 $R^2$ | 포지션 내부 $R^2$ (평균 ± 표준편차) | Δ vs FS-Trad | 보정 SE | 승수 | 판정 |
|---|---|---:|---:|---:|---:|---:|---|
| 절대 (`g_smax_top5pct_mean`) | `FS-Trad` (전통 12개) | 0.9220 | 0.1780 ± 0.1369 | — | — | — | 기준 |
| 절대 (`g_smax_top5pct_mean`) | `FS-Phys` (+물리·나이·결측플래그) | 0.9159 | 0.1176 ± 0.1553 | −0.0604 | 0.0246 | 1/25 | 열위 |
| 절대 (`g_smax_top5pct_mean`) | `FS-Biomech` (+3축·F-V·스킬구제·드릴Z) | 0.9137 | 0.1112 ± 0.1590 | −0.0669 | 0.0419 | 3/25 | 차이 없음 |
| 절대 (`g_smax_top5pct_mean`) | `FS-Full` (+NFL NGS 점수) | 0.9128 | 0.1001 ± 0.1623 | −0.0780 | 0.0458 | 3/25 | 차이 없음 |
| 코호트 상대 (`g_smax_top5pct_mean_cohort_z`) | `FS-Trad` | 0.1163 | 0.1362 ± 0.0600 | — | — | — | 기준 |
| 코호트 상대 | `FS-Phys` | 0.1102 | 0.1298 ± 0.0646 | −0.0064 | 0.0122 | 10/25 | 차이 없음 |
| 코호트 상대 | `FS-Biomech` | 0.1003 | 0.1229 ± 0.0802 | −0.0134 | 0.0252 | 8/25 | 차이 없음 |
| 코호트 상대 | `FS-Full` | 0.1003 | 0.1228 ± 0.0799 | −0.0134 | 0.0262 | 8/25 | 차이 없음 |

- **피처를 더해도 나아지지 않았습니다.** 절대 타깃에서는 전통 12개(`FS-Trad`)가 가장 높고, `FS-Phys`는 25개 폴드 중 24개에서 졌습니다(`열위`). 나머지는 평균이 더 낮지만 보정 SE 2배 안이라 `차이 없음`입니다.
- 이전 타깃(고정 상위 5개 평균)에서는 피처를 더할수록 평균이 올라가는 것처럼 보였습니다(FS-Trad 0.048 → FS-Full 0.118). 그 상승분의 상당 부분은 타깃에 섞인 **출전량(≈지명 순위)**을 맞힌 몫이었던 것으로 보입니다. 예를 들어 이전 최상위 피처였던 `draft_age`는 포지션 안에서 플레이 수와 상관이 있습니다(어릴수록 많이 뜀, r = −0.22). 출전량 편향을 줄이자 그 효과가 사라졌습니다.
- 지금 표본(273명)에서는 피처 100여 개를 Ridge에 넣으면 신호보다 노이즈가 늘어납니다. 모델링 단계에서는 `FS-Trad`를 기준 모델로 두고, 추가 피처는 선별해서 넣는 편이 안전합니다.

### 4.2 결측 대치 · 스케일러 (`FS-Biomech`, `g_smax_top5pct_mean`)
| 비교 | 후보 | 포지션 내부 $R^2$ | Δ vs 기준 | 보정 SE | 승수 | 판정 |
|---|---|---:|---:|---:|---:|---|
| 결측 대치 | `group_median` | 0.1112 | — | — | — | 기준 |
| 결측 대치 | `knn` | 0.1131 | +0.0019 | 0.0139 | 12/25 | 차이 없음 |
| 결측 대치 | `mice` | 0.0924 | −0.0188 | 0.0243 | 8/25 | 차이 없음 |
| 스케일러 | `robust` | 0.1112 | — | — | — | 기준 |
| 스케일러 | `standard` | 0.1186 | +0.0075 | 0.0155 | 15/25 | 차이 없음 |
| 스케일러 | `yeo_johnson` | 0.1239 | +0.0127 | 0.0214 | 16/25 | 차이 없음 |

- 세 대치법, 세 스케일러 사이에 **판정할 만한 차이가 없습니다.** 따라서 `build_preprocessor()`의 선형 모드 기본값인 가장 단순한 조합(`group_median` + `robust`)을 유지합니다.

### 4.3 PCA / 정규화 점검 (`FS-Biomech`, Train 273명, 타깃 `g_smax_top5pct_mean_cohort_z`)
- 총 107개 피처(이상치 점수 포함) 중 PC 13개로 80%, PC 20개로 90% 분산을 설명합니다.
- Lasso는 6개, ElasticNet은 9개 피처만 남겼습니다. ElasticNet 계수 크기 상위는 `zpos_combine_weight`(−), `zcoh_forty`(−), `zcoh_combine_weight`(−), `zpos_forty`(−), `zcoh_comb_any_smax_max`(+), `zpos_skill_smax_max`(+) 순입니다 (`4_top30_enet_lasso_features.csv`). "같은 포지션 안에서 가볍고 40야드가 빠르고 드릴 최고속도가 높을수록 경기에서 빠르다"는 물리적으로 자연스러운 방향입니다.
- Lasso는 서로 상관된 피처 중 하나만 남기는 경향이 있어, "살아남음 = 중요함"으로 해석하지 않습니다. 탐색용 참고 자료입니다.

---

## 5. 실행 방법 (로컬 / Google Colab)

### 로컬 환경 설정 (팀 공통 Python 3.11.9)
`requirements.txt`의 모든 패키지는 `==`로 고정되어 있습니다. 패키지를 추가할 때도 버전을 고정해서 적고 팀원에게 알려 주세요.

**macOS (conda)**
```bash
conda create -n nfl python=3.11.9 -y
conda activate nfl
python --version          # Python 3.11.9 확인
pip install -r requirements.txt
nbstripout --install      # 환경을 새로 만들었으면 반드시 다시 실행 (git 필터가 이 환경의 python 경로를 가리킴)
```

**Windows (venv, python.org 3.11.9 설치본)**
```powershell
py -3.11 -m venv .venv
.venv\Scripts\activate
python --version          # Python 3.11.9 확인
pip install -r requirements.txt
nbstripout --install
```
원본 CSV 9개 파일을 `data/raw/`에 배치한 뒤 아래 명령으로 Stage 2~4 전체 파이프라인을 재현할 수 있습니다:
```bash
python -m src.features.combine_fe
python -m src.features.game_targets
python -m src.features.preprocessing
```

### 모델링 단계에서 전처리 파이프라인 바로 사용하기
```python
import pandas as pd
from src import config
from src.features.preprocessing import build_preprocessor, create_or_load_test_split

master = pd.read_parquet(config.PROCESSED_FILES["master_player_table"])
trials = pd.read_parquet(config.PROCESSED_FILES["combine_trials"])
train_df, test_df = create_or_load_test_split(master)

# 선형 계열 (Ridge / Lasso / ElasticNet)
# 4.2절: 대치·스케일러 간 판정할 만한 차이가 없으므로 가장 단순한 기본 조합을 사용
prep_linear = build_preprocessor(
    feature_set="FS-Biomech", mode="linear",
    imputer="group_median", scaler="robust", trials_df=trials,
)

# 트리 계열 (XGBoost / CatBoost / LightGBM — 네이티브 NaN 보존)
prep_tree = build_preprocessor(
    feature_set="FS-Biomech", mode="tree",
    imputer="none", scaler="none", trials_df=trials,
)

# 상대 타깃(백분위·코호트 Z·폭발/주전 플래그)은 Train에서만 기준을 학습
from src.features.game_targets import RelativeTargetEncoder
enc = RelativeTargetEncoder().fit(train_df)
train_y = enc.transform(train_df)["g_smax_top5pct_mean_cohort_z"]
test_y = enc.transform(test_df)["g_smax_top5pct_mean_cohort_z"]
# CV 안에서는 폴드마다 Train 폴드로 다시 fit 할 것 (preprocessing._relative_or_raw_target 참고)
```
