"""[컴바인+경기 통합] 마스터 테이블 조인 · 층화 분할 · 결측/스케일링 방식 비교 · 누수 방지 Pipeline.

이 모듈은 Stage 2(T1·T2)와 Stage 3(T3·T4) 산출물을 통합해:
1. `master_player_table.parquet` (510명 마스터 테이블)
2. `data/splits/test_ids.csv` (포지션 × 드래프트 티어 층화 80/20 고정 테스트 분할)
3. 누수 없는 커스텀 scikit-learn Transformer
   - `WithinGroupZScoreAdder`: 포지션 내 + 시즌 코호트(draft_year × position) 내 상대 Z-score (Train-fit)
   - `WithinGroupMedianImputer`: 포지션 그룹별 중앙값 대치 (Train-fit)
   - `DrillZScoreFeaturizer`: 동일 `drill_name` 대비 상대 운동능력 Z-score (Train-fit, 510명 전원 구제)
   - `IsolationForestAnomalyAdder`: 다변량 이상치 점수 보조 피처 (Train-fit)
4. 재사용 전처리 파이프라인 팩토리 `build_preprocessor(mode="linear" | "tree", ...)`
5. Train(408명) 5×5 층화 CV 기반 객관적 비교 함수 (이미 배운 Ridge/Lasso/ElasticNet/PCA 사용)
   - 결측치 방식 비교: `group_median` vs `knn` vs `mice`
   - 스케일링 방식 비교: `standard` vs `robust` vs `yeo_johnson`
   - 피처 세트 비교: `FS-Trad` vs `FS-Phys` vs `FS-Biomech` vs `FS-Full`
   - PCA 설명분산 & Lasso/ElasticNet 생존 피처 점검
   - 모든 비교는 같은 25개 fold의 점수를 짝지어(paired) 기준 설정 대비 차이와 보정 표준오차로 판정

절대 준수 규칙:
- `draft_round`, `draft_overall_pick`, `draft_tier`는 입력 피처 X에 절대 포함하지 않는다.
- 모든 Transformer는 CV 폴드의 Train 데이터에서만 `fit`하고 Validation/Test에는 `transform`만 수행한다.
- 상대 기준 타깃(`*_cohort_z`, `*_pct`, `starter_flag` 등)도 같은 원칙: `RelativeTargetEncoder`를
  CV에서는 학습 fold로, 최종 평가에서는 Train 408명으로만 fit 한다.
"""
from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.linear_model import ElasticNetCV, LassoCV, Ridge, RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import LeaveOneGroupOut, RepeatedStratifiedKFold, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PowerTransformer, RobustScaler, StandardScaler

from src import config
from src.features.game_targets import (
    RelativeTargetEncoder,
    check_relative_labels_ignore_holdout,
    relative_label_base,
)

# =====================================================================
# [데이터: 컴바인+경기 통합] [단계: 피처 세트 정의] 드래프트 결과 컬럼 원천 차단
# =====================================================================
FEATURE_GROUPS: dict[str, list[str]] = {
    "trad_num": config.BODY_COLS + config.DRILL_COLS,
    "phys_meta_num": [
        "draft_age", "is_power5",
        "bmi", "wing_to_height", "arm_to_height", "hand_to_height",
        "speed_score", "hass_score", "burst_score", "sayers_power_w", "split10_to_forty",
    ],
    "missing_flags": [
        "n_missing_drills", "did_forty", "did_agility_any", "did_bench",
        "miss_forty", "miss_three_cone", "miss_short_shuttle", "miss_bench_reps",
    ],
    "biomech_40": [
        "tr40_smax_best", "tr40_smax_mean", "tr40_a_pos_best",
        "tr40_t5yd_best", "tr40_t10yd_best", "tr40_t20yd_best", "tr40_fly10_30_best",
        "tr40_fv_vmax", "tr40_fv_tau", "tr40_fv_a0", "tr40_fv_pmax_rel", "tr40_fv_pmax_abs_w",
        "tr40_late_retention",
    ],
    "biomech_rescue_skill": [
        "skill_n_trials", "skill_smax_max", "skill_smax_p90", "skill_smax_mean",
        "skill_a_pos_max", "skill_a_pos_p90", "skill_decel_max", "skill_decel_p90",
        "skill_a_lat_max", "skill_a_lat_p90", "skill_jerk_p95_max",
        "skill_path_eff_mean", "skill_cod_frames_mean", "skill_turn_deg_mean",
        "comb_any_smax_max", "comb_any_a_pos_max", "comb_any_decel_max", "comb_any_a_lat_max",
    ],
    "ngs_num": config.NGS_COLS,
    "cat_cols": ["combine_position", "sub_position"],
}

ZSCORE_CORE_COLS = [
    "combine_height", "combine_weight", "forty", "ten_yd_split",
    "vertical", "broad_jump", "speed_score", "burst_score",
    "skill_smax_max", "skill_a_pos_max", "skill_decel_max", "skill_a_lat_max",
    "comb_any_smax_max",
]


def get_feature_columns(
    feature_set: Literal["FS-Trad", "FS-Phys", "FS-Biomech", "FS-Full"] = "FS-Biomech",
) -> tuple[list[str], list[str]]:
    """피처 세트별 (수치형 컬럼 목록, 범주형 컬럼 목록)을 반환하고 드래프트 누수를 차단한다."""
    num: list[str] = list(FEATURE_GROUPS["trad_num"])
    cat: list[str] = ["combine_position"]

    if feature_set in ("FS-Phys", "FS-Biomech", "FS-Full"):
        num += FEATURE_GROUPS["phys_meta_num"] + FEATURE_GROUPS["missing_flags"]
        cat = list(FEATURE_GROUPS["cat_cols"])
    if feature_set in ("FS-Biomech", "FS-Full"):
        num += FEATURE_GROUPS["biomech_40"] + FEATURE_GROUPS["biomech_rescue_skill"]
    if feature_set == "FS-Full":
        num += FEATURE_GROUPS["ngs_num"]

    # 중복 제거 및 드래프트 결과 컬럼 금지 단언
    num = list(dict.fromkeys(num))
    cat = list(dict.fromkeys(cat))
    forbidden_hit = set(num + cat).intersection(config.DRAFT_FORBIDDEN_FEATURE_COLS)
    assert not forbidden_hit, f"치명적 누수 오류: 드래프트 결과 컬럼 포함됨 -> {forbidden_hit}"
    return num, cat


# =====================================================================
# [데이터: 컴바인+경기 통합] [단계: 마스터 테이블 조인] T2 + T3 + T4 -> T5
# =====================================================================
def build_master_table(save: bool = True) -> pd.DataFrame:
    """510명 선수 기준으로 T2(컴바인), T3(경기 운동학), T4(경기 역할/생산성)를 Left Join한다."""
    t2 = pd.read_parquet(config.PROCESSED_FILES["combine_player_features"])
    t3 = pd.read_parquet(config.PROCESSED_FILES["game_rookie_athletic"])
    t4 = pd.read_parquet(config.PROCESSED_FILES["game_rookie_role"])

    drop_t3 = [c for c in ["draft_year", "combine_position", "sub_position"] if c in t3.columns]
    drop_t4 = [c for c in ["draft_year", "combine_position", "sub_position"] if c in t4.columns]

    master = (
        t2.merge(t3.drop(columns=drop_t3), on="nfl_id", how="left")
        .merge(t4.drop(columns=drop_t4), on="nfl_id", how="left")
    )
    assert len(master) == 510 and master["nfl_id"].nunique() == 510, "마스터 테이블 510행 무결성 위반!"

    if save:
        master.to_parquet(config.PROCESSED_FILES["master_player_table"], index=False)

    print(
        f"[확인된 결과] T5 마스터 테이블 조인 완료: shape={master.shape} | "
        f"중복 nfl_id=0 | 노출 충족={(~master['low_exposure']).sum()}명"
    )
    return master


# =====================================================================
# [데이터: 컴바인+경기 통합] [단계: 데이터 분할 (누수 차단)]
#   - 포지션 × 드래프트 티어 층화 80/20 고정 테스트 분할 (test_ids.csv)
#   - 시즌 코호트 시간적 일반화 CV (LeaveOneGroupOut on draft_year)
# =====================================================================
def create_or_load_test_split(
    master: pd.DataFrame,
    overwrite: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """고정 테스트 분할(`data/splits/test_ids.csv`)을 생성/로드하여 (train_df, test_df)를 반환한다."""
    config.ensure_dirs()
    if config.TEST_IDS_PATH.exists() and not overwrite:
        test_ids_df = pd.read_csv(config.TEST_IDS_PATH)
    else:
        strat_key = master["combine_position"].astype(str) + "_" + master["draft_tier"].astype(str)
        sss = StratifiedShuffleSplit(n_splits=1, test_size=config.TEST_SIZE, random_state=config.SEED)
        _, test_idx = next(sss.split(master, strat_key))
        test_ids_df = (
            master.iloc[test_idx][["nfl_id", "combine_position", "draft_year", "draft_tier"]]
            .sort_values("nfl_id")
            .reset_index(drop=True)
        )
        test_ids_df.to_csv(config.TEST_IDS_PATH, index=False)

    test_set = set(test_ids_df["nfl_id"])
    train_df = master[~master["nfl_id"].isin(test_set)].reset_index(drop=True)
    test_df = master[master["nfl_id"].isin(test_set)].reset_index(drop=True)

    print(
        f"[확인된 결과] 층화 분할 완료: Train={len(train_df)}명 (80%) | "
        f"Test={len(test_df)}명 (20%, 봉인 -> {config.TEST_IDS_PATH.name})"
    )
    return train_df, test_df


def get_temporal_cv_splits(df: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray]]:
    """드래프트 시즌(2023/2024/2025) 기준 Leave-One-Draft-Year-Out 분할을 반환한다."""
    logo = LeaveOneGroupOut()
    return list(logo.split(df, groups=df["draft_year"]))


# =====================================================================
# [데이터: 컴바인+경기 통합] [단계: 누수 방지 커스텀 Transformer 4종]
#   1) WithinGroupZScoreAdder  : 포지션 내 + 시즌 코호트 내 Z-score (Train-fit)
#   2) DrillZScoreFeaturizer   : 동일 drill_name 대비 상대 운동능력 Z-score (Train-fit)
#   3) WithinGroupMedianImputer: 포지션 그룹별 중앙값 대치 (Train-fit)
#   4) IsolationForestAnomalyAdder: 다변량 이상치 점수 보조 피처 (Train-fit)
# =====================================================================
class WithinGroupZScoreAdder(BaseEstimator, TransformerMixin):
    """Train 통계만 사용해 포지션 내 & 시즌 코호트(draft_year×position) 내 Z-score 피처를 추가한다."""

    def __init__(
        self,
        value_cols: list[str],
        group_col: str = "combine_position",
        add_cohort_z: bool = True,
        min_group_size: int = 5,
    ):
        self.value_cols = value_cols
        self.group_col = group_col
        self.add_cohort_z = add_cohort_z
        self.min_group_size = min_group_size

    def fit(self, X: pd.DataFrame, y=None):
        self.active_cols_ = [c for c in self.value_cols if c in X.columns]
        self.global_stats_ = {}
        self.pos_stats_ = {}
        self.cohort_stats_ = {}

        for col in self.active_cols_:
            s = pd.to_numeric(X[col], errors="coerce")
            g_mean = float(s.mean()) if s.notna().any() else 0.0
            g_std = float(s.std()) if s.notna().sum() > 1 else 1.0
            if not np.isfinite(g_std) or g_std < 1e-6:
                g_std = 1.0
            self.global_stats_[col] = (g_mean, g_std)

            for grp, sub in s.groupby(X[self.group_col]):
                if sub.notna().sum() >= self.min_group_size and sub.std() > 1e-6:
                    self.pos_stats_[(col, grp)] = (float(sub.mean()), float(sub.std()))
                else:
                    self.pos_stats_[(col, grp)] = (g_mean, g_std)

            if self.add_cohort_z and "draft_year" in X.columns:
                for (dy, grp), sub in s.groupby([X["draft_year"], X[self.group_col]]):
                    parent_m, parent_s = self.pos_stats_.get((col, grp), (g_mean, g_std))
                    n_valid = int(sub.notna().sum())
                    if n_valid >= self.min_group_size and sub.std() > 1e-6:
                        # 소표본 코호트 분산 폭발을 막기 위해 포지션 통계로 부드럽게 수축 (Shrinkage)
                        w = n_valid / (n_valid + 10.0)
                        m = w * float(sub.mean()) + (1.0 - w) * parent_m
                        sd = w * float(sub.std()) + (1.0 - w) * parent_s
                        self.cohort_stats_[(col, dy, grp)] = (m, max(sd, 1e-6))
                    else:
                        self.cohort_stats_[(col, dy, grp)] = (parent_m, parent_s)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        new_cols = {}
        pos_arr = X[self.group_col].to_numpy()
        dy_arr = X["draft_year"].to_numpy() if "draft_year" in X.columns else None

        for col in self.active_cols_:
            vals = pd.to_numeric(X[col], errors="coerce").to_numpy(dtype=np.float64)
            g_m, g_s = self.global_stats_[col]
            means = np.array([self.pos_stats_.get((col, g), (g_m, g_s))[0] for g in pos_arr])
            stds = np.array([self.pos_stats_.get((col, g), (g_m, g_s))[1] for g in pos_arr])
            new_cols[f"zpos_{col}"] = (vals - means) / stds

            if self.add_cohort_z and dy_arr is not None:
                c_means = np.array([
                    self.cohort_stats_.get((col, dy, g), self.pos_stats_.get((col, g), (g_m, g_s)))[0]
                    for dy, g in zip(dy_arr, pos_arr)
                ])
                c_stds = np.array([
                    self.cohort_stats_.get((col, dy, g), self.pos_stats_.get((col, g), (g_m, g_s)))[1]
                    for dy, g in zip(dy_arr, pos_arr)
                ])
                new_cols[f"zcoh_{col}"] = (vals - c_means) / c_stds

        return pd.concat([out, pd.DataFrame(new_cols, index=X.index)], axis=1)


class DrillZScoreFeaturizer(BaseEstimator, TransformerMixin):
    """Train 선수 트라이얼로만 `drill_name`별 평균·표준편차를 fit하여 동일 드릴 대비 상대 실력을 산출한다.

    모든 510명이 트라이얼을 보유하므로, 전통 드릴을 포기(결측)한 선수를 공정하게 구제하는 핵심 피처다.
    """

    def __init__(
        self,
        trials_df: pd.DataFrame,
        metrics: tuple[str, ...] = ("s_max", "a_tan_pos_max", "a_tan_neg_max", "a_lat_max"),
    ):
        self.trials_df = trials_df
        self.metrics = metrics

    def fit(self, X: pd.DataFrame, y=None):
        train_ids = set(X["nfl_id"])
        valid = self.trials_df[
            (~self.trials_df["flag_truncated"])
            & (~self.trials_df["flag_glitch"])
            & (self.trials_df["nfl_id"].isin(train_ids))
        ]
        self.drill_stats_: dict[str, dict[str, tuple[float, float]]] = {}
        self.global_stats_: dict[str, tuple[float, float]] = {}

        for m in self.metrics:
            gm = float(valid[m].mean()) if valid[m].notna().any() else 0.0
            gs = float(valid[m].std()) if valid[m].notna().sum() > 1 else 1.0
            if not np.isfinite(gs) or gs < 1e-6:
                gs = 1.0
            self.global_stats_[m] = (gm, gs)
            self.drill_stats_[m] = {}
            for dname, sub in valid.groupby("drill_name")[m]:
                if sub.notna().sum() >= 5 and sub.std() > 1e-6:
                    self.drill_stats_[m][dname] = (float(sub.mean()), float(sub.std()))
                else:
                    self.drill_stats_[m][dname] = (gm, gs)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        target_ids = set(X["nfl_id"])
        valid = self.trials_df[
            (~self.trials_df["flag_truncated"])
            & (~self.trials_df["flag_glitch"])
            & (self.trials_df["nfl_id"].isin(target_ids))
        ].copy()

        z_cols = []
        for m in self.metrics:
            gm, gs = self.global_stats_[m]
            means = valid["drill_name"].map(lambda d: self.drill_stats_[m].get(d, (gm, gs))[0])
            stds = valid["drill_name"].map(lambda d: self.drill_stats_[m].get(d, (gm, gs))[1])
            col_z = f"_dz_{m}"
            valid[col_z] = (valid[m] - means) / stds
            z_cols.append(col_z)

        # 스킬 드릴 우선, 전체 트라이얼 폴백으로 선수별 평균 상대 실력 집계
        agg_all = valid.groupby("nfl_id")[z_cols].mean()
        agg_all.columns = [f"drill_z_{m}_mean" for m in self.metrics]
        agg_all["drill_z_composite"] = agg_all.mean(axis=1)

        merged = X[["nfl_id"]].merge(agg_all, on="nfl_id", how="left").drop(columns=["nfl_id"])
        merged.index = X.index
        return pd.concat([out, merged], axis=1)


class WithinGroupMedianImputer(BaseEstimator, TransformerMixin):
    """Train 포지션 그룹별 중앙값으로 결측치를 대치한다 (전체 중앙값의 포지션 편향 방지)."""

    def __init__(self, value_cols: list[str], group_col: str = "combine_position"):
        self.value_cols = value_cols
        self.group_col = group_col

    def fit(self, X: pd.DataFrame, y=None):
        self.active_cols_ = [c for c in self.value_cols if c in X.columns]
        self.global_medians_ = {}
        self.group_medians_ = {}
        for col in self.active_cols_:
            s = pd.to_numeric(X[col], errors="coerce")
            g_med = float(s.median()) if s.notna().any() else 0.0
            self.global_medians_[col] = g_med
            for grp, sub in s.groupby(X[self.group_col]):
                self.group_medians_[(col, grp)] = float(sub.median()) if sub.notna().any() else g_med
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        pos_arr = X[self.group_col].to_numpy()
        for col in self.active_cols_:
            vals = pd.to_numeric(out[col], errors="coerce").to_numpy(dtype=np.float64)
            na_mask = ~np.isfinite(vals)
            if na_mask.any():
                g_med = self.global_medians_[col]
                fills = np.array([self.group_medians_.get((col, g), g_med) for g in pos_arr[na_mask]])
                vals[na_mask] = fills
            out[col] = vals
        return out


class IsolationForestAnomalyAdder(BaseEstimator, TransformerMixin):
    """대치·스케일링된 수치 행렬에 Isolation Forest 이상치 점수를 보조 피처로 추가한다."""

    def __init__(self, contamination: float = config.IFOREST_CONTAMINATION, random_state: int = config.SEED):
        self.contamination = contamination
        self.random_state = random_state

    def fit(self, X: np.ndarray, y=None):
        self.iso_ = IsolationForest(
            n_estimators=200,
            contamination=self.contamination,
            random_state=self.random_state,
        )
        self.iso_.fit(X)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        score = (-self.iso_.decision_function(X)).reshape(-1, 1)
        return np.hstack([X, score])


class _ColumnSelector(BaseEstimator, TransformerMixin):
    """DataFrame에서 지정한 수치·범주 컬럼만 추출하는 내부 헬퍼."""

    def __init__(self, num_cols: list[str], cat_cols: list[str]):
        self.num_cols = num_cols
        self.cat_cols = cat_cols

    def fit(self, X: pd.DataFrame, y=None):
        # Z-score / Drill-Z 로 동적 추가된 컬럼도 자동 포함
        extra_num = [
            c for c in X.columns
            if c.startswith("zpos_") or c.startswith("zcoh_") or c.startswith("drill_z_")
        ]
        self.selected_num_ = list(dict.fromkeys([c for c in self.num_cols if c in X.columns] + extra_num))
        self.selected_cat_ = [c for c in self.cat_cols if c in X.columns]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X[self.selected_num_ + self.selected_cat_].copy()


class _MatrixBlockPreprocessor(BaseEstimator, TransformerMixin):
    """[단계: 결측치 처리 + 인코딩 + 스케일링]을 누수 없이 일괄 수행하는 핵심 블록."""

    def __init__(
        self,
        num_cols: list[str],
        cat_cols: list[str],
        mode: Literal["linear", "tree"] = "linear",
        imputer: Literal["group_median", "knn", "mice", "none"] = "group_median",
        scaler: Literal["standard", "robust", "yeo_johnson", "none"] = "robust",
        add_iso_score: bool = False,
    ):
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        self.mode = mode
        self.imputer = imputer
        self.scaler = scaler
        self.add_iso_score = add_iso_score

    def _make_scaler(self):
        if self.scaler == "standard":
            return StandardScaler()
        if self.scaler == "robust":
            return RobustScaler()
        if self.scaler == "yeo_johnson":
            return PowerTransformer(method="yeo-johnson", standardize=True)
        return "passthrough"

    def fit(self, X: pd.DataFrame, y=None):
        self.selector_ = _ColumnSelector(self.num_cols, self.cat_cols).fit(X)
        self.num_cols_ = self.selector_.selected_num_
        self.cat_cols_ = self.selector_.selected_cat_

        X_sel = self.selector_.transform(X)
        self.group_imp_ = None
        if self.imputer == "group_median":
            self.group_imp_ = WithinGroupMedianImputer(self.num_cols_, group_col="combine_position").fit(X_sel)
            X_sel = self.group_imp_.transform(X_sel)

        # 범주형 One-Hot 인코더
        self.ohe_ = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        X_cat = self.ohe_.fit_transform(X_sel[self.cat_cols_]) if self.cat_cols_ else np.empty((len(X_sel), 0))
        X_num = X_sel[self.num_cols_].to_numpy(dtype=np.float64)

        # 수치형 스케일러 / 다변량 대치(KNN, MICE는 포지션 원핫 정보를 함께 활용해 대치!)
        self.pre_scaler_ = None
        self.multi_imp_ = None
        self.final_scaler_ = self._make_scaler()
        self.fallback_imp_ = None

        if self.mode == "tree" and self.imputer == "none":
            X_num_proc = X_num
        elif self.imputer == "knn":
            self.pre_scaler_ = RobustScaler().fit(X_num)
            X_num_sc = self.pre_scaler_.transform(X_num)
            X_joint = np.hstack([X_num_sc, X_cat * 2.0])  # 포지션 가중치 부여로 같은 포지션 이웃 우선
            self.multi_imp_ = KNNImputer(n_neighbors=7, weights="distance").fit(X_joint)
            X_num_imp = self.pre_scaler_.inverse_transform(self.multi_imp_.transform(X_joint)[:, : X_num.shape[1]])
            X_num_proc = self.final_scaler_.fit_transform(X_num_imp) if self.final_scaler_ != "passthrough" else X_num_imp
        elif self.imputer == "mice":
            # 스케일 차이(예: 파워 5000 vs 10yd 1.6)로 인한 수렴 불안정을 막기 위해 사전 RobustScaler 적용
            self.pre_scaler_ = RobustScaler().fit(X_num)
            X_num_sc = self.pre_scaler_.transform(X_num)
            X_joint = np.hstack([X_num_sc, X_cat])
            self.multi_imp_ = IterativeImputer(
                estimator=Ridge(alpha=10.0),
                n_nearest_features=12,
                max_iter=10,
                tol=1e-2,
                random_state=config.SEED,
                initial_strategy="median",
            ).fit(X_joint)
            X_num_imp = self.pre_scaler_.inverse_transform(self.multi_imp_.transform(X_joint)[:, : X_num.shape[1]])
            X_num_proc = self.final_scaler_.fit_transform(X_num_imp) if self.final_scaler_ != "passthrough" else X_num_imp
        else:
            # group_median 또는 simple fallback
            self.fallback_imp_ = SimpleImputer(strategy="median").fit(X_num)
            X_num_imp = self.fallback_imp_.transform(X_num)
            X_num_proc = self.final_scaler_.fit_transform(X_num_imp) if self.final_scaler_ != "passthrough" else X_num_imp

        X_mat = np.hstack([X_num_proc, X_cat])
        self.iso_ = None
        if self.add_iso_score and not np.isnan(X_mat).any():
            self.iso_ = IsolationForestAnomalyAdder().fit(X_mat)

        cat_names = list(self.ohe_.get_feature_names_out(self.cat_cols_)) if self.cat_cols_ else []
        self.feature_names_out_ = list(self.num_cols_) + cat_names + (["iso_anomaly_score"] if self.iso_ else [])
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        X_sel = self.selector_.transform(X)
        if self.group_imp_ is not None:
            X_sel = self.group_imp_.transform(X_sel)

        X_cat = self.ohe_.transform(X_sel[self.cat_cols_]) if self.cat_cols_ else np.empty((len(X_sel), 0))
        X_num = X_sel[self.num_cols_].to_numpy(dtype=np.float64)

        if self.mode == "tree" and self.imputer == "none":
            X_num_proc = X_num
        elif self.imputer == "knn":
            X_num_sc = self.pre_scaler_.transform(X_num)
            X_joint = np.hstack([X_num_sc, X_cat * 2.0])
            X_num_imp = self.pre_scaler_.inverse_transform(self.multi_imp_.transform(X_joint)[:, : X_num.shape[1]])
            X_num_proc = self.final_scaler_.transform(X_num_imp) if self.final_scaler_ != "passthrough" else X_num_imp
        elif self.imputer == "mice":
            X_num_sc = self.pre_scaler_.transform(X_num)
            X_joint = np.hstack([X_num_sc, X_cat])
            X_num_imp = self.pre_scaler_.inverse_transform(self.multi_imp_.transform(X_joint)[:, : X_num.shape[1]])
            X_num_proc = self.final_scaler_.transform(X_num_imp) if self.final_scaler_ != "passthrough" else X_num_imp
        else:
            X_num_imp = self.fallback_imp_.transform(X_num)
            X_num_proc = self.final_scaler_.transform(X_num_imp) if self.final_scaler_ != "passthrough" else X_num_imp

        X_mat = np.hstack([X_num_proc, X_cat])
        if self.iso_ is not None:
            X_mat = self.iso_.transform(X_mat)
        return X_mat

    def get_feature_names_out(self) -> list[str]:
        return list(self.feature_names_out_)


# =====================================================================
# [데이터: 컴바인+경기 통합] [단계: 최종 재사용 파이프라인 빌더]
# =====================================================================
def build_preprocessor(
    feature_set: Literal["FS-Trad", "FS-Phys", "FS-Biomech", "FS-Full"] = "FS-Biomech",
    mode: Literal["linear", "tree"] = "linear",
    imputer: Literal["group_median", "knn", "mice", "none"] | None = None,
    scaler: Literal["standard", "robust", "yeo_johnson", "none"] | None = None,
    add_within_z: bool = True,
    add_cohort_z: bool = True,
    trials_df: pd.DataFrame | None = None,
    add_iso_score: bool = False,
) -> Pipeline:
    """선형 계열(`mode='linear'`) 및 트리 계열(`mode='tree'`) 전처리 Pipeline을 생성한다."""
    num_cols, cat_cols = get_feature_columns(feature_set)
    if imputer is None:
        imputer = "group_median" if mode == "linear" else "none"
    if scaler is None:
        scaler = "robust" if mode == "linear" else "none"

    steps: list[tuple[str, BaseEstimator]] = []
    if trials_df is not None and feature_set in ("FS-Biomech", "FS-Full"):
        steps.append(("drill_z", DrillZScoreFeaturizer(trials_df=trials_df)))
    if add_within_z:
        z_cols = [c for c in ZSCORE_CORE_COLS if c in num_cols]
        if z_cols:
            steps.append((
                "group_z",
                WithinGroupZScoreAdder(value_cols=z_cols, group_col="combine_position", add_cohort_z=add_cohort_z),
            ))

    steps.append((
        "matrix",
        _MatrixBlockPreprocessor(
            num_cols=num_cols,
            cat_cols=cat_cols,
            mode=mode,
            imputer=imputer,
            scaler=scaler,
            add_iso_score=add_iso_score,
        ),
    ))
    return Pipeline(steps)


# =====================================================================
# [데이터: 컴바인+경기 통합] [단계: CV 비교 실험 (결측 / 스케일링 / 피처세트)]
#   - 이미 배운 RidgeCV + 5×5 층화 CV로 Train(408명) 안에서만 공정 비교
#   - 전체 R²뿐 아니라 "포지션 평균 제거 잔차 R²(within-group R²)"를 반드시 함께 보고!
# =====================================================================
def _within_group_r2(y_true: np.ndarray, y_pred: np.ndarray, groups: np.ndarray) -> float:
    """포지션 그룹 평균을 제거한 순수 포지션 내부 R² (심슨의 역설 통제 지표)."""
    df_e = pd.DataFrame({"y": y_true, "p": y_pred, "g": groups})
    y_res = df_e["y"] - df_e.groupby("g")["y"].transform("mean")
    p_res = df_e["p"] - df_e.groupby("g")["p"].transform("mean")
    return float(r2_score(y_res, p_res))


def _relative_or_raw_target(
    X_fit: pd.DataFrame,
    frames: list[pd.DataFrame],
    target_col: str,
) -> list[np.ndarray]:
    """원본 타깃이면 그대로, 상대 기준 타깃이면 X_fit(학습 데이터)으로만 기준을 잡아 각 frame의 라벨을 만든다."""
    if relative_label_base(target_col) is None:
        return [f[target_col].to_numpy(dtype=np.float64) for f in frames]
    enc = RelativeTargetEncoder().fit(X_fit)
    return [enc.transform(f)[target_col].to_numpy(dtype=np.float64) for f in frames]


def evaluate_pipeline_cv(
    train_df: pd.DataFrame,
    target_col: str,
    preprocessor_kwargs: dict,
    min_exposure_only: bool = True,
) -> dict:
    """Train 데이터 내부에서 5×5 Repeated Stratified CV로 RidgeCV 성능을 평가한다.

    - 상대 기준 타깃(예: `g_smax_top5mean_cohort_z`)은 fold마다 학습 fold로만 기준을 fit 한다.
    - 반환값에는 요약(평균·표준편차)과 함께 fold별 점수 배열(`*_folds`)이 들어 있어,
      설정끼리 같은 fold 점수를 짝지어 비교할 수 있다.
    """
    base_col = relative_label_base(target_col) or target_col
    sub = train_df[train_df[base_col].notna()].copy()
    if min_exposure_only and "low_exposure" in sub.columns:
        sub = sub[~sub["low_exposure"]].copy()
    sub = sub.reset_index(drop=True)

    pos = sub["combine_position"].to_numpy()
    rskf = RepeatedStratifiedKFold(
        n_splits=config.CV_N_SPLITS,
        n_repeats=config.CV_N_REPEATS,
        random_state=config.SEED,
    )

    r2_all_list, r2_within_list, mae_list = [], [], []
    alphas = np.logspace(-2, 4, 25)

    for tr_idx, va_idx in rskf.split(sub, pos):
        X_tr, X_va = sub.iloc[tr_idx], sub.iloc[va_idx]
        y_tr, y_va = _relative_or_raw_target(X_tr, [X_tr, X_va], target_col)
        keep_tr, keep_va = np.isfinite(y_tr), np.isfinite(y_va)
        X_tr, y_tr = X_tr[keep_tr], y_tr[keep_tr]
        X_va, y_va = X_va[keep_va], y_va[keep_va]

        pipe = Pipeline([
            ("prep", build_preprocessor(**preprocessor_kwargs)),
            ("model", RidgeCV(alphas=alphas)),
        ])
        pipe.fit(X_tr, y_tr)
        pred = pipe.predict(X_va)

        r2_all_list.append(r2_score(y_va, pred))
        r2_within_list.append(_within_group_r2(y_va, pred, X_va["combine_position"].to_numpy()))
        mae_list.append(mean_absolute_error(y_va, pred))

    return {
        "n_train_eval": len(sub),
        "r2_overall_mean": round(float(np.mean(r2_all_list)), 4),
        "r2_overall_std": round(float(np.std(r2_all_list)), 4),
        "r2_within_pos_mean": round(float(np.mean(r2_within_list)), 4),
        "r2_within_pos_std": round(float(np.std(r2_within_list)), 4),
        "mae_mean": round(float(np.mean(mae_list)), 4),
        "r2_overall_folds": np.asarray(r2_all_list),
        "r2_within_pos_folds": np.asarray(r2_within_list),
        "mae_folds": np.asarray(mae_list),
    }


# =====================================================================
# [단계: 비교 판정] 같은 fold 점수를 짝지은 비교 (Nadeau–Bengio 보정 표준오차)
#   - fold마다 d_i = 점수_i(후보) − 점수_i(기준) 을 계산하면 fold 난이도 차이가 상쇄된다.
#   - 반복 k-fold는 fold끼리 학습 데이터가 겹쳐 단순 표준오차가 과소추정되므로
#     보정 분산 (1/J + n_test/n_train) × var(d) 를 쓴다. (J = fold 수 25, n_test/n_train = 1/4)
#   - 판정: delta_mean > 2 × SE → "우위", < −2 × SE → "열위", 그 외 → "차이 없음"
# =====================================================================
_FOLD_KEYS = ("r2_overall_folds", "r2_within_pos_folds", "mae_folds")


def _paired_vs_baseline(
    results: dict[str, dict],
    baseline: str,
    metric: str = "r2_within_pos_folds",
) -> dict[str, dict]:
    """설정별 fold 점수를 기준 설정과 짝지어 차이 통계를 계산한다."""
    k = config.CV_N_SPLITS
    test_train_ratio = 1.0 / (k - 1)
    base = results[baseline][metric]
    out = {}
    for name, res in results.items():
        d = res[metric] - base
        n = len(d)
        if name == baseline:
            out[name] = {"delta_mean": 0.0, "delta_std": 0.0, "wins": "-",
                         "delta_se_corrected": 0.0, "verdict": "기준"}
            continue
        var_d = float(np.var(d, ddof=1))
        se = float(np.sqrt((1.0 / n + test_train_ratio) * var_d))
        dm = float(np.mean(d))
        if dm > 2 * se:
            verdict = "우위"
        elif dm < -2 * se:
            verdict = "열위"
        else:
            verdict = "차이 없음"
        out[name] = {
            "delta_mean": round(dm, 4),
            "delta_std": round(float(np.std(d, ddof=1)), 4),
            "wins": f"{int((d > 0).sum())}/{n}",
            "delta_se_corrected": round(se, 4),
            "verdict": verdict,
        }
    return out


def _fold_scores_long(comparison: str, target: str, config_name: str, res: dict) -> pd.DataFrame:
    """fold별 점수를 long format으로 변환한다 (보고서 박스플롯용)."""
    n = len(res["r2_within_pos_folds"])
    idx = np.arange(n)
    return pd.DataFrame({
        "comparison": comparison,
        "target": target,
        "config": config_name,
        "repeat": idx // config.CV_N_SPLITS,
        "fold": idx % config.CV_N_SPLITS,
        "r2_overall": res["r2_overall_folds"],
        "r2_within_pos": res["r2_within_pos_folds"],
        "mae": res["mae_folds"],
    })


def _run_comparison(
    comparison: str,
    target_col: str,
    configs: dict[str, dict],
    baseline: str,
    train_df: pd.DataFrame,
    label_col: str,
) -> pd.DataFrame:
    """여러 전처리 설정을 같은 fold로 평가하고, 기준 대비 짝지은 비교 열을 붙인 표를 반환한다.

    반환 DataFrame의 `attrs["fold_scores"]`에 fold별 원점수(long format)가 들어 있다.
    """
    results = {name: evaluate_pipeline_cv(train_df, target_col, kw) for name, kw in configs.items()}
    paired = _paired_vs_baseline(results, baseline)
    rows, folds = [], []
    for name, res in results.items():
        summary = {k: v for k, v in res.items() if k not in _FOLD_KEYS}
        rows.append({"target": target_col, label_col: name, **summary, **paired[name]})
        folds.append(_fold_scores_long(comparison, target_col, name, res))
    df_res = pd.DataFrame(rows)
    df_res.attrs["fold_scores"] = pd.concat(folds, ignore_index=True)
    return df_res


def compare_imputers_cv(
    train_df: pd.DataFrame,
    trials_df: pd.DataFrame,
    target_col: str = "g_smax_top5mean",
) -> pd.DataFrame:
    """[단계: 결측 방식 비교] group_median(기준) vs knn vs mice — 같은 25개 fold 짝지은 비교."""
    configs = {
        imp: dict(feature_set="FS-Biomech", mode="linear", imputer=imp,
                  scaler="robust", trials_df=trials_df)
        for imp in ("group_median", "knn", "mice")
    }
    df_res = _run_comparison("imputer", target_col, configs, "group_median", train_df, "imputer")
    print("[확인된 결과] 결측치 처리 방식 비교 (5×5 CV, Ridge, 기준=group_median):")
    print(df_res.to_string(index=False))
    return df_res


def compare_scalers_cv(
    train_df: pd.DataFrame,
    trials_df: pd.DataFrame,
    target_col: str = "g_smax_top5mean",
    imputer: str = "group_median",
) -> pd.DataFrame:
    """[단계: 스케일링 비교] robust(기준) vs standard vs yeo_johnson — 같은 25개 fold 짝지은 비교."""
    configs = {
        sc: dict(feature_set="FS-Biomech", mode="linear", imputer=imputer,
                 scaler=sc, trials_df=trials_df)
        for sc in ("robust", "standard", "yeo_johnson")
    }
    df_res = _run_comparison("scaler", target_col, configs, "robust", train_df, "scaler")
    print("[확인된 결과] 스케일링·분포변환 비교 (5×5 CV, Ridge, 기준=robust):")
    print(df_res.to_string(index=False))
    return df_res


def compare_feature_sets_cv(
    train_df: pd.DataFrame,
    trials_df: pd.DataFrame,
    targets: tuple[str, ...] = ("g_smax_top5mean", "g_smax_top5mean_cohort_z"),
) -> pd.DataFrame:
    """[단계: 피처 세트 비교] 타깃마다 FS-Trad(기준) → FS-Full 짝지은 비교.

    코호트 상대 타깃은 fold마다 학습 fold로만 기준을 잡는다 (evaluate_pipeline_cv 참고).
    """
    tables = []
    for tgt in targets:
        configs = {
            fs: dict(feature_set=fs, mode="linear", imputer="group_median",
                     scaler="robust", trials_df=trials_df)
            for fs in ("FS-Trad", "FS-Phys", "FS-Biomech", "FS-Full")
        }
        tables.append(_run_comparison("feature_set", tgt, configs, "FS-Trad", train_df, "feature_set"))
    fold_scores = pd.concat([t.attrs["fold_scores"] for t in tables], ignore_index=True)
    for t in tables:
        t.attrs = {}  # attrs 안의 DataFrame 때문에 pd.concat 이 attrs 비교에서 실패하지 않도록 비운다
    df_res = pd.concat(tables, ignore_index=True)
    df_res.attrs["fold_scores"] = fold_scores
    print("[확인된 결과] 피처 세트별 성능 비교 (기준=FS-Trad, 절대 타깃 vs 시즌 코호트 상대 타깃):")
    print(df_res.to_string(index=False))
    return df_res


def inspect_pca_and_lasso(
    train_df: pd.DataFrame,
    trials_df: pd.DataFrame,
    target_col: str = "g_smax_top5mean_cohort_z",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """[단계: 차원·피처 점검] 이미 배운 PCA 설명분산 및 Lasso/ElasticNet 생존 피처를 확인한다.

    상대 기준 타깃이면 Train 데이터로만 기준을 잡아 라벨을 만든다 (테스트셋 미사용).
    """
    base_col = relative_label_base(target_col) or target_col
    sub = train_df[train_df[base_col].notna() & (~train_df["low_exposure"])].reset_index(drop=True)
    (y,) = _relative_or_raw_target(sub, [sub], target_col)
    keep = np.isfinite(y)
    sub, y = sub[keep].reset_index(drop=True), y[keep]

    prep = build_preprocessor(
        feature_set="FS-Biomech", mode="linear", imputer="group_median",
        scaler="robust", trials_df=trials_df, add_iso_score=True,
    )
    X_mat = prep.fit_transform(sub)
    feat_names = prep.named_steps["matrix"].get_feature_names_out()

    # 1) PCA 누적 설명분산
    pca = PCA(random_state=config.SEED).fit(X_mat)
    cum_var = np.cumsum(pca.explained_variance_ratio_)
    pca_tbl = pd.DataFrame({
        "n_components": np.arange(1, len(cum_var) + 1),
        "explained_var": pca.explained_variance_ratio_.round(4),
        "cum_explained_var": cum_var.round(4),
    })
    n80 = int(np.searchsorted(cum_var, 0.80) + 1)
    n90 = int(np.searchsorted(cum_var, 0.90) + 1)

    # 2) LassoCV & ElasticNetCV 계수 (코호트 상대 타깃 기준 생존 핵심 피처)
    lasso = LassoCV(cv=5, random_state=config.SEED, max_iter=10000).fit(X_mat, y)
    enet = ElasticNetCV(
        l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9], cv=5, random_state=config.SEED, max_iter=10000
    ).fit(X_mat, y)

    coef_tbl = pd.DataFrame({
        "feature": feat_names,
        "lasso_coef": lasso.coef_,
        "enet_coef": enet.coef_,
    })
    coef_tbl["abs_enet"] = coef_tbl["enet_coef"].abs()
    coef_tbl = coef_tbl.sort_values("abs_enet", ascending=False).reset_index(drop=True)

    print(
        f"[확인된 결과] PCA 점검: 총 {X_mat.shape[1]}개 피처 중 "
        f"PC {n80}개로 80%, PC {n90}개로 90% 분산 설명 | "
        f"Lasso 생존 피처={(coef_tbl['lasso_coef'].abs() > 1e-4).sum()}개, "
        f"ElasticNet 생존 피처={(coef_tbl['enet_coef'].abs() > 1e-4).sum()}개"
    )
    return pca_tbl, coef_tbl


def run_stage4(save_tables: bool = True) -> dict[str, pd.DataFrame]:
    """Stage 4 통합·분할·비교 실험 및 누수 무결성 검증 전체 실행."""
    config.ensure_dirs()
    master = build_master_table(save=True)
    train_df, test_df = create_or_load_test_split(master, overwrite=False)
    trials = pd.read_parquet(config.PROCESSED_FILES["combine_trials"])

    # 타깃 누수 검증: 테스트 선수의 원본 타깃을 섞어도 Train 상대 라벨이 바뀌지 않아야 한다
    check_relative_labels_ignore_holdout(train_df, test_df)
    print("[확인된 결과] 상대 기준 타깃 교란 테스트 통과: 테스트 정답을 바꿔도 Train 라벨 불변")

    # 피처 누수 무결성 검증: Train fit -> Test transform 시 NaN/Inf 0건
    prep_lin = build_preprocessor(
        feature_set="FS-Biomech", mode="linear", imputer="group_median",
        scaler="robust", trials_df=trials, add_iso_score=True,
    )
    X_tr_mat = prep_lin.fit_transform(train_df)
    X_te_mat = prep_lin.transform(test_df)
    assert np.isfinite(X_tr_mat).all() and np.isfinite(X_te_mat).all(), "Linear 파이프라인에 NaN/Inf 존재!"

    prep_tree = build_preprocessor(
        feature_set="FS-Biomech", mode="tree", imputer="none",
        scaler="none", trials_df=trials,
    )
    X_tr_tree = prep_tree.fit_transform(train_df)
    X_te_tree = prep_tree.transform(test_df)
    print(
        f"[확인된 결과] 파이프라인 무결성 통과: Linear={X_tr_mat.shape}/{X_te_mat.shape} (NaN 0건) | "
        f"Tree={X_tr_tree.shape}/{X_te_tree.shape} (네이티브 NaN 보존)"
    )

    imp_tbl = compare_imputers_cv(train_df, trials)
    sc_tbl = compare_scalers_cv(train_df, trials)
    fs_tbl = compare_feature_sets_cv(train_df, trials)
    pca_tbl, coef_tbl = inspect_pca_and_lasso(train_df, trials)

    if save_tables:
        imp_tbl.to_csv(config.TABLES_DIR / "4_imputer_comparison.csv", index=False)
        sc_tbl.to_csv(config.TABLES_DIR / "4_scaler_comparison.csv", index=False)
        fs_tbl.to_csv(config.TABLES_DIR / "4_feature_set_comparison.csv", index=False)
        coef_tbl.head(30).to_csv(config.TABLES_DIR / "4_top30_enet_lasso_features.csv", index=False)
        fold_scores = pd.concat(
            [t.attrs["fold_scores"] for t in (imp_tbl, sc_tbl, fs_tbl)], ignore_index=True
        )
        fold_scores.to_csv(config.TABLES_DIR / "4_cv_fold_scores.csv", index=False)
        print("[확인된 결과] Stage 4 비교표 5종 저장 완료 (fold별 원점수 4_cv_fold_scores.csv 포함) -> reports/tables/")

    return {
        "master": master,
        "train": train_df,
        "test": test_df,
        "imputer_cmp": imp_tbl,
        "scaler_cmp": sc_tbl,
        "feature_set_cmp": fs_tbl,
        "pca": pca_tbl,
        "coefs": coef_tbl,
    }


if __name__ == "__main__":
    run_stage4(save_tables=True)
