"""[컴바인 단독] 전처리 · 이상치 정제 · 결측(포기 선수) 구제 · 생체역학 피처 추출.

이 모듈은 `combine_results.csv`, `combine_tracking.csv`, `players.csv`(드래프트 전 인적 정보만)를
입력으로 받아 아래 4개의 Parquet 산출물을 생성한다:
1. combine_trials.parquet          : 트라이얼(6,310개) 단위 3축 생체역학 운동학 지표
2. combine_trial_flags.parquet     : 트라이얼별 품질 플래그(글리치·잘림·갭)
3. combine_forty_curves.parquet    : 40야드 달리기 5초/50점 리샘플링 속도 곡선
4. combine_player_features.parquet : 선수(510명) 단위 전통 + 물리파생 + 결측플래그 + 트래킹 구제 피처

주의:
- `draft_round`, `draft_overall_pick`, `draft_tier`는 드래프트 전 스카우팅 원칙에 따라 절대 포함하지 않는다.
- `combine_tracking.csv`의 `dis` 컬럼은 실제 `(x, y)` 이동거리의 ~53%만 기록되는 오염이 확인되었으므로
  사용하지 않고 `hypot(dx, dy)`로 직접 재계산한다.
"""
from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from src import config


# =====================================================================
# [데이터: 컴바인 단독] [단계: 데이터 로딩] 원본 CSV 로드 (드래프트 결과 제외)
# =====================================================================
def load_combine_raw() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """컴바인 결과, 컴바인 10Hz 트래킹, 드래프트 전 인적 정보만 로딩한다."""
    cr = pd.read_csv(config.RAW_FILES["combine_results"])
    ct = pd.read_csv(config.RAW_FILES["combine_tracking"])
    # 드래프트 순위 컬럼은 의도적으로 읽지 않음 (드래프트 전 예측 원칙)
    pl_cols = [
        "nfl_id", "display_name", "draft_year", "nfl_position",
        "birth_date", "college_name", "college_conference",
    ]
    pl = pd.read_csv(config.RAW_FILES["players"], usecols=pl_cols)
    print(
        f"[확인된 결과] 로딩 완료: combine_results={cr.shape}, "
        f"combine_tracking={ct.shape}, players(사전정보)={pl.shape}"
    )
    return cr, ct, pl


# =====================================================================
# [데이터: 컴바인 단독] [단계: 데이터 정제 & 이상치 탐지·처리]
#   - PLAYER 필터링, TRIAL_KEY 정렬
#   - 오염된 dis 컬럼 폐기 -> (x, y) 유클리드 거리 재계산
#   - 글리치 프레임 탐지 및 3축 가속도(추진 / 제동 / 구심) 분해
# =====================================================================
def clean_and_compute_frame_kinematics(ct: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """프레임 단위 정제 및 3축 생체역학(추진/제동/구심가속)을 벡터화 계산한다."""
    ply = ct[ct["entity_type"] == "PLAYER"].copy()
    ply["time"] = pd.to_datetime(ply["time"])
    ply = ply.sort_values(config.TRIAL_KEY + ["time"]).reset_index(drop=True)

    # 트라이얼 경계 마스크: 같은 트라이얼 내 직전 프레임과 연결되는 행만 True
    same_trial = np.ones(len(ply), dtype=bool)
    same_trial[0] = False
    for col in config.TRIAL_KEY:
        vals = ply[col].to_numpy()
        same_trial[1:] &= vals[1:] == vals[:-1]

    # 시간 간격 dt (초)
    t_ns = ply["time"].astype("int64").to_numpy()
    dt = np.full(len(ply), np.nan, dtype=np.float64)
    dt[1:] = (t_ns[1:] - t_ns[:-1]) / 1e9
    dt[~same_trial] = np.nan

    # (x, y) 기반 실제 이동거리 step_dist 재계산 (오염된 dis 컬럼 폐기!)
    x = ply["x"].to_numpy(dtype=np.float64)
    y = ply["y"].to_numpy(dtype=np.float64)
    dx = np.full(len(ply), 0.0, dtype=np.float64)
    dy = np.full(len(ply), 0.0, dtype=np.float64)
    dx[1:] = np.where(same_trial[1:], x[1:] - x[:-1], 0.0)
    dy[1:] = np.where(same_trial[1:], y[1:] - y[:-1], 0.0)
    step_dist = np.hypot(dx, dy)

    # 오염된 dis 대비 실제 step_dist 진단 통계
    moving_mask = same_trial & (step_dist > 0.2)
    raw_dis = ply["dis"].to_numpy(dtype=np.float64)
    dis_ratio_median = float(np.nanmedian(raw_dis[moving_mask] / step_dist[moving_mask]))

    # 프레임 단위 물리 글리치 탐지 (D5, D6)
    s = ply["s"].to_numpy(dtype=np.float64)
    a = ply["a"].to_numpy(dtype=np.float64)
    flag_glitch = (
        (s > config.SPEED_GLITCH_YDS)
        | (a > config.ACCEL_GLITCH_YDS2)
        | (a < 0.0)
    )

    # 부호 있는 종방향 가속도 ds/dt (추진 a_tan_pos vs 제동 a_tan_neg 분리)
    ds_dt = np.full(len(ply), np.nan, dtype=np.float64)
    valid_dt = same_trial & (dt > 0.05) & (dt <= config.MAX_FRAME_GAP_SEC) & (~flag_glitch)
    ds_dt[1:] = np.where(valid_dt[1:], (s[1:] - s[:-1]) / np.where(valid_dt[1:], dt[1:], 1.0), np.nan)
    ds_dt[np.abs(ds_dt) > config.DECEL_GLITCH_YDS2] = np.nan

    a_tan_pos = np.where(np.isfinite(ds_dt), np.maximum(ds_dt, 0.0), np.nan)
    a_tan_neg = np.where(np.isfinite(ds_dt), np.maximum(-ds_dt, 0.0), np.nan)

    # 방향각 변화 d_dir (359도 -> 1도 wrap-around 보정) & 구심/횡가속 a_lat = s * |omega|
    direction = ply["dir"].to_numpy(dtype=np.float64)
    d_dir = np.full(len(ply), np.nan, dtype=np.float64)
    raw_ddir = ((direction[1:] - direction[:-1] + 180.0) % 360.0) - 180.0
    # 정지/저속(s < 1.5 yd/s)에서는 방향각 노이즈가 크므로 구심가속·COD 계산에서 제외
    moving_turn = valid_dt & (s >= 1.5)
    d_dir[1:] = np.where(moving_turn[1:], raw_ddir, np.nan)
    omega_rad = np.abs(np.deg2rad(d_dir)) / np.where(valid_dt, dt, 1.0)
    a_lat = np.where(np.isfinite(omega_rad), np.minimum(s * omega_rad, config.ACCEL_GLITCH_YDS2), np.nan)

    # 저크(Jerk): |d(ds_dt)/dt| (폭발력 전환 속도)
    jerk = np.full(len(ply), np.nan, dtype=np.float64)
    valid_jerk = valid_dt.copy()
    valid_jerk[1:] &= np.isfinite(ds_dt[1:]) & np.isfinite(ds_dt[:-1])
    jerk[1:] = np.where(
        valid_jerk[1:],
        np.abs(ds_dt[1:] - ds_dt[:-1]) / np.where(valid_jerk[1:], dt[1:], 1.0),
        np.nan,
    )
    jerk[jerk > 120.0] = np.nan

    ply["dt"] = dt
    ply["step_dist"] = step_dist
    ply["flag_glitch"] = flag_glitch
    ply["ds_dt"] = ds_dt
    ply["a_tan_pos"] = a_tan_pos
    ply["a_tan_neg"] = a_tan_neg
    ply["d_dir"] = d_dir
    ply["abs_d_dir"] = np.abs(d_dir)
    ply["a_lat"] = a_lat
    ply["jerk"] = jerk

    stats = {
        "n_player_frames": len(ply),
        "n_glitch_frames": int(flag_glitch.sum()),
        "raw_dis_to_xy_median_ratio": round(dis_ratio_median, 4),
    }
    print(
        f"[확인된 결과] PLAYER 프레임={stats['n_player_frames']:,} | "
        f"글리치 프레임={stats['n_glitch_frames']} | "
        f"원본 dis / 실제(x,y) 거리 중앙비={stats['raw_dis_to_xy_median_ratio']} (dis 폐기 완료)"
    )
    return ply, stats


# =====================================================================
# [데이터: 컴바인 단독] [단계: 피처 추가] 40야드 Samozino-Morin F-V 곡선 피팅
#   수식: v(t) = v_max * (1 - exp(-t / tau))
#   - v_max: 이론 최고속도, tau: 가속 시상수, a0 = v_max / tau: 초기 폭발력
#   - pmax_rel = 0.25 * v_max * a0: 단위체중당 최대 기계적 파워
# =====================================================================
def _fv_exp_model(t: np.ndarray, v_max: float, tau: float) -> np.ndarray:
    return v_max * (1.0 - np.exp(-t / tau))


def _fit_single_forty_trial(
    t_sec: np.ndarray,
    s_arr: np.ndarray,
    step_dist: np.ndarray,
) -> tuple[dict[str, float], np.ndarray]:
    """단일 40야드 트라이얼에서 정밀 스플릿, F-V 파라미터, 50점 속도 곡선을 추출한다."""
    curve_grid = np.linspace(0.0, config.FORTY_CURVE_SECONDS, config.FORTY_CURVE_POINTS)
    nan_curve = np.full(config.FORTY_CURVE_POINTS, np.nan, dtype=np.float64)
    out = {
        "t_5yd": np.nan, "t_10yd": np.nan, "t_20yd": np.nan,
        "t_30yd": np.nan, "t_40yd": np.nan, "fly_10_30_speed": np.nan,
        "fv_vmax": np.nan, "fv_tau": np.nan, "fv_a0": np.nan,
        "fv_pmax_rel": np.nan, "fv_r2": np.nan, "late_speed_retention": np.nan,
    }

    onset_idx = np.flatnonzero(s_arr >= config.MOTION_ONSET_SPEED)
    if len(onset_idx) == 0:
        return out, nan_curve

    i0 = int(onset_idx[0])
    t_rel = t_sec[i0:] - t_sec[i0]
    s_rel = s_arr[i0:]
    # onset 시점부터의 실제 (x, y) 누적거리
    d_rel = np.cumsum(step_dist[i0:]) - step_dist[i0]

    if len(t_rel) < 10 or d_rel[-1] < 20.0:
        return out, nan_curve

    # 1) (x, y) 누적거리 기반 정밀 구간 스플릿 선형 보간
    for yd in (5.0, 10.0, 20.0, 30.0, 40.0):
        if d_rel[-1] >= yd:
            out[f"t_{int(yd)}yd"] = float(np.interp(yd, d_rel, t_rel))
    if np.isfinite(out["t_10yd"]) and np.isfinite(out["t_30yd"]) and out["t_30yd"] > out["t_10yd"]:
        out["fly_10_30_speed"] = float(20.0 / (out["t_30yd"] - out["t_10yd"]))

    # 2) 후반 속도 유지율 (최고속도 대비 종료 전 0.5초 평균 속도 비율)
    peak_idx = int(np.argmax(s_rel))
    s_peak = float(s_rel[peak_idx])
    if s_peak > 0.0 and len(s_rel) >= 6:
        out["late_speed_retention"] = float(np.mean(s_rel[-5:]) / s_peak)

    # 3) Samozino-Morin Force-Velocity 지수 곡선 피팅 (onset -> peak 구간)
    fit_end = max(peak_idx + 1, min(len(t_rel), 35))
    t_fit = t_rel[:fit_end]
    s_fit = s_rel[:fit_end]
    if len(t_fit) >= 8:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                popt, _ = curve_fit(
                    _fv_exp_model,
                    t_fit,
                    s_fit,
                    p0=[max(s_peak, 8.0), 1.1],
                    bounds=([5.0, 0.25], [13.0, 3.0]),
                    maxfev=2000,
                )
            v_max, tau = float(popt[0]), float(popt[1])
            pred = _fv_exp_model(t_fit, v_max, tau)
            ss_res = float(np.sum((s_fit - pred) ** 2))
            ss_tot = float(np.sum((s_fit - np.mean(s_fit)) ** 2))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-8 else np.nan
            a0 = v_max / tau
            out["fv_vmax"] = v_max
            out["fv_tau"] = tau
            out["fv_a0"] = a0
            out["fv_pmax_rel"] = 0.25 * v_max * a0
            out["fv_r2"] = r2
        except Exception:
            pass

    # 4) 50점 리샘플링 속도 곡선 (추후 PCA/FPCA 비교용)
    curve = np.interp(curve_grid, t_rel, s_rel, left=s_rel[0], right=s_rel[-1])
    return out, curve


# =====================================================================
# [데이터: 컴바인 단독] [단계: 피처 추가] 트라이얼(6,310개) 단위 운동학 집계
# =====================================================================
def extract_trial_table(ply: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """프레임 데이터를 트라이얼 단위 생체역학 표(T1), 플래그 표, 40야드 곡선 표로 집계한다."""
    # 글리치 프레임 제외 후 집계
    clean = ply[~ply["flag_glitch"]].copy()

    # 0.3초(3프레임) 이동창 누적 방향전환각으로 실제 방향전환(COD >= 60도) 이벤트 탐지
    # 각 트라이얼 내 rolling sum
    roll3_turn = (
        clean.groupby(config.TRIAL_KEY, sort=False)["d_dir"]
        .transform(lambda s: s.rolling(3, min_periods=1).sum().abs())
    )
    clean["is_cod_frame"] = (roll3_turn >= config.COD_ANGLE_DEG).astype(np.int8)

    # 기본 트라이얼 집계
    grp = clean.groupby(config.TRIAL_KEY + ["draft_year", "drill_type"], sort=False)
    trials = grp.agg(
        n_frames=("s", "size"),
        t0=("time", "min"),
        t1=("time", "max"),
        max_gap=("dt", "max"),
        x_start=("x", "first"),
        y_start=("y", "first"),
        x_end=("x", "last"),
        y_end=("y", "last"),
        total_dist_xy=("step_dist", "sum"),
        s_max=("s", "max"),
        s_p95=("s", lambda x: float(np.nanpercentile(x, 95))),
        s_mean=("s", "mean"),
        a_max=("a", "max"),
        a_tan_pos_max=("a_tan_pos", "max"),
        a_tan_pos_p95=("a_tan_pos", lambda x: float(np.nanpercentile(x.dropna(), 95)) if x.notna().any() else np.nan),
        a_tan_neg_max=("a_tan_neg", "max"),
        a_tan_neg_p95=("a_tan_neg", lambda x: float(np.nanpercentile(x.dropna(), 95)) if x.notna().any() else np.nan),
        a_lat_max=("a_lat", "max"),
        a_lat_p95=("a_lat", lambda x: float(np.nanpercentile(x.dropna(), 95)) if x.notna().any() else np.nan),
        jerk_p95=("jerk", lambda x: float(np.nanpercentile(x.dropna(), 95)) if x.notna().any() else np.nan),
        total_turn_deg=("abs_d_dir", "sum"),
        cod_frames=("is_cod_frame", "sum"),
    ).reset_index()

    trials["duration_s"] = (trials["t1"] - trials["t0"]).dt.total_seconds()
    trials["straight_disp_xy"] = np.hypot(
        trials["x_end"] - trials["x_start"],
        trials["y_end"] - trials["y_start"],
    )
    # (x, y) 재계산 덕분에 path_efficiency 는 항상 [0, 1] 물리 범위 보장
    trials["path_efficiency"] = np.where(
        trials["total_dist_xy"] > 0.5,
        np.clip(trials["straight_disp_xy"] / trials["total_dist_xy"], 0.0, 1.0),
        np.nan,
    )

    # 트라이얼 품질 플래그 (D5, D7: drill_name 중앙값의 50% 미만 = 잘림)
    drill_median_frames = trials.groupby("drill_name")["n_frames"].transform("median")
    trials["frac_of_drill_median"] = trials["n_frames"] / drill_median_frames
    trials["flag_truncated"] = trials["frac_of_drill_median"] < config.TRUNCATED_MEDIAN_FRAC
    trials["flag_gap"] = trials["max_gap"].fillna(0.0) > config.MAX_FRAME_GAP_SEC
    glitch_by_trial = ply.groupby(config.TRIAL_KEY)["flag_glitch"].any().rename("flag_glitch")
    trials = trials.merge(glitch_by_trial, on=config.TRIAL_KEY, how="left")
    trials["flag_glitch"] = trials["flag_glitch"].fillna(False)

    # 드릴 패밀리 분류
    def _classify_family(dt_name: str) -> str:
        if dt_name == "FORTY_YARD_DASH":
            return "SPRINT"
        if dt_name in ("SHORT_SHUTTLE", "THREE_CONE_DRILL"):
            return "AGILITY_TIMED"
        return "SKILL_DRILL"

    trials["drill_family"] = trials["drill_type"].map(_classify_family)

    # 40야드 트라이얼 전용 F-V 프로파일 & 50점 곡선 추출
    forty_frames = clean[clean["drill_type"] == "FORTY_YARD_DASH"].copy()
    forty_frames["t_sec"] = (forty_frames["time"] - forty_frames["time"].min()).dt.total_seconds()

    fv_rows = []
    curve_rows = []
    curve_cols = [f"s_t{i:02d}" for i in range(config.FORTY_CURVE_POINTS)]

    for key_vals, sub in forty_frames.groupby(config.TRIAL_KEY, sort=False):
        key_dict = dict(zip(config.TRIAL_KEY, key_vals))
        fv_dict, curve_arr = _fit_single_forty_trial(
            sub["t_sec"].to_numpy(dtype=np.float64),
            sub["s"].to_numpy(dtype=np.float64),
            sub["step_dist"].to_numpy(dtype=np.float64),
        )
        fv_rows.append({**key_dict, **fv_dict})
        curve_rows.append({**key_dict, **dict(zip(curve_cols, curve_arr))})

    fv_df = pd.DataFrame(fv_rows)
    curves_df = pd.DataFrame(curve_rows)
    trials = trials.merge(fv_df, on=config.TRIAL_KEY, how="left")

    flag_cols = config.TRIAL_KEY + [
        "drill_type", "n_frames", "frac_of_drill_median",
        "flag_glitch", "flag_truncated", "flag_gap",
    ]
    flags_df = trials[flag_cols].copy()

    # 불필요한 좌표/타임스탬프 정리
    trials = trials.drop(columns=["t0", "t1", "x_start", "y_start", "x_end", "y_end"])

    print(
        f"[확인된 결과] 트라이얼 집계 완료: 총 {len(trials):,}개 | "
        f"잘림(truncated)={int(flags_df['flag_truncated'].sum())}개 | "
        f"40야드 F-V 피팅 중앙 R²={trials['fv_r2'].median():.4f} | "
        f"path_efficiency 최대값={trials['path_efficiency'].max():.4f}"
    )
    return trials, flags_df, curves_df


# =====================================================================
# [데이터: 컴바인 단독] [단계: 결측 구제 피처 & 도메인 물리 파생 통합]
#   - 전통 측정치 + 체격/물리 파생 + 드래프트 전 인적 메타(나이, 컨퍼런스, 서브포지션)
#   - 포기(결측) 선수 구제용 결측 지시자 + 전원(510/510) 스킬 드릴 운동학 집계
# =====================================================================
def build_combine_player_features(
    cr: pd.DataFrame,
    pl: pd.DataFrame,
    trials: pd.DataFrame,
) -> pd.DataFrame:
    """510명 선수 단위의 컴바인 마스터 피처 테이블(T2)을 생성한다."""
    df = pl.merge(cr, on=["nfl_id", "draft_year"], how="left")
    assert len(df) == 510 and df["nfl_id"].nunique() == 510, "선수 수(510명) 정합성 오류!"

    # 1) 드래프트 전 인적·대학 메타데이터
    birth = pd.to_datetime(df["birth_date"], errors="coerce")
    draft_ref = pd.to_datetime(df["draft_year"].astype(str) + "-04-28")
    df["draft_age"] = (draft_ref - birth).dt.days / 365.25
    df["sub_position"] = df["nfl_position"].map(config.SUB_POSITION_MAP).fillna(df["combine_position"])
    df["is_power5"] = df["college_conference"].isin(config.POWER5_CONFERENCES).astype(np.int8)

    # 2) 도메인 물리 파생 피처 (체격 비율 · 스피드스코어 · 폭발력 · 파워)
    df["bmi"] = (df["combine_weight"] / (df["combine_height"] ** 2)) * 703.0
    df["wing_to_height"] = df["wing_span"] / df["combine_height"]
    df["arm_to_height"] = df["arm_length"] / df["combine_height"]
    df["hand_to_height"] = df["hand_size"] / df["combine_height"]
    df["speed_score"] = (df["combine_weight"] * 200.0) / (df["forty"] ** 4)
    df["hass_score"] = df["speed_score"] * (df["combine_height"] / 73.0)
    df["burst_score"] = df["vertical"] + df["broad_jump"]
    # Sayers Peak Anaerobic Power (Watts): 60.7*vertical(cm) + 45.3*mass(kg) - 2055
    df["sayers_power_w"] = (
        60.7 * (df["vertical"] * 2.54)
        + 45.3 * (df["combine_weight"] * 0.45359237)
        - 2055.0
    )
    df["split10_to_forty"] = df["ten_yd_split"] / df["forty"]

    # 3) 결측 지시자 플래그 (포기/전략적 불참 신호 포착 — MNAR/MAR)
    df["n_missing_drills"] = df[config.DRILL_COLS].isna().sum(axis=1).astype(np.int8)
    df["did_forty"] = df["forty"].notna().astype(np.int8)
    df["did_agility_any"] = (df["three_cone"].notna() | df["short_shuttle"].notna()).astype(np.int8)
    df["did_bench"] = df["bench_reps"].notna().astype(np.int8)
    for col in config.DRILL_COLS:
        df[f"miss_{col}"] = df[col].isna().astype(np.int8)

    # 유효 트라이얼(잘림·글리치 제외)
    valid_tr = trials[(~trials["flag_truncated"]) & (~trials["flag_glitch"])].copy()

    # 4) 40야드 트래킹 집계 (최고 시도 + 평균 + F-V 파라미터 + 시도 일관성)
    f40 = valid_tr[valid_tr["drill_family"] == "SPRINT"]
    f40_agg = f40.groupby("nfl_id").agg(
        tr40_n_attempts=("s_max", "size"),
        tr40_smax_best=("s_max", "max"),
        tr40_smax_mean=("s_max", "mean"),
        tr40_smax_spread=("s_max", lambda s: float(s.max() - s.min()) if len(s) >= 2 else np.nan),
        tr40_a_pos_best=("a_tan_pos_max", "max"),
        tr40_t5yd_best=("t_5yd", "min"),
        tr40_t10yd_best=("t_10yd", "min"),
        tr40_t20yd_best=("t_20yd", "min"),
        tr40_t40yd_best=("t_40yd", "min"),
        tr40_fly10_30_best=("fly_10_30_speed", "max"),
        tr40_fv_vmax=("fv_vmax", "max"),
        tr40_fv_tau=("fv_tau", "min"),
        tr40_fv_a0=("fv_a0", "max"),
        tr40_fv_pmax_rel=("fv_pmax_rel", "max"),
        tr40_late_retention=("late_speed_retention", "max"),
    )
    df = df.merge(f40_agg, on="nfl_id", how="left")
    # SI 단위(Watts) 이론 최대 스프린트 파워: 0.25 * m(kg) * v_max(m/s) * a0(m/s^2)
    df["tr40_fv_pmax_abs_w"] = (
        df["tr40_fv_pmax_rel"] * (df["combine_weight"] * 0.45359237) * (0.9144 ** 2)
    )

    # 5) 포지션 스킬 드릴 집계 (포기/불참 선수 구제의 핵심 무기! 510명 중 거의 전원 보유)
    skill = valid_tr[valid_tr["drill_family"] == "SKILL_DRILL"]
    skill_agg = skill.groupby("nfl_id").agg(
        skill_n_trials=("s_max", "size"),
        skill_smax_max=("s_max", "max"),
        skill_smax_p90=("s_max", lambda x: float(np.nanpercentile(x, 90))),
        skill_smax_mean=("s_max", "mean"),
        skill_a_pos_max=("a_tan_pos_max", "max"),
        skill_a_pos_p90=("a_tan_pos_p95", lambda x: float(np.nanpercentile(x.dropna(), 90)) if x.notna().any() else np.nan),
        skill_decel_max=("a_tan_neg_max", "max"),
        skill_decel_p90=("a_tan_neg_p95", lambda x: float(np.nanpercentile(x.dropna(), 90)) if x.notna().any() else np.nan),
        skill_a_lat_max=("a_lat_max", "max"),
        skill_a_lat_p90=("a_lat_p95", lambda x: float(np.nanpercentile(x.dropna(), 90)) if x.notna().any() else np.nan),
        skill_jerk_p95_max=("jerk_p95", "max"),
        skill_path_eff_mean=("path_efficiency", "mean"),
        skill_cod_frames_mean=("cod_frames", "mean"),
        skill_turn_deg_mean=("total_turn_deg", "mean"),
    )
    df = df.merge(skill_agg, on="nfl_id", how="left")

    # 6) 전체 컴바인 트래킹 통합 피크 (510/510 전원 100% 결측 0% 보장 안전망)
    any_agg = valid_tr.groupby("nfl_id").agg(
        comb_any_n_trials=("s_max", "size"),
        comb_any_smax_max=("s_max", "max"),
        comb_any_a_pos_max=("a_tan_pos_max", "max"),
        comb_any_decel_max=("a_tan_neg_max", "max"),
        comb_any_a_lat_max=("a_lat_max", "max"),
    )
    df = df.merge(any_agg, on="nfl_id", how="left")

    # 검증: 10야드 스플릿 상관 및 포기 선수 구제 커버리지
    corr_10yd = df["ten_yd_split"].corr(df["tr40_t10yd_best"])
    corr_40yd = df["forty"].corr(df["tr40_t40yd_best"])
    skip_all7 = df[df["n_missing_drills"] == 7]
    rescued_pct = skip_all7["comb_any_smax_max"].notna().mean() * 100.0

    print(
        f"[확인된 결과] 선수 피처 생성 완료: shape={df.shape} | "
        f"(x,y) 재계산 t_10yd vs 공식 ten_yd_split r={corr_10yd:.3f}, "
        f"t_40yd vs 공식 forty r={corr_40yd:.3f} | "
        f"7개 드릴 전부 포기({len(skip_all7)}명) 트래킹 구제율={rescued_pct:.1f}%"
    )
    return df


def run_combine_fe(save: bool = True) -> dict[str, pd.DataFrame]:
    """Stage 2 전체 파이프라인 실행 및 Parquet 저장."""
    config.ensure_dirs()
    cr, ct, pl = load_combine_raw()
    ply, _ = clean_and_compute_frame_kinematics(ct)
    trials, flags, curves = extract_trial_table(ply)
    player_fe = build_combine_player_features(cr, pl, trials)

    if save:
        trials.to_parquet(config.PROCESSED_FILES["combine_trials"], index=False)
        flags.to_parquet(config.PROCESSED_FILES["combine_trial_flags"], index=False)
        curves.to_parquet(config.PROCESSED_FILES["combine_forty_curves"], index=False)
        player_fe.to_parquet(config.PROCESSED_FILES["combine_player_features"], index=False)
        print("[확인된 결과] Stage 2 Parquet 4종 저장 완료 -> data/processed/")

    return {
        "trials": trials,
        "flags": flags,
        "curves": curves,
        "player_features": player_fe,
    }


if __name__ == "__main__":
    run_combine_fe(save=True)
