"""[경기 단독] 루키 시즌 경기 운동능력 · 역할 · 2중 신입 기준(절대 + 시즌 코호트) 타깃 생성.

이 모듈은 `game_tracking_2023~2025.csv`(~2GB, 2,265만 행), `games.csv`, `players.csv`,
`player_play.csv`를 읽어 아래 3개의 Parquet 산출물을 생성한다:
1. game_play_peaks.parquet      : 루키 시즌 (경기, 플레이, 선수) 세그먼트별 3축 운동학 피크
2. game_rookie_athletic.parquet : 선수별 루키 경기 운동능력 타깃 (REG+POST 주버전 + PRE 포함 민감도)
                                  + 절대 폭발력 백분위 + 시즌 코호트(draft_year × position) 상대 지표
3. game_rookie_role.parquet     : 선수별 루키 정규시즌 스냅·주전(starter_flag)·포지션별 NGS 생산성
                                  + 사후 진단용 드래프트 버스트/스틸 메타 컬럼

절대 준수 규칙:
- `season`은 절대로 `game_id[:4]`에서 자르지 않고 항상 `games.csv`를 조인해 부여한다 (D3).
- 루키 시즌은 `season == draft_year`로 정의한다 (D4).
- 경기 트래킹 3개 파일은 `usecols`, `int32`/`float32`, `chunksize=1_000_000` 1회 스트리밍 패스로 처리한다.
"""
from __future__ import annotations

import time
import numpy as np
import pandas as pd

from src import config


# =====================================================================
# [데이터: 경기 단독] [단계: 데이터 로딩] 메타데이터 로드 (games, players)
# =====================================================================
def load_game_meta() -> tuple[pd.DataFrame, pd.DataFrame]:
    """games.csv와 players.csv(포지션·드래프트 연도/티어 포함)를 로딩한다."""
    games = pd.read_csv(
        config.RAW_FILES["games"],
        usecols=["game_id", "season", "season_type", "week"],
    )
    players = pd.read_csv(
        config.RAW_FILES["players"],
        usecols=["nfl_id", "draft_year", "nfl_position", "draft_round", "draft_overall_pick"],
    )
    cr_pos = pd.read_csv(
        config.RAW_FILES["combine_results"],
        usecols=["nfl_id", "combine_position"],
    )
    players = players.merge(cr_pos, on="nfl_id", how="left")
    players["sub_position"] = (
        players["nfl_position"].map(config.SUB_POSITION_MAP).fillna(players["combine_position"])
    )
    tier = pd.cut(players["draft_round"], bins=[0, 3, 7], labels=["1-3R", "4-7R"])
    players["draft_tier"] = tier.cat.add_categories("UDFA").fillna("UDFA").astype(str)

    print(
        f"[확인된 결과] 경기 메타 로딩 완료: games={len(games):,}경기 | "
        f"players={len(players)}명"
    )
    return games, players


# =====================================================================
# [데이터: 경기 단독] [단계: 데이터 정제 & 이상치 제거]
#   - 2GB 트래킹 1회 청크 스트리밍 패스
#   - 루키 시즌(season == draft_year) 조기 필터링으로 메모리/속도 최적화
#   - 글리치 프레임(s > 12.5, a > 12.0, a < 0, |ds/dt| > 12.0) 제거
#   - 3축 실전 운동학(속도, 추진가속, 제동감속, 구심가속 s|w|, 자세-이동각 전환) 추출
# =====================================================================
def stream_rookie_play_peaks(
    games: pd.DataFrame,
    players: pd.DataFrame,
    chunksize: int = config.TRACKING_CHUNKSIZE,
) -> pd.DataFrame:
    """3개 연도 경기 트래킹 CSV를 1회 스트리밍해 루키 시즌 세그먼트별 운동학 피크를 집계한다."""
    game_season = dict(zip(games["game_id"], games["season"]))
    player_dy = dict(zip(players["nfl_id"], players["draft_year"]))

    trk_cols = ["game_id", "play_id", "nfl_id", "s", "a", "dir", "o"]
    trk_dtypes = {
        "game_id": "int64",
        "play_id": "int32",
        "nfl_id": "int32",
        "s": "float32",
        "a": "float32",
        "dir": "float32",
        "o": "float32",
    }
    seg_key = ["game_id", "play_id", "nfl_id"]

    parts: list[pd.DataFrame] = []
    total_frames = 0
    rookie_frames = 0
    glitch_frames = 0

    for year, path in config.GAME_TRACKING_FILES.items():
        t0 = time.time()
        year_frames = 0
        for ch in pd.read_csv(path, usecols=trk_cols, dtype=trk_dtypes, chunksize=chunksize):
            year_frames += len(ch)
            total_frames += len(ch)

            # [단계: 데이터 정제] games.csv 기준 시즌 매핑 후 루키 시즌만 조기 필터링
            seasons = ch["game_id"].map(game_season)
            dys = ch["nfl_id"].map(player_dy)
            rk_mask = (seasons == dys).to_numpy()
            if not rk_mask.any():
                continue

            sub = ch.loc[rk_mask].reset_index(drop=True)
            rookie_frames += len(sub)

            # [단계: 이상치 제거] 글리치 프레임 제외
            s = sub["s"].to_numpy(dtype=np.float32)
            a = sub["a"].to_numpy(dtype=np.float32)
            bad = (
                (~np.isfinite(s))
                | (~np.isfinite(a))
                | (s > config.SPEED_GLITCH_YDS)
                | (a > config.ACCEL_GLITCH_YDS2)
                | (a < 0.0)
            )
            glitch_frames += int(bad.sum())
            if bad.any():
                sub = sub.loc[~bad].reset_index(drop=True)
                if len(sub) == 0:
                    continue
                s = sub["s"].to_numpy(dtype=np.float32)
                a = sub["a"].to_numpy(dtype=np.float32)

            # 같은 세그먼트 연속 프레임 마스크 (10Hz: dt = 0.1s)
            g_arr = sub["game_id"].to_numpy()
            p_arr = sub["play_id"].to_numpy()
            n_arr = sub["nfl_id"].to_numpy()
            same_seg = np.zeros(len(sub), dtype=bool)
            if len(sub) > 1:
                same_seg[1:] = (
                    (g_arr[1:] == g_arr[:-1])
                    & (p_arr[1:] == p_arr[:-1])
                    & (n_arr[1:] == n_arr[:-1])
                )

            # 부호 있는 종방향 가속/감속 (dt = 0.1s)
            ds_dt = np.full(len(sub), np.nan, dtype=np.float32)
            if len(sub) > 1:
                ds_dt[1:] = np.where(same_seg[1:], (s[1:] - s[:-1]) * 10.0, np.nan)
            ds_dt[np.abs(ds_dt) > config.DECEL_GLITCH_YDS2] = np.nan

            a_tan_pos = np.where(np.isfinite(ds_dt), np.maximum(ds_dt, 0.0), np.nan).astype(np.float32)
            a_tan_neg = np.where(np.isfinite(ds_dt), np.maximum(-ds_dt, 0.0), np.nan).astype(np.float32)

            # 구심/횡가속 a_lat = s * |omega| (s >= 1.5 yd/s 구간)
            direction = sub["dir"].to_numpy(dtype=np.float32)
            a_lat = np.full(len(sub), np.nan, dtype=np.float32)
            if len(sub) > 1:
                d_dir = ((direction[1:] - direction[:-1] + 180.0) % 360.0) - 180.0
                omega = np.abs(np.deg2rad(d_dir)) * 10.0
                valid_turn = same_seg[1:] & (s[1:] >= 1.5) & np.isfinite(omega)
                a_lat[1:] = np.where(valid_turn, np.minimum(s[1:] * omega, config.ACCEL_GLITCH_YDS2), np.nan)

            # 자세-이동각 전환율 (|o - dir|의 변화 속도: DB 백페달→브레이크, OL 패스셋 민첩성)
            orient = sub["o"].to_numpy(dtype=np.float32)
            od_diff = np.abs(((orient - direction + 180.0) % 360.0) - 180.0)
            hip_flip = np.full(len(sub), np.nan, dtype=np.float32)
            if len(sub) > 1:
                d_od = np.abs(od_diff[1:] - od_diff[:-1]) * 10.0  # deg/s
                valid_hf = same_seg[1:] & (s[1:] >= 1.5) & np.isfinite(d_od) & (d_od <= 900.0)
                hip_flip[1:] = np.where(valid_hf, d_od, np.nan)

            moving = s >= config.MOVING_SPEED
            work = pd.DataFrame({
                "game_id": g_arr,
                "play_id": p_arr,
                "nfl_id": n_arr,
                "s": s,
                "a": a,
                "a_tan_pos": a_tan_pos,
                "a_tan_neg": a_tan_neg,
                "a_lat": a_lat,
                "hip_flip": hip_flip,
                "s_moving": np.where(moving, s, 0.0).astype(np.float32),
                "is_moving": moving.astype(np.int32),
            })

            part = work.groupby(seg_key, sort=False).agg(
                n_frames=("s", "size"),
                s_max=("s", "max"),
                a_max=("a", "max"),
                a_tan_pos_max=("a_tan_pos", "max"),
                a_tan_neg_max=("a_tan_neg", "max"),
                a_lat_max=("a_lat", "max"),
                hip_flip_max=("hip_flip", "max"),
                s_moving_sum=("s_moving", "sum"),
                n_moving=("is_moving", "sum"),
            )
            parts.append(part)

        print(f"  - {path.name}: {year_frames:,} frames scanned ({time.time() - t0:.1f}s)")

    # 2단계 집계: 청크 경계에 걸친 세그먼트를 정확히 병합
    peaks = (
        pd.concat(parts)
        .groupby(level=seg_key)
        .agg(
            n_frames=("n_frames", "sum"),
            s_max=("s_max", "max"),
            a_max=("a_max", "max"),
            a_tan_pos_max=("a_tan_pos_max", "max"),
            a_tan_neg_max=("a_tan_neg_max", "max"),
            a_lat_max=("a_lat_max", "max"),
            hip_flip_max=("hip_flip_max", "max"),
            s_moving_sum=("s_moving_sum", "sum"),
            n_moving=("n_moving", "sum"),
        )
        .reset_index()
    )
    peaks = peaks.merge(games[["game_id", "season", "season_type"]], on="game_id", how="left")

    print(
        f"[확인된 결과] 1회 스트리밍 완료: 전체 {total_frames:,}프레임 -> "
        f"루키 {rookie_frames:,}프레임 (글리치 {glitch_frames}건 제거) -> "
        f"세그먼트 {len(peaks):,}개 / 루키 선수 {peaks['nfl_id'].nunique()}명"
    )
    return peaks


# =====================================================================
# [데이터: 경기 단독] [단계: 타깃 생성 — ① 절대 폭발력 + ② 시즌 코호트 상대 우위]
# =====================================================================
def _agg_athletic_slice(peaks_sub: pd.DataFrame, suffix: str = "") -> pd.DataFrame:
    """주어진 경기 유형 슬라이스(REG+POST 또는 ALL)에서 선수별 경기 운동학 타깃을 집계한다."""
    if len(peaks_sub) == 0:
        return pd.DataFrame(columns=["nfl_id"])

    # 플레이별 최고속도 상위 K개 평균 (단일 최댓값 노이즈 방어)
    top_k = (
        peaks_sub.sort_values("s_max", ascending=False)
        .groupby("nfl_id", sort=False)
        .head(config.TOP_K_PLAYS)
        .groupby("nfl_id")["s_max"]
        .mean()
        .rename(f"g_smax_top5mean{suffix}")
    )

    grp = peaks_sub.groupby("nfl_id")
    agg = grp.agg(
        g_n_plays=("s_max", "size"),
        g_smax_p95=("s_max", lambda x: float(np.nanpercentile(x, 95))),
        g_amax_p99=("a_max", lambda x: float(np.nanpercentile(x, 99))),
        g_accel_tan_p99=("a_tan_pos_max", lambda x: float(np.nanpercentile(x.dropna(), 99)) if x.notna().any() else np.nan),
        g_decel_tan_p99=("a_tan_neg_max", lambda x: float(np.nanpercentile(x.dropna(), 99)) if x.notna().any() else np.nan),
        g_centripetal_p99=("a_lat_max", lambda x: float(np.nanpercentile(x.dropna(), 99)) if x.notna().any() else np.nan),
        g_hip_flip_p95=("hip_flip_max", lambda x: float(np.nanpercentile(x.dropna(), 95)) if x.notna().any() else np.nan),
        _s_mov_sum=("s_moving_sum", "sum"),
        _n_mov=("n_moving", "sum"),
    )
    agg[f"g_s_mean_moving{suffix}"] = np.where(agg["_n_mov"] > 0, agg["_s_mov_sum"] / agg["_n_mov"], np.nan)
    agg = agg.drop(columns=["_s_mov_sum", "_n_mov"])
    if suffix:
        agg = agg.rename(columns={c: f"{c}{suffix}" for c in agg.columns if not c.endswith(suffix)})
    out = agg.join(top_k, how="left").reset_index()
    return out


def _add_abs_and_cohort_ranks(
    df: pd.DataFrame,
    target_cols: list[str],
    valid_mask: pd.Series | None = None,
) -> pd.DataFrame:
    """절대 폭발력(포지션 전체 내 백분위) + 시즌 코호트 상대 우위(draft_year × position)를 추가한다."""
    df = df.copy()
    mask = pd.Series(True, index=df.index) if valid_mask is None else valid_mask.fillna(False)

    for col in target_cols:
        if col not in df.columns:
            continue
        # 유효 표본에서만 순위·Z-score 계산 (저노출 노이즈가 순위 오염 방지)
        eff = df[col].where(mask)
        # ① 절대 기준: 전체 풀의 같은 포지션 내 백분위 [0, 1]
        df[f"{col}_pos_pct"] = eff.groupby(df["combine_position"]).rank(pct=True)
        # ② 시즌 코호트 기준: 같은 draft_year × combine_position 내 백분위 및 Z-score
        cohort_key = [df["draft_year"], df["combine_position"]]
        df[f"{col}_cohort_pct"] = eff.groupby(cohort_key).rank(pct=True)
        c_mean = eff.groupby(cohort_key).transform("mean")
        c_std = eff.groupby(cohort_key).transform("std")
        df[f"{col}_cohort_z"] = (eff - c_mean) / c_std.replace(0, np.nan)

    return df


def build_rookie_athletic_targets(
    peaks: pd.DataFrame,
    players: pd.DataFrame,
) -> pd.DataFrame:
    """선수(510명) 기준 루키 경기 운동능력 타깃 테이블(T3)을 생성한다."""
    main_peaks = peaks[peaks["season_type"].isin(config.TARGET_SEASON_TYPES)]
    ath_main = _agg_athletic_slice(main_peaks, suffix="")
    ath_pre = _agg_athletic_slice(peaks, suffix="_with_pre")

    base = players[["nfl_id", "draft_year", "combine_position", "sub_position"]].copy()
    ath = base.merge(ath_main, on="nfl_id", how="left").merge(ath_pre, on="nfl_id", how="left")

    ath["g_n_plays"] = ath["g_n_plays"].fillna(0).astype(int)
    ath["g_n_plays_with_pre"] = ath["g_n_plays_with_pre"].fillna(0).astype(int)
    ath["low_exposure"] = ath["g_n_plays"] < config.MIN_PLAYS_EXPOSURE
    ath["exposure_weight"] = ath["g_n_plays"] / (ath["g_n_plays"] + config.EXPOSURE_WEIGHT_K)

    # 충분한 노출(>= MIN_PLAYS_EXPOSURE) 선수 기준으로 절대(포지션 전체) & 시즌 코호트 상대 지표 산출
    valid_expo = ~ath["low_exposure"]
    ath = _add_abs_and_cohort_ranks(
        ath,
        target_cols=["g_smax_top5mean", "g_accel_tan_p99", "g_decel_tan_p99", "g_centripetal_p99"],
        valid_mask=valid_expo,
    )

    # 폭발적 신입 이진 플래그 (상위 20%: 절대 기준 vs 해당 시즌 코호트 기준)
    cut_pct = 1.0 - config.EXPLOSIVE_TOP_FRACTION
    ath["is_explosive_speed_abs"] = np.where(
        ath["g_smax_top5mean_pos_pct"].notna(),
        (ath["g_smax_top5mean_pos_pct"] >= cut_pct).astype(float),
        np.nan,
    )
    ath["is_explosive_speed_cohort"] = np.where(
        ath["g_smax_top5mean_cohort_pct"].notna(),
        (ath["g_smax_top5mean_cohort_pct"] >= cut_pct).astype(float),
        np.nan,
    )

    print(
        f"[확인된 결과] T3 루키 경기 운동능력 타깃 생성: {len(ath)}명 | "
        f"노출>={config.MIN_PLAYS_EXPOSURE} 충족={(~ath['low_exposure']).sum()}명 | "
        f"REG+POST 최고속도 중앙={ath.loc[valid_expo, 'g_smax_top5mean'].median():.2f} yd/s"
    )
    return ath


# =====================================================================
# [데이터: 경기 단독] [단계: 타깃 생성 — 역할 · 주전 · 포지션별 NGS 생산성]
# =====================================================================
def build_rookie_role_targets(
    games: pd.DataFrame,
    players: pd.DataFrame,
) -> pd.DataFrame:
    """player_play.csv에서 루키 스냅·주전(절대/코호트)·포지션별 NGS 생산성(T4)을 집계한다."""
    pp_cols = [
        "game_id", "play_id", "nfl_id", "expected_points_added",
        "route_ran", "target", "rec_yards", "yards_after_catch", "expected_yards_after_catch",
        "separation_at_pass_forward", "cushion",
        "pressure_allowed", "peak_pressure_probability_allowed", "pass_rushers_encountered",
        "blitzing", "player_get_off", "time_to_pressure", "quick_pressure", "sack", "tackle",
        "coverage_assignment",
    ]
    pp = pd.read_csv(config.RAW_FILES["player_play"], usecols=pp_cols)
    pp = pp.merge(games[["game_id", "season", "season_type"]], on="game_id", how="left")
    pp = pp.merge(players[["nfl_id", "draft_year", "combine_position"]], on="nfl_id", how="left")
    rk = pp[pp["season"] == pp["draft_year"]].copy()

    # 정규시즌 스냅 & 경기 수
    rk_reg = rk[rk["season_type"] == "REG"]
    rk_rp = rk[rk["season_type"].isin(config.TARGET_SEASON_TYPES)]

    role = players[[
        "nfl_id", "draft_year", "combine_position", "sub_position",
        "draft_round", "draft_overall_pick", "draft_tier",
    ]].copy()
    role["rookie_snaps_reg"] = role["nfl_id"].map(rk_reg.groupby("nfl_id").size()).fillna(0).astype(int)
    role["rookie_games_reg"] = role["nfl_id"].map(rk_reg.groupby("nfl_id")["game_id"].nunique()).fillna(0).astype(int)
    role["rookie_snaps_reg_post"] = role["nfl_id"].map(rk_rp.groupby("nfl_id").size()).fillna(0).astype(int)
    role["rookie_snaps_all"] = role["nfl_id"].map(rk.groupby("nfl_id").size()).fillna(0).astype(int)

    # ① 절대 기준 주전(starter_flag): 전체 포지션 그룹 내 정규시즌 스냅 상위 1/3 (D9)
    q_cut = 1.0 - config.STARTER_TOP_FRACTION
    pos_cut = role.groupby("combine_position")["rookie_snaps_reg"].transform(lambda s: s.quantile(q_cut))
    role["starter_flag"] = (role["rookie_snaps_reg"] >= pos_cut).astype(np.int8)

    # ② 시즌 코호트 기준 주전(starter_flag_cohort): 같은 draft_year × position 내 상위 1/3
    cohort_cut = role.groupby(["draft_year", "combine_position"])["rookie_snaps_reg"].transform(
        lambda s: s.quantile(q_cut)
    )
    role["starter_flag_cohort"] = (role["rookie_snaps_reg"] >= cohort_cut).astype(np.int8)

    # 스냅의 절대 백분위 & 시즌 코호트 백분위/Z-score
    role = _add_abs_and_cohort_ranks(role, target_cols=["rookie_snaps_reg"])

    # 사후 오류 분석 전용 플래그 (드래프트 실패/버스트 vs 스틸 진단 — 입력 X로 절대 사용 금지!)
    pos_med_snaps = role.groupby("combine_position")["rookie_snaps_reg"].transform("median")
    role["diag_draft_bust"] = ((role["draft_tier"] == "1-3R") & (role["rookie_snaps_reg"] < pos_med_snaps)).astype(np.int8)
    role["diag_draft_steal"] = ((role["draft_tier"] != "1-3R") & (role["starter_flag"] == 1)).astype(np.int8)

    # 포지션별 유효 NGS 생산성 지표 (REG+POST 기준, eda.md 3.5 TRUE 비율 근거)
    epa_avg = rk_rp.groupby("nfl_id")["expected_points_added"].mean().rename("avg_epa_on_field")
    role = role.merge(epa_avg, on="nfl_id", how="left")

    # WR/TE 리시빙 생산성
    wrte = rk_rp[rk_rp["route_ran"].notna()].copy()
    wrte["yac_oe"] = wrte["yards_after_catch"] - wrte["expected_yards_after_catch"]
    wrte_agg = wrte.groupby("nfl_id").agg(
        wrte_routes=("route_ran", "size"),
        wrte_target_rate=("target", lambda s: float(s.astype(float).mean())),
        wrte_avg_separation=("separation_at_pass_forward", "mean"),
        wrte_avg_cushion=("cushion", "mean"),
        wrte_avg_yac_oe=("yac_oe", "mean"),
    )
    role = role.merge(wrte_agg, on="nfl_id", how="left")

    # OL 패스블로킹 생산성
    ol_pb = rk_rp[rk_rp["pass_rushers_encountered"].notna()]
    ol_agg = ol_pb.groupby("nfl_id").agg(
        ol_pass_block_snaps=("pressure_allowed", "size"),
        ol_pressure_allowed_rate=("pressure_allowed", lambda s: float(s.astype(float).mean())),
        ol_avg_peak_pressure_prob=("peak_pressure_probability_allowed", "mean"),
    )
    role = role.merge(ol_agg, on="nfl_id", how="left")

    # DL 패스러시·폭발력(player_get_off) 생산성
    dl_rp = rk_rp[rk_rp["combine_position"] == "DL"]
    dl_agg = dl_rp.groupby("nfl_id").agg(
        dl_blitz_rate=("blitzing", lambda s: float(s.astype(float).mean())),
        dl_avg_get_off=("player_get_off", "mean"),
        dl_avg_time_to_pressure=("time_to_pressure", "mean"),
        dl_tackle_rate=("tackle", lambda s: float(s.astype(float).mean())),
    )
    role = role.merge(dl_agg, on="nfl_id", how="left")

    # DB 커버리지·태클 생산성
    db_rp = rk_rp[rk_rp["combine_position"] == "DB"]
    db_agg = db_rp.groupby("nfl_id").agg(
        db_coverage_snaps=("coverage_assignment", lambda s: int(s.notna().sum())),
        db_tackle_rate=("tackle", lambda s: float(s.astype(float).mean())),
    )
    role = role.merge(db_agg, on="nfl_id", how="left")

    print(
        f"[확인된 결과] T4 루키 역할·생산성 타깃 생성: {len(role)}명 | "
        f"주전(전체상위1/3)={int(role['starter_flag'].sum())}명 | "
        f"주전(코호트상위1/3)={int(role['starter_flag_cohort'].sum())}명 | "
        f"사후진단 버스트={int(role['diag_draft_bust'].sum())}명 / 스틸={int(role['diag_draft_steal'].sum())}명"
    )
    return role


def run_game_targets(save: bool = True) -> dict[str, pd.DataFrame]:
    """Stage 3 전체 파이프라인 실행 및 Parquet 저장."""
    config.ensure_dirs()
    games, players = load_game_meta()
    peaks = stream_rookie_play_peaks(games, players)
    ath = build_rookie_athletic_targets(peaks, players)
    role = build_rookie_role_targets(games, players)

    if save:
        peaks.to_parquet(config.PROCESSED_FILES["game_play_peaks"], index=False)
        ath.to_parquet(config.PROCESSED_FILES["game_rookie_athletic"], index=False)
        role.to_parquet(config.PROCESSED_FILES["game_rookie_role"], index=False)
        print("[확인된 결과] Stage 3 Parquet 3종 저장 완료 -> data/processed/")

    return {"peaks": peaks, "athletic": ath, "role": role}


if __name__ == "__main__":
    run_game_targets(save=True)
