"""[경기 단독] 루키 시즌 경기 운동능력 · 역할 · 2중 신입 기준(절대 + 시즌 코호트) 타깃 생성.

이 모듈은 `game_tracking_2023~2025.csv`(~2GB, 2,265만 행), `games.csv`, `players.csv`,
`player_play.csv`를 읽어 아래 3개의 Parquet 산출물을 생성한다:
1. game_play_peaks.parquet      : 루키 시즌 (경기, 플레이, 선수) 세그먼트별 3축 운동학 피크
2. game_rookie_athletic.parquet : 선수별 루키 경기 운동능력 원본 타깃 (REG+POST 주버전 + PRE 포함 민감도)
3. game_rookie_role.parquet     : 선수별 루키 정규시즌 스냅·경기 수·포지션별 NGS 생산성 원본 값

상대 기준 라벨(포지션/시즌 코호트 백분위·Z-score, 폭발적 신입 플래그, starter_flag, 버스트/스틸 진단)은
"다른 선수들의 값"으로 기준을 정하므로 Parquet에 저장하지 않는다. 반드시 `RelativeTargetEncoder`를
Train 선수로만 `fit`한 뒤 Train/Valid/Test에 `transform`해서 만든다 (테스트셋 정답 누수 방지).

절대 준수 규칙:
- `season`은 절대로 `game_id[:4]`에서 자르지 않고 항상 `games.csv`를 조인해 부여한다 (D3).
- 루키 시즌은 `season == draft_year`로 정의한다 (D4).
- 경기 트래킹 3개 파일은 `usecols`, `int32`/`float32`, `chunksize=1_000_000` 1회 스트리밍 패스로 처리한다.
- 상대 기준 라벨의 기준값(평균·표준편차·분위수 컷)은 Train 선수만으로 계산한다.
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


# =====================================================================
# [데이터: 경기 단독] [단계: 상대 기준 라벨 — Train 전용 fit / transform]
#   포지션·시즌 코호트 대비 상대 라벨은 "다른 선수들의 값"으로 기준(평균·표준편차·분위수 컷)을 정한다.
#   510명 전체로 기준을 정하면 테스트 선수의 정답이 학습 라벨에 섞이므로(누수),
#   기준값은 Train 선수만으로 fit 하고 Train/Valid/Test 에는 transform 만 적용한다.
# =====================================================================
RELATIVE_SPEED_COLS = ["g_smax_top5mean", "g_accel_tan_p99", "g_decel_tan_p99", "g_centripetal_p99"]
RELATIVE_SNAP_COL = "rookie_snaps_reg"
_REL_SUFFIXES = ("_pos_pct", "_cohort_pct", "_cohort_z")
_REL_SPECIAL = {
    "is_explosive_speed_abs": "g_smax_top5mean",
    "is_explosive_speed_cohort": "g_smax_top5mean",
    "starter_flag": RELATIVE_SNAP_COL,
    "starter_flag_cohort": RELATIVE_SNAP_COL,
    "diag_draft_bust": RELATIVE_SNAP_COL,
    "diag_draft_steal": RELATIVE_SNAP_COL,
}


def relative_label_base(label: str) -> str | None:
    """상대 라벨 이름이면 그 라벨을 만드는 원본 컬럼명을, 아니면 None을 반환한다."""
    if label in _REL_SPECIAL:
        return _REL_SPECIAL[label]
    for base in RELATIVE_SPEED_COLS + [RELATIVE_SNAP_COL]:
        for suf in _REL_SUFFIXES:
            if label == f"{base}{suf}":
                return base
    return None


def _avg_rank_pct(ref_sorted: np.ndarray, x: np.ndarray) -> np.ndarray:
    """학습 분포(ref) 대비 x의 백분위. ref에 포함된 값은 pandas rank(pct=True, average)와 같은 값을 준다."""
    out = np.full(len(x), np.nan, dtype=np.float64)
    n = len(ref_sorted)
    ok = np.isfinite(x)
    if n == 0 or not ok.any():
        return out
    lo = np.searchsorted(ref_sorted, x[ok], side="left")
    hi = np.searchsorted(ref_sorted, x[ok], side="right")
    out[ok] = ((lo + hi + 1) / 2.0) / n
    return out


class RelativeTargetEncoder:
    """상대 기준 라벨을 Train 통계로만 만드는 fit/transform 인코더.

    fit(train_df)   : Train 선수만으로 포지션별·코호트(draft_year × position)별 기준값을 저장한다.
                      - 속도·가속 라벨: 노출 충족(~low_exposure) 선수만 기준에 사용
                      - 스냅 라벨    : 스냅 0인 선수 포함 전원 사용 (D9)
                      - 코호트 표본이 min_group_size 미만이면 포지션 기준으로 대체
    transform(df)   : 저장한 기준값으로 아래 컬럼을 추가한 DataFrame 사본을 반환한다.
                      {col}_pos_pct, {col}_cohort_pct, {col}_cohort_z  (col = 속도 4종 + rookie_snaps_reg)
                      is_explosive_speed_abs / _cohort, starter_flag / _cohort, diag_draft_bust / _steal
    """

    def __init__(
        self,
        speed_cols: list[str] | None = None,
        snap_col: str = RELATIVE_SNAP_COL,
        pos_col: str = "combine_position",
        cohort_col: str = "draft_year",
        min_group_size: int = 5,
    ):
        self.speed_cols = list(RELATIVE_SPEED_COLS if speed_cols is None else speed_cols)
        self.snap_col = snap_col
        self.pos_col = pos_col
        self.cohort_col = cohort_col
        self.min_group_size = min_group_size

    # ---------------- fit ----------------
    def _fit_col(self, df: pd.DataFrame, col: str, valid: pd.Series) -> dict:
        vals = pd.to_numeric(df[col], errors="coerce").where(valid)
        stats = {"pos": {}, "cohort": {}}
        for pos, s in vals.groupby(df[self.pos_col]):
            s = s.dropna().to_numpy(dtype=np.float64)
            stats["pos"][pos] = {
                "ref": np.sort(s),
                "mean": float(s.mean()) if len(s) else np.nan,
                "std": float(s.std(ddof=1)) if len(s) > 1 else np.nan,
            }
        for (dy, pos), s in vals.groupby([df[self.cohort_col], df[self.pos_col]]):
            s = s.dropna().to_numpy(dtype=np.float64)
            if len(s) >= self.min_group_size:
                stats["cohort"][(dy, pos)] = {
                    "ref": np.sort(s),
                    "mean": float(s.mean()),
                    "std": float(s.std(ddof=1)),
                }
        return stats

    def fit(self, df: pd.DataFrame, y=None) -> "RelativeTargetEncoder":
        valid_speed = (
            ~df["low_exposure"].fillna(True).astype(bool)
            if "low_exposure" in df.columns
            else pd.Series(True, index=df.index)
        )
        all_rows = pd.Series(True, index=df.index)

        self.stats_ = {}
        for col in self.speed_cols:
            if col in df.columns:
                self.stats_[col] = self._fit_col(df, col, valid_speed)
        if self.snap_col in df.columns:
            self.stats_[self.snap_col] = self._fit_col(df, self.snap_col, all_rows)

            # 주전 컷(상위 1/3)과 버스트 진단용 중앙값 — Train 분포에서만 계산
            q = 1.0 - config.STARTER_TOP_FRACTION
            snaps = pd.to_numeric(df[self.snap_col], errors="coerce")
            self.starter_cut_pos_ = snaps.groupby(df[self.pos_col]).quantile(q).to_dict()
            self.snap_median_pos_ = snaps.groupby(df[self.pos_col]).median().to_dict()
            grp = snaps.groupby([df[self.cohort_col], df[self.pos_col]])
            sizes = grp.size()
            cut_c = grp.quantile(q)
            self.starter_cut_cohort_ = {
                k: float(v) for k, v in cut_c.items() if sizes.get(k, 0) >= self.min_group_size
            }
        self.n_fit_ = int(len(df))
        return self

    # ---------------- transform ----------------
    def _lookup(self, stats: dict, pos, dy, use_cohort: bool) -> dict | None:
        if use_cohort and (dy, pos) in stats["cohort"]:
            return stats["cohort"][(dy, pos)]
        return stats["pos"].get(pos)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "stats_"):
            raise RuntimeError("RelativeTargetEncoder: fit(train_df)를 먼저 호출하세요.")
        out = df.copy()
        pos_arr = df[self.pos_col].to_numpy()
        dy_arr = df[self.cohort_col].to_numpy()
        valid_speed = (
            ~df["low_exposure"].fillna(True).astype(bool).to_numpy()
            if "low_exposure" in df.columns
            else np.ones(len(df), dtype=bool)
        )
        new_cols: dict[str, np.ndarray] = {}

        for col, stats in self.stats_.items():
            x = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float64)
            if col in self.speed_cols:
                x = np.where(valid_speed, x, np.nan)
            pos_pct = np.full(len(df), np.nan)
            coh_pct = np.full(len(df), np.nan)
            coh_z = np.full(len(df), np.nan)
            for key in set(zip(dy_arr, pos_arr)):
                dy, pos = key
                idx = np.where((dy_arr == dy) & (pos_arr == pos))[0]
                p_st = self._lookup(stats, pos, dy, use_cohort=False)
                c_st = self._lookup(stats, pos, dy, use_cohort=True)
                if p_st is not None:
                    pos_pct[idx] = _avg_rank_pct(p_st["ref"], x[idx])
                if c_st is not None:
                    coh_pct[idx] = _avg_rank_pct(c_st["ref"], x[idx])
                    sd = c_st["std"]
                    if np.isfinite(sd) and sd > 1e-9:
                        coh_z[idx] = (x[idx] - c_st["mean"]) / sd
            new_cols[f"{col}_pos_pct"] = pos_pct
            new_cols[f"{col}_cohort_pct"] = coh_pct
            new_cols[f"{col}_cohort_z"] = coh_z

        # 폭발적 신입 플래그 (상위 EXPLOSIVE_TOP_FRACTION, 기준은 Train 백분위)
        cut_pct = 1.0 - config.EXPLOSIVE_TOP_FRACTION
        for name, src in (("is_explosive_speed_abs", "g_smax_top5mean_pos_pct"),
                          ("is_explosive_speed_cohort", "g_smax_top5mean_cohort_pct")):
            if src in new_cols:
                p = new_cols[src]
                new_cols[name] = np.where(np.isfinite(p), (p >= cut_pct).astype(float), np.nan)

        # 주전 플래그 & 사후 진단 (Train 컷 / Train 중앙값 적용)
        if self.snap_col in self.stats_:
            snaps = pd.to_numeric(df[self.snap_col], errors="coerce").to_numpy(dtype=np.float64)
            cut_p = np.array([self.starter_cut_pos_.get(p, np.nan) for p in pos_arr], dtype=np.float64)
            cut_c = np.array(
                [self.starter_cut_cohort_.get((d, p), self.starter_cut_pos_.get(p, np.nan))
                 for d, p in zip(dy_arr, pos_arr)],
                dtype=np.float64,
            )
            starter = (snaps >= cut_p).astype(np.int8)
            new_cols["starter_flag"] = starter
            new_cols["starter_flag_cohort"] = (snaps >= cut_c).astype(np.int8)
            if "draft_tier" in df.columns:
                tier = df["draft_tier"].astype(str).to_numpy()
                med = np.array([self.snap_median_pos_.get(p, np.nan) for p in pos_arr], dtype=np.float64)
                new_cols["diag_draft_bust"] = ((tier == "1-3R") & (snaps < med)).astype(np.int8)
                new_cols["diag_draft_steal"] = ((tier != "1-3R") & (starter == 1)).astype(np.int8)

        for name, arr in new_cols.items():
            out[name] = arr
        return out

    def fit_transform(self, df: pd.DataFrame, y=None) -> pd.DataFrame:
        return self.fit(df).transform(df)


def check_relative_labels_ignore_holdout(
    train_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    seed: int = config.SEED,
) -> bool:
    """교란 테스트: 홀드아웃(테스트) 선수의 원본 타깃을 무작위로 바꿔도 Train 라벨이 그대로인지 확인한다.

    Train에 fit 한 인코더 → Train 라벨 A.
    홀드아웃 원본 값을 섞은 뒤 같은 절차를 다시 실행 → Train 라벨 B.  A == B 이어야 한다.
    (예전 방식처럼 510명 전체로 기준을 잡으면 이 테스트에서 A != B 가 된다.)
    """
    rng = np.random.default_rng(seed)
    cols = [c for c in RELATIVE_SPEED_COLS + [RELATIVE_SNAP_COL] if c in holdout_df.columns]

    def labels_for_train(hold: pd.DataFrame) -> pd.DataFrame:
        enc = RelativeTargetEncoder().fit(train_df)
        both = pd.concat([train_df, hold], ignore_index=True)
        lab = enc.transform(both).iloc[: len(train_df)]
        return lab[[c for c in lab.columns if relative_label_base(c) is not None]]

    a = labels_for_train(holdout_df)
    hold_perturbed = holdout_df.copy()
    for c in cols:
        hold_perturbed[c] = rng.permutation(hold_perturbed[c].to_numpy()) * rng.uniform(0.5, 1.5)
    b = labels_for_train(hold_perturbed)
    pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True))
    return True


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

    # 상대 기준 라벨(백분위·코호트 Z·폭발적 신입 플래그)은 여기서 만들지 않는다.
    # -> Train 선수로만 기준을 잡는 RelativeTargetEncoder 로 분할 이후 생성 (테스트 누수 방지)
    valid_expo = ~ath["low_exposure"]

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

    # starter_flag(상위 1/3, D9) · 스냅 백분위/코호트 Z · 버스트/스틸 진단은 여기서 만들지 않는다.
    # -> 상위 1/3 컷과 중앙값을 Train 선수로만 정하는 RelativeTargetEncoder 로 분할 이후 생성

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
        f"[확인된 결과] T4 루키 역할·생산성 원본 값 생성: {len(role)}명 | "
        f"정규시즌 스냅 1회 이상={int((role['rookie_snaps_reg'] > 0).sum())}명 | "
        f"스냅 중앙값={role['rookie_snaps_reg'].median():.0f} "
        f"(starter_flag 등 상대 라벨은 분할 후 RelativeTargetEncoder로 생성)"
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
