"""프로젝트 공용 설정: 로컬/Colab 겸용 경로 · 시드 · 도메인 상수 · 환경 점검.

모든 노트북 첫 셀 (로컬 VS Code / Google Colab 공통, StopIteration 방지):
    import os, sys
    from pathlib import Path

    def find_project_root() -> Path:
        candidates = []
        if "NFL_PROJECT_ROOT" in os.environ:
            candidates.append(Path(os.environ["NFL_PROJECT_ROOT"]))
        candidates.extend([Path.cwd(), *Path.cwd().parents, Path("/content/nfl-bdb2027")])
        if "google.colab" in sys.modules or "COLAB_RELEASE_TAG" in os.environ:
            from google.colab import drive
            drive.mount("/content/drive", force_remount=False)
            candidates.extend([
                Path("/content/drive/MyDrive/nfl-bdb2027"),
                Path("/content/drive/MyDrive/Colab Notebooks/nfl-bdb2027"),
            ])
        for p in candidates:
            if (p / "src" / "config.py").exists():
                return p.resolve()
        raise FileNotFoundError(
            f"src/config.py를 찾지 못했습니다 (현재 cwd={Path.cwd()}). "
            "환경변수 NFL_PROJECT_ROOT를 지정하거나 프로젝트 폴더 안에서 실행하세요."
        )

    ROOT = find_project_root()
    os.chdir(ROOT)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from src import config
    config.setup()
"""
from __future__ import annotations

import os
import platform
import random
import sys
from importlib import metadata
from pathlib import Path

import numpy as np

# ============================================================
# 1. 실행 환경 및 경로 (로컬 + Colab 자동 호환)
# ============================================================
IN_COLAB: bool = "COLAB_RELEASE_TAG" in os.environ or "google.colab" in sys.modules
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _resolve_data_dir() -> Path:
    """데이터 폴더 탐색 우선순위:
    1) 환경변수 NFL_DATA_DIR
    2) 프로젝트 루트 아래 data/ (raw/combine_results.csv 존재 시 최우선)
    3) Colab 실행 시 Google Drive 공유 폴더 자동 탐색
    4) 기본값: PROJECT_ROOT / "data"
    """
    if "NFL_DATA_DIR" in os.environ:
        return Path(os.environ["NFL_DATA_DIR"]).expanduser()

    local_data = PROJECT_ROOT / "data"
    if (local_data / "raw" / "combine_results.csv").exists():
        return local_data

    if IN_COLAB:
        colab_candidates = [
            Path("/content/drive/MyDrive/nfl-bdb2027/data"),
            Path("/content/drive/MyDrive/nfl-bdb2027-data"),
            Path("/content/drive/MyDrive/Colab Notebooks/nfl-bdb2027/data"),
        ]
        for cand in colab_candidates:
            if (cand / "raw" / "combine_results.csv").exists():
                return cand
            if (cand / "combine_results.csv").exists():
                return cand.parent

    return local_data


DATA_DIR = _resolve_data_dir()
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
TEST_IDS_PATH = SPLITS_DIR / "test_ids.csv"

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
TABLES_DIR = REPORTS_DIR / "tables"

# --- 원본 CSV (data/raw/) ---
RAW_FILES = {
    "combine_results": RAW_DIR / "combine_results.csv",
    "combine_tracking": RAW_DIR / "combine_tracking.csv",
    "games": RAW_DIR / "games.csv",
    "players": RAW_DIR / "players.csv",
    "player_play": RAW_DIR / "player_play.csv",
    "player_career_successes": RAW_DIR / "player_career_successes.csv",
}
GAME_TRACKING_FILES = {
    2023: RAW_DIR / "game_tracking_2023.csv",
    2024: RAW_DIR / "game_tracking_2024.csv",
    2025: RAW_DIR / "game_tracking_2025.csv",
}

# --- 중간 산출물 (data/processed/, Parquet) ---
PROCESSED_FILES = {
    "combine_trials": PROCESSED_DIR / "combine_trials.parquet",                    # T1
    "combine_trial_flags": PROCESSED_DIR / "combine_trial_flags.parquet",          # T1 flags
    "combine_forty_curves": PROCESSED_DIR / "combine_forty_curves.parquet",        # 40yd curves
    "combine_player_features": PROCESSED_DIR / "combine_player_features.parquet",  # T2
    "game_play_peaks": PROCESSED_DIR / "game_play_peaks.parquet",                  # T3 intermediate
    "game_rookie_athletic": PROCESSED_DIR / "game_rookie_athletic.parquet",        # T3
    "game_rookie_role": PROCESSED_DIR / "game_rookie_role.parquet",                # T4
    "master_player_table": PROCESSED_DIR / "master_player_table.parquet",          # T5
}

# ============================================================
# 2. 재현성
# ============================================================
SEED = 42

# ============================================================
# 3. 도메인 상수 (Stage 1 EDA + 물리 검증 확정값)
# ============================================================
POSITION_GROUPS = ["DB", "OL", "DL", "WR", "TE"]

# 세부 NFL 포지션 -> 8개 생체역학 서브포지션 아키타입 (DL 내부 EDGE vs IDL 교란 분리 등)
SUB_POSITION_MAP = {
    "DE": "EDGE",
    "OLB": "EDGE",
    "DT": "IDL",
    "NT": "IDL",
    "CB": "CB",
    "DB": "CB",
    "SS": "S",
    "FS": "S",
    "ILB": "S",
    "MLB": "S",
    "RB": "CB",
    "T": "OT",
    "G": "IOL",
    "C": "IOL",
    "WR": "WR",
    "TE": "TE",
    "FB": "TE",
}

# Power-5 대학 컨퍼런스 (경쟁 수준 티어 피처용)
POWER5_CONFERENCES = {
    "Southeastern Conference",
    "Big Ten Conference",
    "Atlantic Coast Conference",
    "Big Twelve Conference",
    "Pacific Twelve Conference",
}

# 컴바인 컬럼 그룹 (D2: 실제 CSV 컬럼명 기준)
BODY_COLS = ["combine_height", "combine_weight", "hand_size", "arm_length", "wing_span"]
DRILL_COLS = [
    "ten_yd_split", "forty", "vertical", "broad_jump",
    "three_cone", "short_shuttle", "bench_reps",
]
NGS_COLS = ["ngs_athleticism_score", "ngs_college_production_score", "ngs_final_score"]

# 드래프트 결과 컬럼 (절대 입력 피처 X로 사용 금지 — 층화 분할 및 사후 버스트/스틸 진단 전용)
DRAFT_FORBIDDEN_FEATURE_COLS = [
    "draft_round", "draft_pick_within_round", "draft_overall_pick", "draft_tier",
]

# 컴바인 트라이얼 키 및 잘림 기준 (D1, D7)
TRIAL_KEY = ["nfl_id", "event_id", "drill_name", "attempt"]
TRUNCATED_MEDIAN_FRAC = 0.50   # drill_name 중앙값 프레임의 50% 미만이면 잘림(truncated)
MAX_FRAME_GAP_SEC = 0.15       # 10Hz 기준 프레임 간격이 0.15초 초과이면 결손(gap)

# 센서 글리치 기준 (D5, D6)
SPEED_GLITCH_YDS = 12.5        # s > 12.5 yd/s 프레임 제거
ACCEL_GLITCH_YDS2 = 12.0       # a > 12.0 또는 a < 0 yd/s² 프레임 제거
DECEL_GLITCH_YDS2 = 12.0       # |Δs/Δt| > 12.0 yd/s² 프레임 쌍은 감속 계산 제외

# 컴바인 트라이얼 피처 상수
MOTION_ONSET_SPEED = 0.5       # 움직임 시작 = 속도가 이 값을 처음 넘는 프레임 (yd/s)
COD_ANGLE_DEG = 60.0           # 방향 전환(Change of Direction)으로 셀 최소 각도
FORTY_CURVE_SECONDS = 5.0      # 40야드 속도 곡선 길이 (시작점 이후, 초)
FORTY_CURVE_POINTS = 50        # 리샘플링 점 수
TEN_YD_SPLIT_MIN_CORR = 0.65   # t_10yd_from_onset 검증 합격선
IFOREST_CONTAMINATION = 0.05   # Isolation Forest 이상치 비율

# 경기 타깃 상수
MOVING_SPEED = 1.0             # g_s_mean_moving 계산 기준 (yd/s)
TOP_K_PLAYS = 5                # g_smax_top5mean 의 K
TARGET_SEASON_TYPES = ["REG", "POST"]   # 주 타깃 경기 유형 (민감도 버전: PRE 포함)
MIN_PLAYS_EXPOSURE = 20        # 루키 REG+POST 트래킹 플레이 수 최소 노출 기준
EXPOSURE_WEIGHT_K = 20.0       # 노출 신뢰도 가중치 w = n / (n + K)
STARTER_TOP_FRACTION = 1 / 3   # 포지션 그룹 내 정규시즌 스냅 상위 비율 (D9)
EXPLOSIVE_TOP_FRACTION = 0.20  # 절대/시즌 코호트 상위 20% 폭발적 신입 기준
TRACKING_CHUNKSIZE = 1_000_000  # 경기 트래킹 CSV 청크 크기(행)

# ============================================================
# 4. 실험 · 분할 설정
# ============================================================
TEST_SIZE = 0.20
CV_N_SPLITS = 5
CV_N_REPEATS = 5

RESULT_COLUMNS = [
    "task", "position_group", "feature_set", "imputer",
    "model", "cv_mean", "cv_std", "n_train",
]

FIG_DPI = 300

# ============================================================
# 5. 헬퍼
# ============================================================
_VERSION_PACKAGES = [
    "numpy", "pandas", "scipy", "scikit-learn",
    "matplotlib", "seaborn", "pyarrow", "joblib", "tqdm",
]


def set_seed(seed: int = SEED) -> None:
    """random / numpy 시드 고정. sklearn 모델에는 random_state=SEED 를 따로 넘길 것."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def ensure_dirs() -> None:
    """쓰기용 폴더 생성 (원본 폴더 data/raw 는 생성하지 않음)."""
    for d in (PROCESSED_DIR, SPLITS_DIR, FIGURES_DIR, TABLES_DIR):
        d.mkdir(parents=True, exist_ok=True)


def fig_path(stage: int | str, name: str) -> Path:
    """그림 저장 경로: reports/figures/<stage>_<name>.png"""
    return FIGURES_DIR / f"{stage}_{name}.png"


def library_versions() -> dict[str, str]:
    """실행 환경 및 핵심 패키지 버전 반환."""
    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "colab": "yes" if IN_COLAB else "no",
    }
    for pkg in _VERSION_PACKAGES:
        try:
            info[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            info[pkg] = "not installed"
    return info


def print_versions() -> None:
    for k, v in library_versions().items():
        print(f"{k:>14}: {v}")


def all_raw_files() -> dict[str, Path]:
    """전체 원본 CSV 경로 딕셔너리 반환."""
    files = dict(RAW_FILES)
    for year, path in GAME_TRACKING_FILES.items():
        files[f"game_tracking_{year}"] = path
    return files


def check_raw_files(verbose: bool = True) -> list[Path]:
    """없는 원본 파일 목록을 반환하고 경고를 출력."""
    missing = [p for p in all_raw_files().values() if not p.exists()]
    if missing and verbose:
        print(f"[config] 원본 파일 {len(missing)}개 없음 (RAW_DIR={RAW_DIR}):")
        for p in missing:
            print(f"  - {p.name}")
    return missing


def setup(verbose: bool = True) -> None:
    """노트북 첫 셀에서 호출: 시드 고정 + 폴더 생성 + 원본 확인 + 버전 출력."""
    set_seed()
    ensure_dirs()
    missing = check_raw_files(verbose=True)
    if verbose:
        print(f"[config] PROJECT_ROOT = {PROJECT_ROOT}")
        print(f"[config] DATA_DIR     = {DATA_DIR}")
        print(f"[config] RAW_DIR      = {RAW_DIR} ({len(all_raw_files()) - len(missing)}/{len(all_raw_files())} CSVs ready)")
        print_versions()
