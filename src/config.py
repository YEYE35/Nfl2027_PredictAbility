"""프로젝트 공용 설정: 실행 환경 감지 · 경로 · 시드 · 임계값.

모든 노트북 첫 셀 (Colab):
    from google.colab import drive
    drive.mount("/content/drive")
    %cd /content/drive/MyDrive/nfl-bdb2027        # 각자 Drive에 clone한 저장소
    import sys; sys.path.insert(0, ".")
    from src import config
    config.setup()

경로 규칙
    - 원본·중간 산출물(data/raw, data/processed): Google Drive 공유 폴더
      (Colab VM은 세션이 끝나면 지워지므로 산출물은 반드시 Drive에 저장)
    - 테스트 분할·그림·결과표: 저장소 안 (git으로 공유)
    - 환경변수 NFL_DATA_DIR 를 지정하면 데이터 위치를 덮어쓴다.
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
# 1. 실행 환경
# ============================================================
IN_COLAB: bool = "COLAB_RELEASE_TAG" in os.environ or "google.colab" in sys.modules
COLAB_RUNTIME_VERSION = "2026.10"  # requirements.txt 기준 런타임

# ============================================================
# 2. 경로
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DRIVE_ROOT = Path("/content/drive/MyDrive")
# 공유받은 사람은 Drive에서 "바로가기 추가 → 내 드라이브"로 같은 이름이 되게 할 것
DRIVE_DATA_DIR_NAME = "nfl-bdb2027-data"


def _resolve_data_dir() -> Path:
    if "NFL_DATA_DIR" in os.environ:
        return Path(os.environ["NFL_DATA_DIR"]).expanduser()
    if IN_COLAB:
        return DRIVE_ROOT / DRIVE_DATA_DIR_NAME
    return PROJECT_ROOT / "data"


DATA_DIR = _resolve_data_dir()
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
TEST_IDS_PATH = SPLITS_DIR / "test_ids.csv"

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
TABLES_DIR = REPORTS_DIR / "tables"

# --- 원본 CSV ---
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

# --- 중간 산출물 (계획서 2장 데이터 계약) ---
PROCESSED_FILES = {
    "combine_trials": PROCESSED_DIR / "combine_trials.parquet",            # T1 (A)
    "combine_trial_flags": PROCESSED_DIR / "combine_trial_flags.parquet",  # (A)
    "combine_forty_curves": PROCESSED_DIR / "combine_forty_curves.parquet",  # (A)
    "combine_player_features": PROCESSED_DIR / "combine_player_features.parquet",  # T2 (A→B)
    "game_play_peaks": PROCESSED_DIR / "game_play_peaks.parquet",          # (B)
    "game_rookie_athletic": PROCESSED_DIR / "game_rookie_athletic.parquet",  # T3 (B→A)
    "game_rookie_role": PROCESSED_DIR / "game_rookie_role.parquet",        # T4 (B→A)
    "master_player_table": PROCESSED_DIR / "master_player_table.parquet",  # T5 (A+B)
}

# ============================================================
# 3. 재현성
# ============================================================
SEED = 42

# ============================================================
# 4. 도메인 상수 (Stage 1 EDA 후 확정 — 바꾸면 상대에게 알릴 것)
# ============================================================
POSITION_GROUPS = ["DB", "OL", "DL", "WR", "TE"]

# 센서 글리치 임계값 (후보값)
SPEED_GLITCH_YDS = 12.5        # yd/s 초과 프레임 제거
ACCEL_GLITCH_YDS2 = 12.0       # yd/s² 초과 프레임 제거

# 컴바인 트라이얼 특징
MOTION_ONSET_SPEED = 0.5       # 움직임 시작 = 속도가 이 값을 처음 넘는 프레임
COD_ANGLE_DEG = 60.0           # 방향 전환으로 셀 최소 각도
FORTY_CURVE_SECONDS = 5.0      # 40야드 속도 곡선 길이(시작점 이후)
FORTY_CURVE_POINTS = 50        # 리샘플링 점 수
TEN_YD_SPLIT_MIN_CORR = 0.65   # t_10yd_from_onset 검증 합격선
IFOREST_CONTAMINATION = 0.025  # Isolation Forest 이상치 비율

# 경기 타깃
MOVING_SPEED = 1.0             # g_s_mean_moving 계산 기준 (yd/s)
TOP_K_PLAYS = 5                # g_smax_top5mean 의 K
MIN_PLAYS_EXPOSURE = 50        # 미만이면 low_exposure 플래그
STARTER_TOP_FRACTION = 1 / 3   # 포지션 그룹 내 정규시즌 스냅 상위 비율
TRACKING_CHUNKSIZE = 1_000_000  # 경기 트래킹 CSV chunk 크기(행)

# ============================================================
# 5. 실험 설정 (계획서 5장)
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
# 6. 헬퍼
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
    """쓰기용 폴더 생성 (원본 폴더는 만들지 않음)."""
    for d in (PROCESSED_DIR, SPLITS_DIR, FIGURES_DIR, TABLES_DIR):
        d.mkdir(parents=True, exist_ok=True)


def fig_path(stage: int | str, name: str) -> Path:
    """그림 저장 경로: reports/figures/<stage>_<name>.png"""
    return FIGURES_DIR / f"{stage}_{name}.png"


def library_versions() -> dict[str, str]:
    """보고서 1절용 실행 환경 정보."""
    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "colab": f"yes (runtime {os.environ.get('COLAB_RELEASE_TAG', '?')})" if IN_COLAB else "no",
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


def check_raw_files() -> list[Path]:
    """없는 원본 파일 목록을 반환하고 경고를 출력."""
    files = list(RAW_FILES.values()) + list(GAME_TRACKING_FILES.values())
    missing = [p for p in files if not p.exists()]
    if missing:
        print(f"[config] 원본 파일 {len(missing)}개 없음 (RAW_DIR={RAW_DIR}):")
        for p in missing:
            print(f"  - {p.name}")
    return missing


def setup(verbose: bool = True) -> None:
    """노트북 첫 셀에서 호출: 시드 고정 + 폴더 생성 + 원본 확인 + 버전 출력.

    Colab에서는 이 함수를 부르기 전에 drive.mount() 가 되어 있어야 한다.
    """
    if IN_COLAB and not DRIVE_ROOT.exists():
        raise RuntimeError("Google Drive가 마운트되지 않았습니다. drive.mount('/content/drive') 먼저 실행.")
    set_seed()
    ensure_dirs()
    check_raw_files()
    if verbose:
        print(f"[config] PROJECT_ROOT = {PROJECT_ROOT}")
        print(f"[config] DATA_DIR     = {DATA_DIR}")
        print_versions()