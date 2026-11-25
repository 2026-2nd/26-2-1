"""데이터·모델·결과를 어디에 두는지 한 곳에서 정한다.

    from src.common import paths
    paths.data_dir()      # 데이터 (원본 zip·전처리 결과)
    paths.models_dir()    # 학습한 모델 파일
    paths.outputs_dir()   # 그 밖의 큰 출력물
    paths.run_dir("knn")  # 실험 기록 폴더 (Git에 올리는 작은 파일만)

위치를 정하는 순서:
  1. 환경변수 `ML_TEAMWORK_DIR`가 있으면 그 폴더 (예: 국내 PC에서 "G:/내 드라이브/ML_TeamWork")
  2. Colab에서 Drive가 연결돼 있으면 `/content/drive/MyDrive/ML_TeamWork`
  3. 둘 다 아니면 저장소 폴더 (data/, models/, outputs/는 .gitignore로 Git에서 빠진다)

데이터 폴더만 따로 바꾸려면 `ML_TEAMWORK_DATA`를 쓴다 (AI Hub 다운로드 도구와 같은 이름).
개인 PC·Drive의 절대 경로를 코드에 직접 적지 않는다.
"""
from __future__ import annotations

import datetime as _dt
import os
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
COLAB_DRIVE_DIR = Path("/content/drive/MyDrive/ML_TeamWork")


def project_dir() -> Path:
    """데이터·모델·출력물을 담는 상위 폴더."""
    env = os.environ.get("ML_TEAMWORK_DIR")
    if env:
        return Path(env)
    if COLAB_DRIVE_DIR.is_dir():
        return COLAB_DRIVE_DIR
    return REPO_ROOT


def data_dir() -> Path:
    env = os.environ.get("ML_TEAMWORK_DATA")
    return Path(env) if env else project_dir() / "data"


def raw_dir() -> Path:
    """원본 데이터. 직접 수정하지 않는다."""
    return data_dir() / "raw"


def processed_dir() -> Path:
    """전처리 결과. 코드로 다시 만들 수 있어야 한다."""
    return data_dir() / "processed"


def models_dir() -> Path:
    return project_dir() / "models"


def outputs_dir() -> Path:
    return project_dir() / "outputs"


def configs_dir() -> Path:
    return REPO_ROOT / "configs"


def run_dir(name: str, now: _dt.datetime | None = None) -> Path:
    """실험 한 번의 기록 폴더 `reports/experiments/<날짜-시각>_<name>/`을 만들어 돌려준다.

    여기에는 설정·점수·작은 그림처럼 Git에 올릴 파일만 둔다. 모델 파일은 models_dir()에 둔다.
    """
    safe = re.sub(r"[^0-9A-Za-z._-]+", "-", name).strip("-") or "run"
    stamp = (now or _dt.datetime.now()).strftime("%Y%m%d-%H%M%S")
    path = REPO_ROOT / "reports" / "experiments" / f"{stamp}_{safe}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def describe() -> str:
    """현재 경로 설정을 사람이 읽기 좋게 보여 준다."""
    rows = [("저장소(코드)", REPO_ROOT), ("데이터", data_dir()), ("모델", models_dir()), ("출력물", outputs_dir())]
    return "\n".join(f"  {k:<10} {v}{'' if Path(v).exists() else '  (아직 없음)'}" for k, v in rows)
