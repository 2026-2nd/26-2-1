"""GitHub main의 코드를 Drive `ML_TeamWork/code/`에 보기용 사본으로 맞춘다.

Colab: colab.setup()이 첫 셀에서 자동으로 실행한다.
내 PC: Drive 데스크톱 연결 후
    $env:ML_TEAMWORK_DIR = "G:\\내 드라이브\\ML_TeamWork"     # PowerShell
    python -m src.common.drive_sync

`code/`는 **보기용**이다. 여기서 고친 내용은 다음 갱신 때 main 내용으로 덮이고 GitHub에도 올라가지 않는다.
코드는 Colab 노트북(colab.push)이나 내 PC의 git으로 고친다 (docs/SETUP_GUIDE.md).
"""
from __future__ import annotations

import argparse
import io
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path

from . import paths

REF = "origin/main"
NOTICE = "_보기용_사본_여기서_고치지_마세요.txt"
KEEP = {NOTICE, "desktop.ini"}   # 지우지 않는 파일 (안내문, Windows Drive가 만드는 파일)


def default_dest() -> Path:
    return paths.project_dir() / "code"


def sync_tree(src: Path, dest: Path) -> dict:
    """dest를 src와 같게 만든다. 내용이 다른 파일만 복사하고, src에 없는 파일은 지운다."""
    dest.mkdir(parents=True, exist_ok=True)
    wanted = {p.relative_to(src) for p in src.rglob("*") if p.is_file()}
    copied = deleted = 0
    for rel in sorted(wanted):
        s, d = src / rel, dest / rel
        if d.is_file() and d.stat().st_size == s.stat().st_size and d.read_bytes() == s.read_bytes():
            continue
        if d.is_dir():
            shutil.rmtree(d)
        d.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(s, d)
        copied += 1
    for p in sorted(dest.rglob("*"), reverse=True):   # 깊은 곳부터
        rel = p.relative_to(dest)
        if p.is_file() and rel not in wanted and not (p.name in KEEP and p.parent == dest) and p.name != "desktop.ini":
            p.unlink()
            deleted += 1
        elif p.is_dir() and not any(c.name != "desktop.ini" for c in p.iterdir()) and not (src / rel).is_dir():
            shutil.rmtree(p)
    return {"copied": copied, "deleted": deleted}


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=paths.REPO_ROOT, capture_output=True, check=True)


def sync(ref: str = REF, dest: Path | None = None, fetch: bool = True) -> dict:
    """`ref`(기본 origin/main)의 파일을 dest(기본 Drive의 ML_TeamWork/code)에 맞춘다."""
    dest = Path(dest) if dest else default_dest()
    if paths.project_dir().resolve() == paths.REPO_ROOT.resolve():
        raise RuntimeError("Drive 위치를 모릅니다. Colab에서 실행하거나 ML_TEAMWORK_DIR를 Drive 폴더로 지정하세요.")
    if paths.REPO_ROOT.resolve() in [dest.resolve(), *dest.resolve().parents]:
        raise RuntimeError(f"저장소 안({dest})에는 사본을 만들지 않습니다.")
    if fetch:
        _git("fetch", "-q", "origin", ref.removeprefix("origin/"))
    sha = _git("rev-parse", "--short", ref).stdout.decode().strip()
    archive = _git("archive", "--format=tar", ref).stdout
    with tempfile.TemporaryDirectory() as tmp, tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        try:
            tar.extractall(tmp, filter="data")
        except TypeError:   # filter를 지원하지 않는 오래된 Python
            tar.extractall(tmp)
        result = sync_tree(Path(tmp), dest)
    notice = dest / NOTICE
    text = (f"이 폴더는 GitHub 2026-2nd/26-2-1 의 {ref} ({sha}) 보기용 사본입니다.\n"
            "여기서 고친 내용은 다음 갱신 때 덮이고 GitHub에 올라가지 않습니다.\n"
            "코드는 Colab 노트북(마지막 셀 colab.push)이나 내 PC의 git으로 고치세요. 안내: docs/SETUP_GUIDE.md\n")
    if not notice.is_file() or notice.read_text(encoding="utf-8") != text:
        notice.write_text(text, encoding="utf-8")
    return {**result, "dest": dest, "commit": sha}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="GitHub main 코드를 Drive의 ML_TeamWork/code/에 맞춘다.")
    ap.add_argument("--ref", default=REF)
    ap.add_argument("--dest", type=Path, help="기본: <ML_TEAMWORK_DIR>/code")
    a = ap.parse_args(argv)
    r = sync(a.ref, a.dest)
    print(f"{r['dest']} ← {a.ref} ({r['commit']}): 바뀐 파일 {r['copied']}개, 지운 파일 {r['deleted']}개")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
