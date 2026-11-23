"""AI Hub 데이터 다운로드·압축 해제 도구 (표준 라이브러리만 사용).

AI Hub 공식 `aihubshell` v0.6 (https://api.aihub.or.kr/api/aihubshell.do)과 같은 API를
호출한다. bash가 없는 Windows나 Colab의 Python 코드에서도 쓸 수 있다.
AI Hub 서버는 이어받기(Range)를 지원하지 않아, 연결이 끊기면 그 파일을 처음부터 다시 받는다(최대 10번). 공식 셸이 쓰는
엔드포인트가 바뀌면 `DOWN_URL`을 셸 스크립트에 맞춰 고친다.

사용 예 (저장소 루트에서 실행):
    python -m src.aihub_download list
    python -m src.aihub_download download --preset smoke_test
    python -m src.aihub_download download --filekey 514447,514448
    python -m src.aihub_download extract --pattern "*실내형분류기*" --dest /content/data/aihub_71385

API 키는 환경변수 `AIHUB_APIKEY`로만 받는다. 코드·노트북·커밋에 키를 적지 않는다.
데이터 위치는 환경변수 `ML_TEAMWORK_DATA`로 정하고, 없으면 저장소의 `data/`를 쓴다.
이용 조건은 docs/DATA_SOURCES.md를 따른다.
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import fnmatch
import hashlib
import http.client
import json
import os
import re
import shutil
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

BASE_URL = "https://api.aihub.or.kr"
DOWN_URL = BASE_URL + "/down/0.6/{dataset_key}.do"
INFO_URL = BASE_URL + "/info/{dataset_key}.do"

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "aihub_71385.json"
MANIFEST_NAME = "MANIFEST.csv"
MANIFEST_FIELDS = ["downloaded_at", "filekey", "path", "size_bytes", "sha256"]
CHUNK = 1024 * 1024


class AIHubError(RuntimeError):
    pass


# ── 설정·경로 ─────────────────────────────────────────────────────────
def load_config(path: Path = DEFAULT_CONFIG) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def data_root() -> Path:
    return Path(os.environ.get("ML_TEAMWORK_DATA") or REPO_ROOT / "data")


def raw_dir(config: dict) -> Path:
    return data_root() / config["raw_subdir"]


def resolve_filekeys(config: dict, presets: list[str], filekeys: list[str]) -> list[int]:
    keys: list[int] = []
    for p in presets:
        if p not in config["presets"]:
            raise AIHubError(f"알 수 없는 preset: {p} (가능: {', '.join(config['presets'])})")
        keys += config["presets"][p]["filekeys"]
    for item in filekeys:
        keys += [int(k) for k in item.split(",") if k.strip()]
    seen, ordered = set(), []
    for k in keys:
        if k not in seen:
            seen.add(k)
            ordered.append(k)
    return ordered


_KEY_RE = re.compile(r"[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}")


def get_api_key() -> str:
    """AIHUB_APIKEY에서 키를 꺼낸다. 앞뒤에 '키:'나 한글 등이 섞여 있으면 키 형식 부분만 쓴다."""
    raw = os.environ.get("AIHUB_APIKEY", "").strip()
    if not raw:
        raise AIHubError("환경변수 AIHUB_APIKEY가 없습니다. Colab은 Secrets에, 로컬은 환경변수에 등록하세요.")
    m = _KEY_RE.search(raw)
    if m:
        return m.group(0)
    if not raw.isascii():
        raise AIHubError("API 키에 한글 등 영문·숫자가 아닌 문자가 들어 있습니다. 한/영 키를 영문으로 두고 "
                         "키(예: XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX)만 다시 붙여 넣으세요.")
    return raw


# ── API 호출 ─────────────────────────────────────────────────────────
def fetch_file_tree(dataset_key: str) -> str:
    # AI Hub info API는 정상 내용을 HTTP 502와 함께 보내기도 한다 (curl은 상태 코드를 무시해서 드러나지 않음).
    try:
        with urllib.request.urlopen(INFO_URL.format(dataset_key=dataset_key), timeout=60) as r:
            return r.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        if body.strip():
            return body
        raise AIHubError(f"파일 목록 조회 실패 HTTP {e.code}") from None


RETRIES = 10
# 연결이 중간에 끊겼을 때 다시 시도할 오류 (HTTP 오류 응답은 제외)
_TRANSIENT = (http.client.IncompleteRead, http.client.RemoteDisconnected, ConnectionError, TimeoutError,
              urllib.error.URLError, OSError)


def _download_once(url: str, api_key: str, dest: Path) -> None:
    """공식 셸의 `curl -L -H "apikey:..." "$DOWN_URL?fileSn=..."`과 같은 요청.

    AI Hub 서버는 Range(이어받기) 요청을 무시하고 항상 처음부터 보낸다 (2026-11-23 확인). 그래서 매번 새로 쓴다.
    """
    req = urllib.request.Request(url, headers={"apikey": api_key})
    try:
        resp = urllib.request.urlopen(req, timeout=120)
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace").strip()
        hint = ""
        if "인증" in body or "권한" in body:
            hint = " → AI Hub에서 이 데이터셋의 다운로드 신청이 승인됐는지, API 키가 맞는지 확인하세요."
        raise AIHubError(f"다운로드 실패 HTTP {e.code}: {body}{hint}") from None
    with resp, open(dest, "wb") as f:
        done, last = 0, 0
        while chunk := resp.read(CHUNK):
            f.write(chunk)
            done += len(chunk)
            if done - last >= 256 * 2**20:   # 256MB마다 진행 상황 출력
                print(f"  {done / 2**30:,.2f} GB", flush=True)
                last = done


def _download_tar(dataset_key: str, filekeys: list[int], api_key: str, dest: Path,
                  retries: int = RETRIES, wait: float = 10.0) -> None:
    """연결이 끊기면 처음부터 다시 받는다 (최대 retries회, 대기 시간은 점점 늘림)."""
    url = DOWN_URL.format(dataset_key=dataset_key) + "?fileSn=" + ",".join(map(str, filekeys))
    for attempt in range(1, retries + 1):
        try:
            _download_once(url, api_key, dest)
            return
        except AIHubError:
            raise
        except _TRANSIENT as e:
            size = dest.stat().st_size / 2**20 if dest.exists() else 0
            if attempt == retries:
                raise AIHubError(f"연결이 {retries}번 끊겨 중단했습니다 (마지막 오류: {e!r})") from None
            delay = min(wait * 2 ** (attempt - 1), 300)
            print(f"  ⚠️ 연결 끊김 ({size:,.0f} MB에서, {type(e).__name__}). {delay:.0f}초 후 처음부터 다시 받습니다 "
                  f"({attempt}/{retries - 1})", flush=True)
            time.sleep(delay)


# ── 압축 처리 ────────────────────────────────────────────────────────
def _is_within(base: Path, target: Path) -> bool:
    base, target = base.resolve(), target.resolve()
    return target == base or base in target.parents


def safe_extract_tar(tar_path: Path, dest: Path) -> None:
    with tarfile.open(tar_path) as tf:
        members = tf.getmembers()
        for m in members:
            if not _is_within(dest, dest / m.name) or m.issym() or m.islnk():
                raise AIHubError(f"tar 안에 허용되지 않는 경로가 있습니다: {m.name}")
        if hasattr(tarfile, "data_filter"):  # Python 3.12+ (3.8~3.11 일부 패치 버전 포함)
            tf.extractall(dest, members=members, filter="data")
        else:
            tf.extractall(dest, members=members)


_PART_RE = re.compile(r"^(?P<prefix>.+)\.part(?P<offset>\d+)$")


def merge_parts(root: Path) -> list[Path]:
    """`X.zip.part0`, `X.zip.part1073741824`… 를 숫자 순으로 이어 붙여 `X.zip`을 만들고 part를 지운다."""
    groups: dict[Path, list[tuple[int, Path]]] = {}
    for p in root.rglob("*.part*"):
        m = _PART_RE.match(p.name)
        if m and p.is_file():
            groups.setdefault(p.with_name(m["prefix"]), []).append((int(m["offset"]), p))
    merged = []
    for target, parts in groups.items():
        parts.sort()
        with open(target, "wb") as out:
            for _, part in parts:
                with open(part, "rb") as src:
                    shutil.copyfileobj(src, out, CHUNK)
        for _, part in parts:
            part.unlink()
        merged.append(target)
    return merged


def sha256sum(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(CHUNK):
            h.update(chunk)
    return h.hexdigest()


def _append_manifest(raw: Path, rows: list[dict]) -> None:
    manifest = raw / MANIFEST_NAME
    new = not manifest.exists()
    with open(manifest, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        if new:
            w.writeheader()
        w.writerows(rows)


def downloaded_filekeys(raw: Path) -> set[int]:
    """MANIFEST.csv에 기록되어 있고 파일도 실제로 남아 있는 filekey."""
    manifest = raw / MANIFEST_NAME
    if not manifest.exists():
        return set()
    with open(manifest, encoding="utf-8") as f:
        return {int(r["filekey"]) for r in csv.DictReader(f) if (raw / r["path"]).exists()}


def download(config: dict, filekeys: list[int], work_dir: Path | None = None, force: bool = False) -> list[Path]:
    """filekey를 하나씩 받아 병합하고 zip을 raw 폴더로 옮긴 뒤 MANIFEST.csv에 크기·해시를 기록한다."""
    api_key = get_api_key()
    raw = raw_dir(config)
    raw.mkdir(parents=True, exist_ok=True)
    work_root = Path(work_dir or tempfile.mkdtemp(prefix="aihub_"))
    saved: list[Path] = []
    done = downloaded_filekeys(raw)
    for fk in filekeys:
        print(f"=== filekey {fk} ===")
        if fk in done and not force:
            print(f"  이미 받음 (MANIFEST.csv), 건너뜀")
            continue
        work = work_root / str(fk)
        work.mkdir(parents=True, exist_ok=True)
        tar_path = work / "download.tar"
        _download_tar(config["dataset_key"], [fk], api_key, tar_path)
        extract_dir = work / "x"
        safe_extract_tar(tar_path, extract_dir)
        tar_path.unlink()
        merge_parts(extract_dir)
        rows = []
        files = [p for p in extract_dir.rglob("*") if p.is_file()]
        if not files:
            raise AIHubError(f"filekey {fk}: 받은 파일이 없습니다.")
        for f in files:
            dst = raw / f.relative_to(extract_dir)
            if dst.exists() and not force:
                print(f"  이미 있음, 건너뜀: {dst}")
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(f), dst)
            rows.append({
                "downloaded_at": _dt.datetime.now().isoformat(timespec="seconds"),
                "filekey": fk,
                "path": dst.relative_to(raw).as_posix(),
                "size_bytes": dst.stat().st_size,
                "sha256": sha256sum(dst),
            })
            saved.append(dst)
            print(f"  saved {dst}")
        _append_manifest(raw, rows)
        shutil.rmtree(work, ignore_errors=True)
    return saved


def _zip_member_name(info: zipfile.ZipInfo) -> str:
    """UTF-8 플래그가 없는 zip(한국어 Windows 생성)은 cp949로 다시 해석한다."""
    name = info.filename
    if not info.flag_bits & 0x800:
        try:
            name = name.encode("cp437").decode("cp949")
        except UnicodeError:
            pass
    # AI Hub zip은 항목 이름이 '/B2_20220721_000024.json'처럼 '/'로 시작한다. 앞의 구분자만 떼고,
    # '../' 같은 경로는 그대로 두어 extract_zips의 경로 검사에서 막히게 한다.
    return name.lstrip("/\\")


def extract_zips(config: dict, pattern: str, dest: Path, overwrite: bool = False) -> list[Path]:
    """raw 폴더의 zip 중 pattern(파일 이름 기준 glob)에 맞는 것을 dest/<zip 이름>/ 아래에 푼다."""
    raw = raw_dir(config)
    out_dirs = []
    for z in sorted(raw.rglob("*.zip")):
        if not fnmatch.fnmatch(z.name, pattern):
            continue
        out = dest / z.stem
        if out.exists() and not overwrite:
            print(f"skip (이미 있음) {out}")
            out_dirs.append(out)
            continue
        with zipfile.ZipFile(z) as zf:
            for info in zf.infolist():
                target = out / _zip_member_name(info)
                if not _is_within(out, target):
                    raise AIHubError(f"{z.name}: 허용되지 않는 경로 {info.filename}")
                if info.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info) as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst, CHUNK)
        print(f"extracted {out}")
        out_dirs.append(out)
    if not out_dirs:
        print(f"'{pattern}'에 맞는 zip이 {raw}에 없습니다.")
    return out_dirs


# ── CLI ──────────────────────────────────────────────────────────────
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="AI Hub 데이터 다운로드·압축 해제")
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="AI Hub 파일 목록(파일명 | 용량 | filekey) 출력")
    sub.add_parser("presets", help="설정 파일의 preset 목록 출력")

    d = sub.add_parser("download", help="filekey를 받아 raw 폴더에 저장")
    d.add_argument("--preset", action="append", default=[])
    d.add_argument("--filekey", action="append", default=[], help="쉼표로 여러 개 가능")
    d.add_argument("--work-dir", type=Path, help="임시 작업 폴더 (기본: 시스템 임시 폴더)")
    d.add_argument("--force", action="store_true", help="이미 있는 파일도 덮어씀")

    e = sub.add_parser("extract", help="raw 폴더의 zip을 학습용 폴더에 압축 해제")
    e.add_argument("--pattern", default="*.zip", help="zip 파일 이름 glob, 예: '*실내형분류기*'")
    e.add_argument("--dest", type=Path, required=True)
    e.add_argument("--overwrite", action="store_true")

    args = ap.parse_args(argv)
    config = load_config(args.config)
    try:
        if args.cmd == "list":
            print(fetch_file_tree(config["dataset_key"]))
        elif args.cmd == "presets":
            for name, p in config["presets"].items():
                print(f"{name:20s} {len(p['filekeys']):3d} files  {p['description']}")
        elif args.cmd == "download":
            keys = resolve_filekeys(config, args.preset, args.filekey)
            if not keys:
                ap.error("--preset 또는 --filekey를 하나 이상 지정하세요.")
            print(f"저장 위치: {raw_dir(config)}  ({len(keys)} files)")
            download(config, keys, args.work_dir, args.force)
        elif args.cmd == "extract":
            extract_zips(config, args.pattern, args.dest, args.overwrite)
    except AIHubError as err:
        print(f"오류: {err}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
