"""라벨 zip(TL_*.zip) 안의 JSON을 읽어 '물체 1개 = 1행' 표(objects_index.csv.gz)를 만든다.

    from src.preprocess.build_index import build_index
    summary = build_index(paths.raw_dir(), out_csv, paths.configs_dir() / "labels_71385.json")

- 원본 zip은 읽기만 한다. 압축을 풀지 않는다.
- 라벨의 GPS·ADDRESS(촬영 위치)는 읽지도 저장하지도 않는다 (docs/DATA_SOURCES.md 1.3-7).
- zip 파일명의 한글은 NFD로 저장돼 있어서, 비교하기 전에 NFC로 정규화한다.
- 같은 입력이면 항상 같은 순서·같은 내용의 표가 나온다 (zip·JSON 이름 순으로 읽음).
"""
from __future__ import annotations

import csv
import gzip
import json
import shutil
import unicodedata
import zipfile
from collections import Counter
from pathlib import Path

# zip 이름에 들어 있는 수집 환경 → 짧은 코드
SOURCES = {"재활용선별장": "A", "실내형분류기": "B", "어플리케이션": "C"}

COLUMNS = [
    "object_id", "source", "env", "label_zip", "image", "obj_idx", "n_objects",
    "code", "verdict", "reason", "item", "class17", "class_id",
    "x", "y", "w", "h", "img_w", "img_h",
    "method", "date", "day", "day_night", "device", "tracking_id",
]


def nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def load_labels(path: str | Path) -> tuple[dict, list[str]]:
    """configs/labels_71385.json → (코드별 정보, 17개 클래스 순서)."""
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    return cfg["codes"], cfg["classes"]


def env_of(zip_name: str) -> str:
    """'TL_실내형분류기_B1.zip' → '실내형분류기'."""
    parts = nfc(Path(zip_name).stem).split("_")
    return parts[1] if len(parts) > 1 else ""


def find_label_zips(raw_dir: str | Path, envs: list[str] | None = None) -> list[Path]:
    """raw 폴더 아래의 라벨 zip(TL_*.zip)을 찾는다. envs를 주면 그 환경만."""
    zips = [p for p in Path(raw_dir).rglob("*.zip") if nfc(p.name).startswith("TL_")]
    if envs:
        zips = [p for p in zips if env_of(p.name) in envs]
    return sorted(zips, key=lambda p: nfc(p.name))


def _resolution(text) -> tuple[str, str]:
    """'1280/960' → ('1280', '960'). 형식이 다르면 빈 값."""
    try:
        w, h = str(text).replace("*", "/").replace("x", "/").split("/")[:2]
        return str(int(float(w))), str(int(float(h)))
    except (ValueError, TypeError):
        return "", ""


def build_index(raw_dir, out_csv, labels_json, envs=None, cache_dir="/content/label_cache",
                log=print) -> dict:
    """라벨 zip을 모두 읽어 out_csv(.csv.gz)에 저장하고 요약(Counter들)을 돌려준다.

    raw_dir      paths.raw_dir()
    out_csv      저장할 파일 (예: processed/build_index/objects_index.csv.gz)
    labels_json  configs/labels_71385.json
    envs         ["실내형분류기"]처럼 일부 환경만 읽을 때. None이면 전부
    cache_dir    라벨 zip을 복사해 둘 로컬 폴더 (Drive에서 직접 읽으면 느림). None이면 복사 안 함
    """
    codes, classes = load_labels(labels_json)
    class_id = {c: i for i, c in enumerate(classes)}
    zips = find_label_zips(raw_dir, envs)
    if not zips:
        raise FileNotFoundError(f"{raw_dir} 아래에서 라벨 zip(TL_*.zip)을 찾지 못했습니다.")

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if cache_dir:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)

    summary = {"objects": Counter(), "images": Counter(), "bad_json": Counter(),
               "unknown_code": Counter(), "no_object_image": Counter()}

    with gzip.open(out_csv, "wt", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for zp in zips:
            zname = nfc(zp.name)
            env = env_of(zname)
            source = SOURCES.get(env, "?")
            src = zp
            if cache_dir:
                src = Path(cache_dir) / zname
                if not src.exists() or src.stat().st_size != zp.stat().st_size:
                    shutil.copy(zp, src)
            n_before = sum(summary["objects"].values())
            with zipfile.ZipFile(src) as zf:
                names = sorted(n for n in zf.namelist() if n.lower().endswith(".json"))
                for name in names:
                    try:
                        d = json.loads(zf.read(name).decode("utf-8-sig"))
                    except (ValueError, UnicodeDecodeError):
                        summary["bad_json"][source] += 1
                        continue
                    info = d.get("Info") or {}          # GPS·ADDRESS는 꺼내지 않는다
                    objs = d.get("objects") or []
                    image = nfc(str(d.get("Image") or Path(name).with_suffix(".jpg").name))
                    img_w, img_h = _resolution(info.get("RESOLUTION"))
                    date = str(info.get("DATE") or "")
                    summary["images"][source] += 1
                    if not objs:
                        summary["no_object_image"][source] += 1
                    for i, o in enumerate(objs):
                        code = o.get("class_name")
                        lab = codes.get(code)
                        if lab is None:
                            summary["unknown_code"][code] += 1
                            continue
                        c17 = f"{lab['verdict']}-{lab['item']}"
                        coord = (o.get("annotation") or {}).get("coord") or {}
                        writer.writerow({
                            "object_id": f"{Path(image).stem}#{i}", "source": source, "env": env,
                            "label_zip": zname, "image": image, "obj_idx": i, "n_objects": len(objs),
                            "code": code, "verdict": lab["verdict"], "reason": lab["reason"] or "",
                            "item": lab["item"], "class17": c17, "class_id": class_id[c17],
                            "x": coord.get("x"), "y": coord.get("y"),
                            "w": coord.get("width"), "h": coord.get("height"),
                            "img_w": img_w, "img_h": img_h,
                            "method": info.get("METHOD", ""), "date": date, "day": date[:10],
                            "day_night": info.get("DAY/NIGHT", ""), "device": info.get("DEVICE", ""),
                            "tracking_id": o.get("tracking_id", ""),
                        })
                        summary["objects"][source] += 1
            log(f"{zname}: 물체 {sum(summary['objects'].values()) - n_before:,}개")
    log(f"저장: {out_csv}")
    return summary
