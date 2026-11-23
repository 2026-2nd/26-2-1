"""AI Hub 71385 라벨 → 물건 단위 표 → 표본 추출 → 그룹 분할 → crop 이미지 (기획서 6장·8장).

    python -m src.dataset index                 # 라벨 zip을 읽어 objects.csv (박스 1개 = 1행)
    python -m src.dataset build --per-code 1000 # 표본 추출 + 분할 + crop 저장 + manifest.csv

zip을 풀지 않고 바로 읽는다. 입력은 $ML_TEAMWORK_DATA/raw/aihub_71385/ 아래의 zip,
출력은 $ML_TEAMWORK_DATA/processed/<version>/ 이다 (Git 제외).

누수 방지 (기획서 8장 체크리스트):
  - 분할 단위는 그룹이다. 기본 그룹 = 수집 방법 + 촬영일 (예: B2_20220721).
    같은 사진에서 자른 물건과 인접 프레임은 같은 그룹이라 같은 분할에 들어간다.
  - 층화 기준은 1단계 판정(verdict), 비율은 70:15:15 (StratifiedGroupKFold로 근사).
  - crop의 dHash가 같은데 분할이 다른 쌍이 있으면 val/test 쪽을 제외하고 개수를 기록한다.
  - 메타데이터의 GPS·주소는 읽지 않고 저장하지 않는다. 파일명·수집 환경은 입력 피처로 쓰지 않는다.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
LABEL_CONFIG = REPO_ROOT / "configs" / "labels_71385.json"
DEFAULT_VERSION = "crops_v1"
CROP_SIZE = 224
PAD_RATIO = 0.10
MIN_SIDE = 32
SPLIT_RATIOS = (0.70, 0.15, 0.15)

_ZIP_RE = re.compile(r"^(?P<kind>T[SL]|V[SL])_(?P<env>[^_]+)_(?P<part>.+)\.zip$")
_GROUP_RE = re.compile(r"^(?P<method>[^_]+(?:_\d)?)_(?P<date>\d{8})_")


def data_root() -> Path:
    return Path(os.environ.get("ML_TEAMWORK_DATA") or REPO_ROOT / "data")


def raw_dir() -> Path:
    return data_root() / "raw" / "aihub_71385"


def processed_dir(version: str = DEFAULT_VERSION) -> Path:
    return data_root() / "processed" / version


def load_label_config(path: Path = LABEL_CONFIG) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _zip_name(info: zipfile.ZipInfo) -> str:
    name = info.filename
    if not info.flag_bits & 0x800:
        try:
            name = name.encode("cp437").decode("cp949")
        except UnicodeError:
            pass
    return name.lstrip("/\\")


def find_zips(root: Path, kinds: tuple[str, ...]) -> list[tuple[Path, str]]:
    """(zip 경로, 수집 환경) 목록. kinds 예: ("TL", "VL")."""
    out = []
    for z in sorted(root.rglob("*.zip")):
        m = _ZIP_RE.match(z.name)
        if m and m["kind"] in kinds:
            out.append((z, m["env"]))
    return out


def group_of(image_name: str, mode: str = "day") -> str:
    stem = Path(image_name).stem
    if mode == "image":
        return stem
    m = _GROUP_RE.match(stem)
    return f"{m['method']}_{m['date']}" if m else stem


# ── 1. 라벨 → objects.csv ────────────────────────────────────────────
def parse_label(data: dict, json_name: str, env: str, codes: dict) -> list[dict]:
    info = data.get("Info") or {}
    try:
        img_w, img_h = (int(v) for v in str(info.get("RESOLUTION", "")).split("/"))
    except ValueError:
        img_w = img_h = None
    image = data.get("Image") or Path(json_name).with_suffix(".jpg").name
    rows = []
    for k, obj in enumerate(data.get("objects") or []):
        if obj.get("annotation_type", "box") != "box":
            continue
        c = (obj.get("annotation") or {}).get("coord") or {}
        code = obj.get("class_name")
        lab = codes.get(code)
        rows.append({
            "object_id": f"{Path(image).stem}#{k}",
            "image": Path(image).name,
            "env": env,
            "method": info.get("METHOD"),
            "code": code,
            "verdict": lab["verdict"] if lab else None,
            "reason": lab["reason"] if lab else None,
            "item": lab["item"] if lab else None,
            "x": c.get("x"), "y": c.get("y"), "w": c.get("width"), "h": c.get("height"),
            "img_w": img_w, "img_h": img_h,
        })
    return rows


def build_index(root: Path | None = None) -> pd.DataFrame:
    root = root or raw_dir()
    cfg = load_label_config()
    rows, unknown = [], Counter()
    zips = find_zips(root, ("TL", "VL"))
    if not zips:
        raise SystemExit(f"라벨 zip(TL_/VL_)이 {root}에 없습니다. aihub_download로 labels_all을 받으세요.")
    for z, env in zips:
        with zipfile.ZipFile(z) as zf:
            for info in zf.infolist():
                name = _zip_name(info)
                if not name.endswith(".json"):
                    continue
                data = json.loads(zf.read(info).decode("utf-8-sig"))
                for r in parse_label(data, name, env, cfg["codes"]):
                    if r["verdict"] is None:
                        unknown[r["code"]] += 1
                    rows.append(r)
        print(f"  {z.name}: 누적 {len(rows):,} objects")
    if unknown:
        print(f"⚠️ 설정에 없는 라벨 코드 (제외): {dict(unknown)}")
    df = pd.DataFrame(rows)
    return df[df["verdict"].notna()].reset_index(drop=True)


# ── 2. 표본 추출과 분할 ──────────────────────────────────────────────
def filter_and_sample(df: pd.DataFrame, per_code: int, seed: int, min_side: int = MIN_SIDE,
                      envs: list[str] | None = None) -> pd.DataFrame:
    d = df[(df["w"] >= min_side) & (df["h"] >= min_side)]
    if envs:
        d = d[d["env"].isin(envs)]
    parts = [g.sample(n=min(len(g), per_code), random_state=seed) for _, g in d.groupby("code")]
    return pd.concat(parts).sort_values("object_id").reset_index(drop=True)


def group_split(df: pd.DataFrame, seed: int, ratios=SPLIT_RATIOS, group_col: str = "group",
                strat_col: str = "verdict") -> pd.Series:
    """그룹을 깨지 않고 verdict 비율을 맞춘 train/val/test. StratifiedGroupKFold 두 번으로 근사한다."""
    from sklearn.model_selection import StratifiedGroupKFold

    n_groups = df[group_col].nunique()
    if n_groups < 7:
        raise ValueError(f"그룹이 {n_groups}개뿐이라 70:15:15 그룹 분할을 할 수 없습니다. --group-by image 를 검토하세요.")
    split = pd.Series("train", index=df.index)
    k_test = max(2, round(1 / ratios[2]))
    sgkf = StratifiedGroupKFold(n_splits=k_test, shuffle=True, random_state=seed)
    _, test_idx = next(sgkf.split(df, df[strat_col], df[group_col]))
    split.iloc[test_idx] = "test"
    rest = df.index[split == "train"]
    k_val = max(2, round((1 - ratios[2]) / ratios[1]))
    sgkf = StratifiedGroupKFold(n_splits=k_val, shuffle=True, random_state=seed + 1)
    sub = df.loc[rest]
    _, val_idx = next(sgkf.split(sub, sub[strat_col], sub[group_col]))
    split.loc[rest[val_idx]] = "val"
    return split


# ── 3. crop ──────────────────────────────────────────────────────────
def crop_box(img: np.ndarray, x: float, y: float, w: float, h: float, pad: float = PAD_RATIO,
             size: int = CROP_SIZE) -> np.ndarray:
    """박스에 여백 pad를 두고, 정사각형으로 넓혀(종횡비 유지) size×size로 줄인다."""
    import cv2
    H, W = img.shape[:2]
    cx, cy = x + w / 2, y + h / 2
    side = max(w, h) * (1 + 2 * pad)
    x0, y0 = int(round(cx - side / 2)), int(round(cy - side / 2))
    x1, y1 = int(round(cx + side / 2)), int(round(cy + side / 2))
    # 이미지 밖은 가장자리 값으로 채운다
    crop = cv2.copyMakeBorder(
        img[max(y0, 0):min(y1, H), max(x0, 0):min(x1, W)],
        max(0, -y0), max(0, y1 - H), max(0, -x0), max(0, x1 - W), cv2.BORDER_REPLICATE)
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)


def dhash(img: np.ndarray, n: int = 8) -> str:
    import cv2
    g = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (n + 1, n), interpolation=cv2.INTER_AREA)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return f"{int(''.join('1' if b else '0' for b in bits), 2):0{n * n // 4}x}"


def image_index(root: Path) -> dict[str, tuple[Path, zipfile.ZipInfo]]:
    idx = {}
    for z, _ in find_zips(root, ("TS", "VS")):
        with zipfile.ZipFile(z) as zf:
            for info in zf.infolist():
                name = _zip_name(info)
                if name.lower().endswith((".jpg", ".jpeg", ".png")):
                    idx[Path(name).name] = (z, info)
    return idx


def make_crops(df: pd.DataFrame, out_dir: Path, root: Path | None = None) -> pd.DataFrame:
    import cv2
    root = root or raw_dir()
    idx = image_index(root)
    missing = sorted(set(df["image"]) - set(idx))
    if missing:
        print(f"⚠️ 원천 이미지가 없는 사진 {len(missing):,}장 → 해당 물건 제외 (예: {missing[:3]})")
    df = df[df["image"].isin(idx)].copy()
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    paths, hashes = [], []
    open_zips: dict[Path, zipfile.ZipFile] = {}
    try:
        for n, (image, g) in enumerate(df.groupby("image", sort=True), 1):
            z, info = idx[image]
            zf = open_zips.setdefault(z, zipfile.ZipFile(z))
            buf = np.frombuffer(zf.read(info), np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            for i, r in g.iterrows():
                rel = f"images/{r['object_id'].replace('#', '_')}.jpg"
                crop = crop_box(img, r["x"], r["y"], r["w"], r["h"])
                cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tofile(str(out_dir / rel))
                paths.append((i, rel))
                hashes.append((i, dhash(crop)))
            if n % 500 == 0:
                print(f"  crop {n:,}/{df['image'].nunique():,} 사진")
    finally:
        for zf in open_zips.values():
            zf.close()
    df["crop_path"] = pd.Series(dict(paths))
    df["dhash"] = pd.Series(dict(hashes))
    return df


def drop_cross_split_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """같은 dHash가 여러 분할에 있으면 train 쪽만 남기고 val/test 쪽을 제외한다."""
    order = {"train": 0, "val": 1, "test": 2}
    keep_split = df.groupby("dhash")["split"].agg(lambda s: min(s, key=order.get))
    dup = df["split"] != df["dhash"].map(keep_split)
    return df[~dup].reset_index(drop=True), int(dup.sum())


def build(per_code: int, seed: int, version: str, group_by: str, envs: list[str] | None) -> Path:
    out = processed_dir(version)
    out.mkdir(parents=True, exist_ok=True)
    objects_csv = data_root() / "processed" / "objects.csv"
    if not objects_csv.exists():
        raise SystemExit("objects.csv가 없습니다. 먼저 `python -m src.dataset index`를 실행하세요.")
    objects = pd.read_csv(objects_csv)
    # 원천 이미지 zip이 있는 수집 환경·세트만 사용
    have = {Path(n).name for n in image_index(raw_dir())}
    objects = objects[objects["image"].isin(have)]
    if objects.empty:
        raise SystemExit("원천 이미지 zip(TS_/VS_)과 맞는 라벨이 없습니다.")
    df = filter_and_sample(objects, per_code, seed, envs=envs)
    df["group"] = df["image"].map(lambda n: group_of(n, group_by))
    df["split"] = group_split(df, seed)
    df = make_crops(df, out)
    df, n_dup = drop_cross_split_duplicates(df)
    df.to_csv(out / "manifest.csv", index=False, encoding="utf-8")
    meta = {
        "version": version, "label_config": load_label_config()["version"], "seed": seed,
        "per_code": per_code, "min_side": MIN_SIDE, "pad_ratio": PAD_RATIO, "crop_size": CROP_SIZE,
        "group_by": group_by, "envs": envs, "split_ratios_target": SPLIT_RATIOS,
        "n_objects": len(df), "n_groups": int(df["group"].nunique()),
        "split_counts": df["split"].value_counts().to_dict(),
        "split_verdict_counts": df.groupby(["split", "verdict"]).size().unstack(fill_value=0).to_dict("index"),
        "dropped_cross_split_duplicates": n_dup,
        "groups_overlap_between_splits": int((df.groupby("group")["split"].nunique() > 1).sum()),
    }
    with open(out / "split_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("index")
    b = sub.add_parser("build")
    b.add_argument("--per-code", type=int, default=1000, help="세부 라벨별 최대 표본 수 (기획서: 500~1,000)")
    b.add_argument("--seed", type=int, default=42)
    b.add_argument("--version", default=DEFAULT_VERSION)
    b.add_argument("--group-by", choices=["day", "image"], default="day")
    b.add_argument("--env", action="append", help="수집 환경 제한, 예: 실내형분류기 (여러 번 가능)")
    args = ap.parse_args(argv)
    if args.cmd == "index":
        df = build_index()
        out = data_root() / "processed" / "objects.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False, encoding="utf-8")
        print(f"saved {out}  ({len(df):,} objects, {df['image'].nunique():,} images)")
        print(df.groupby(["env", "verdict"]).size().unstack(fill_value=0))
    else:
        build(args.per_code, args.seed, args.version, args.group_by, args.env)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
