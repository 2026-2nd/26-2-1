"""split_manifest의 물체를 원본 zip에서 꺼내 박스대로 잘라 224×224 JPEG로 저장한다.

    from src.preprocess.crop import crop_all
    crops = crop_all(manifest, paths.raw_dir(), out_dir, work_dir="/content/crops_work")

- 원본 zip은 읽기만 한다(압축을 풀지 않고 필요한 사진만 꺼낸다).
- 자르는 방법: 박스의 긴 변 × 1.1(여백 10%)인 정사각형을 박스 가운데에 맞춰 자르고 224×224로 줄인다.
  정사각형으로 자르므로 물건 모양(가로세로 비율)이 찌그러지지 않는다.
  사진 밖으로 나가는 부분은 검은색으로 채워진다(PIL 기본 동작). 모든 클래스에 똑같이 적용된다.
- 휴대폰 사진은 EXIF 회전 정보가 있을 수 있다. 라벨에 적힌 해상도(img_w×img_h)와 실제 사진 크기를 비교해
  가로세로가 뒤바뀐 경우에만 EXIF대로 돌린 뒤 자른다(orient 열에 기록).
- 큰 사진(긴 변 2,500px 이상)은 JPEG를 절반 크기로 읽어 속도를 높인다. 박스 좌표도 같은 비율로 줄인다.
- 세트(split)마다 crops/<split>.zip 하나로 묶는다. 작은 파일 수만 개를 Drive에 따로 두면 매우 느리기 때문이다.
- 이미 만든 세트 zip이 out_dir에 있으면 건너뛴다(중간에 끊겨도 이어서 할 수 있음).
- 5단계(중복 검사)에 쓰도록 잘라 낸 사진마다 dHash(64비트 사진 지문)를 계산해 둔다.
"""
from __future__ import annotations

import io
import shutil
import time
import unicodedata
import zipfile
from pathlib import Path

import pandas as pd
from PIL import Image, ImageOps

SIZE = 224
PAD = 1.10          # 여백 10%
MIN_SIDE = 32       # 원본 박스의 짧은 변이 이보다 작으면 small=True로 표시(지우지는 않음)
HALF_FROM = 2500    # 긴 변이 이 이상인 사진은 절반 크기로 읽음
QUALITY = 90


def nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def find_source_zips(raw_dir) -> list[Path]:
    """원천(이미지) zip(TS_*.zip) 목록."""
    return sorted((p for p in Path(raw_dir).rglob("*.zip") if nfc(p.name).startswith("TS_")),
                  key=lambda p: nfc(p.name))


def image_locations(zips: list[Path], wanted: set[str], log=print) -> dict[str, tuple[Path, zipfile.ZipInfo]]:
    """필요한 사진 이름 → (zip 경로, zip 안의 항목). zip의 파일 목록만 읽는다."""
    loc = {}
    for z in zips:
        with zipfile.ZipFile(z) as zf:
            for info in zf.infolist():
                name = nfc(info.filename.rsplit("/", 1)[-1])
                if name in wanted and name not in loc:
                    loc[name] = (z, info)
        log(f"  목록 읽음: {nfc(z.name)} (누적 {len(loc):,}/{len(wanted):,}장)")
    return loc


def dhash(img: Image.Image, size: int = 8) -> str:
    """64비트 dHash를 16진수 16자리로."""
    g = img.convert("L").resize((size + 1, size), Image.BILINEAR)
    px = list(g.getdata())
    bits = 0
    for r in range(size):
        row = px[r * (size + 1):(r + 1) * (size + 1)]
        for c in range(size):
            bits = (bits << 1) | (row[c] > row[c + 1])
    return f"{bits:016x}"


def _open(data: bytes, img_w, img_h):
    """사진을 열고 (이미지, 박스 좌표 배율, 회전 처리 여부)를 돌려준다."""
    im = Image.open(io.BytesIO(data))
    raw_w, raw_h = im.size
    orient = "as_is"
    try:
        lw, lh = int(float(img_w)), int(float(img_h))
    except (TypeError, ValueError):
        lw = lh = None
    if lw and (lw, lh) != (raw_w, raw_h) and (lh, lw) == (raw_w, raw_h):
        im = ImageOps.exif_transpose(im)        # 라벨은 회전된 사진 기준
        orient = "exif"
    elif lw and (lw, lh) != (raw_w, raw_h):
        orient = "size_mismatch"                # 라벨 해상도와 다르면 표시만 하고 그대로 둠
    full_w = im.size[0]
    if orient == "as_is" and max(raw_w, raw_h) >= HALF_FROM:
        im.draft("RGB", (raw_w // 2, raw_h // 2))
    im = im.convert("RGB")
    return im, im.size[0] / full_w, orient


def _crop(im: Image.Image, x, y, w, h, scale: float) -> Image.Image:
    x, y, w, h = (float(v) * scale for v in (x, y, w, h))
    side = max(w, h) * PAD
    cx, cy = x + w / 2, y + h / 2
    box = (round(cx - side / 2), round(cy - side / 2), round(cx + side / 2), round(cy + side / 2))
    return im.crop(box).resize((SIZE, SIZE), Image.BICUBIC)


def crop_split(rows: pd.DataFrame, loc: dict, zip_out: Path, log=print, every: int = 2000) -> list[dict]:
    """한 세트의 물체를 잘라 zip_out에 저장하고, 물체별 기록을 돌려준다."""
    records, t0, done = [], time.time(), 0
    by_image = rows.groupby("image", sort=False)
    # zip 안에서 앞쪽에 있는 사진부터 읽는다(디스크를 덜 왔다 갔다 함)
    order = sorted(by_image.groups, key=lambda n: (str(loc[n][0]), loc[n][1].header_offset) if n in loc else ("~", 0))
    handles = {}
    with zipfile.ZipFile(zip_out, "w", compression=zipfile.ZIP_STORED) as out:
        for name in order:
            objs = by_image.get_group(name)
            if name not in loc:
                for _, r in objs.iterrows():
                    records.append({"object_id": r["object_id"], "path": "", "error": "원본 사진 없음"})
                continue
            zpath, info = loc[name]
            zf = handles.get(zpath) or handles.setdefault(zpath, zipfile.ZipFile(zpath))
            try:
                first = objs.iloc[0]
                im, scale, orient = _open(zf.read(info), first["img_w"], first["img_h"])
            except Exception as e:  # 깨진 사진
                for _, r in objs.iterrows():
                    records.append({"object_id": r["object_id"], "path": "", "error": f"열기 실패: {e}"[:120]})
                continue
            for _, r in objs.iterrows():
                arc = f"{r['split']}/{r['class_id']:02d}/{r['object_id'].replace('#', '_')}.jpg"
                try:
                    crop = _crop(im, r["x"], r["y"], r["w"], r["h"], scale)
                    buf = io.BytesIO(); crop.save(buf, "JPEG", quality=QUALITY)
                    out.writestr(arc, buf.getvalue())
                    records.append({"object_id": r["object_id"], "path": arc, "dhash": dhash(crop),
                                    "orient": orient, "small": min(float(r["w"]), float(r["h"])) < MIN_SIDE,
                                    "error": ""})
                except Exception as e:
                    records.append({"object_id": r["object_id"], "path": "", "error": f"자르기 실패: {e}"[:120]})
                done += 1
                if done % every == 0:
                    rate = done / (time.time() - t0)
                    log(f"    {done:,}/{len(rows):,}개 ({rate:.1f}개/초, 남은 시간 약 {(len(rows) - done) / rate / 60:.0f}분)")
    for h in handles.values():
        h.close()
    return records


def crop_all(manifest: pd.DataFrame, raw_dir, out_dir, work_dir="/content/crops_work",
             splits: list[str] | None = None, log=print) -> pd.DataFrame:
    """세트별로 잘라 out_dir/crops/<split>.zip에 저장하고, 전체 기록(crops_manifest.csv)을 돌려준다."""
    out_dir, work_dir = Path(out_dir) / "crops", Path(work_dir)
    out_dir.mkdir(parents=True, exist_ok=True); work_dir.mkdir(parents=True, exist_ok=True)
    splits = splits or sorted(manifest["split"].unique())
    todo = [s for s in splits if not (out_dir / f"{s}.zip").exists()]
    for s in sorted(set(splits) - set(todo)):
        log(f"[{s}] 이미 있음 → 건너뜀")

    if todo:
        wanted = set(manifest.loc[manifest["split"].isin(todo), "image"])
        log(f"원본 zip에서 사진 {len(wanted):,}장의 위치를 찾는 중...")
        loc = image_locations(find_source_zips(raw_dir), wanted, log)
        for s in todo:
            rows = manifest[manifest["split"] == s]
            log(f"[{s}] 물체 {len(rows):,}개 자르는 중...")
            tmp = work_dir / f"{s}.zip"
            recs = pd.DataFrame(crop_split(rows, loc, tmp, log))
            recs.to_csv(work_dir / f"{s}_records.csv", index=False)
            shutil.copy(tmp, out_dir / f"{s}.zip")
            shutil.copy(work_dir / f"{s}_records.csv", out_dir / f"{s}_records.csv")
            log(f"[{s}] 완료 → {out_dir / f'{s}.zip'} (실패 {int((recs['error'] != '').sum())}개)")

    recs = pd.concat([pd.read_csv(out_dir / f"{s}_records.csv", keep_default_na=False,
                                  dtype={"dhash": str, "path": str, "orient": str, "error": str})
                      for s in splits], ignore_index=True)
    crops = manifest.merge(recs, on="object_id", how="left")
    crops.to_csv(out_dir / "crops_manifest.csv.gz", index=False)
    log(f"저장: {out_dir / 'crops_manifest.csv.gz'}")
    return crops
