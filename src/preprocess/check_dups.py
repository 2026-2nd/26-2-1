"""잘라 낸 사진의 지문(dHash)으로 서로 다른 세트 사이의 '거의 같은 사진'을 찾는다.

    from src.preprocess.check_dups import cross_split_dups, add_pixel_diff
    pairs = cross_split_dups(crops, [("train", "val"), ("train", "test")], max_dist=4)
    pairs = add_pixel_diff(pairs, crops_dir)      # 후보를 실제 색으로 다시 비교

- dHash는 사진을 9×8 흑백으로 줄여 옆 칸보다 밝은지 어두운지를 64비트로 적은 것이다.
  두 지문에서 다른 비트 수(해밍 거리)가 작을수록 비슷한 사진이다. 0이면 사실상 같은 사진.
- 같은 물체라도 놓인 방향이 다르면 지문이 크게 달라지므로, 이 검사는 '거의 똑같은 사진'만 잡는다.
- 실내형분류기 사진은 대부분 '어두운 배경 + 가운데 밝은 물건'이라 다른 물건도 지문이 같게 나온다.
  그래서 지문은 후보를 넓게 고르는 데만 쓰고, add_pixel_diff로 실제 색을 비교해 진짜 중복을 가린다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def _to_u64(hexes: pd.Series) -> np.ndarray:
    return np.array([int(h, 16) for h in hexes], dtype=np.uint64)


def _hamming(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a(n,) × b(m,) 의 해밍 거리 행렬 (n, m)."""
    x = (a[:, None] ^ b[None, :]).view(np.uint8).reshape(len(a), len(b), 8)
    return _POP[x].sum(axis=2, dtype=np.uint8)


def cross_split_dups(crops: pd.DataFrame, pairs: list[tuple[str, str]], max_dist: int = 4,
                     chunk: int = 2000) -> pd.DataFrame:
    """세트 쌍마다 해밍 거리가 max_dist 이하인 사진 쌍을 모두 찾는다."""
    ok = crops[crops["dhash"].fillna("").str.len() == 16]
    found = []
    for s1, s2 in pairs:
        a, b = ok[ok["split"] == s1], ok[ok["split"] == s2]
        if a.empty or b.empty:
            continue
        ha, hb = _to_u64(a["dhash"]), _to_u64(b["dhash"])
        for i in range(0, len(ha), chunk):
            d = _hamming(ha[i:i + chunk], hb)
            ii, jj = np.nonzero(d <= max_dist)
            for x, y in zip(ii, jj):
                ra, rb = a.iloc[i + x], b.iloc[y]
                found.append({"split_a": s1, "object_a": ra["object_id"], "path_a": ra["path"], "class_a": ra["class17"],
                              "split_b": s2, "object_b": rb["object_id"], "path_b": rb["path"], "class_b": rb["class17"],
                              "dist": int(d[x, y])})
    return pd.DataFrame(found, columns=["split_a", "object_a", "path_a", "class_a",
                                        "split_b", "object_b", "path_b", "class_b", "dist"])


def add_pixel_diff(dups: pd.DataFrame, crops_dir, size: int = 32) -> pd.DataFrame:
    """후보 쌍마다 실제 색을 비교한 차이(pix_diff)를 붙인다.

    두 사진을 size×size 컬러로 줄인 뒤 같은 위치 픽셀의 RGB 값 차이를 평균낸다(0~255).
    거의 같은 사진이면 몇 이하로 작고, 우연히 지문만 비슷한 다른 물건이면 훨씬 크다.
    dHash(흑백 64비트)만으로는 '어두운 배경 + 가운데 밝은 물건' 사진이 모두 비슷하게 나와서 추가했다.
    """
    import io
    import zipfile
    from pathlib import Path
    from PIL import Image

    if dups.empty:
        return dups.assign(pix_diff=pd.Series(dtype=float))
    need = set(zip(dups["split_a"], dups["path_a"])) | set(zip(dups["split_b"], dups["path_b"]))
    thumbs = {}
    for split in sorted({s for s, _ in need}):
        with zipfile.ZipFile(Path(crops_dir) / f"{split}.zip") as z:
            for s, p in sorted(k for k in need if k[0] == split):
                im = Image.open(io.BytesIO(z.read(p))).convert("RGB").resize((size, size), Image.BILINEAR)
                thumbs[(s, p)] = np.asarray(im, dtype=np.int16)
    diff = [float(np.abs(thumbs[(a, pa)] - thumbs[(b, pb)]).mean())
            for a, pa, b, pb in zip(dups["split_a"], dups["path_a"], dups["split_b"], dups["path_b"])]
    return dups.assign(pix_diff=np.round(diff, 1))
