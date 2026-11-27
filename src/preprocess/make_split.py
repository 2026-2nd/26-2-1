"""물체 목록(objects_index)에서 학습·검증·테스트와 폰 평가 세트를 뽑는다.

    from src.preprocess.make_split import make_split
    manifest, short, info = make_split(df, app_images, seed=42)

나누는 기준 (근거: claude/결정_기록.md, 2026-11-26)
- 같은 묶음(group)의 물체는 항상 같은 세트에 들어간다. 같은 사진에서 나온 물체도 당연히 같이 간다.
    B(실내형분류기): 분류기 × 날짜 × 시간(시)  — 같은 시간대에 연달아 찍은 비슷한 사진을 묶는다
    C(앱 휴대폰)  : 폰 기종 × 날짜            — 같은 폰·같은 날 사진이 여러 용도에 섞이지 않게 한다
    A(재활용선별장): 선별장 번호(L·C·R 카메라 합침) × 날짜 × 시간
- 묶음 단위로 나누되 17개 클래스 비율이 세트마다 비슷하도록 StratifiedGroupKFold를 쓴다.
  큰 묶음 하나에 몰린 클래스가 있으면 한 번 나눈 결과가 한쪽으로 쏠릴 수 있어서,
  seed를 바꿔 가며 여러 번(TRIES) 나눠 보고 "클래스당 목표 개수에 가장 덜 모자란"(모자란 비율의 합이 가장 작은) 결과를 고른다.
  (고르는 기준은 개수뿐이고 모델 성능은 보지 않으므로 누수가 아니다. 고른 seed는 기록한다.)
- 나눈 다음, 세트마다 클래스당 최대 개수(CAPS)만큼만 무작위로 뽑는다.
- 같은 입력·같은 seed면 항상 같은 결과가 나온다 (object_id 순으로 정렬한 뒤 처리).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

# 세트별 클래스당 최대 개수
B_CAPS = {"train": 1050, "val": 225, "test": 225}                         # 1,500 = 70:15:15
C_CAPS = {"phone_tune": 300, "phone_final": 500, "phone_train": 300}
A_CAPS = {"extra_a": 1050}                                                # 선별장 추가 실험용(학습에만 더함)
TRIES = 30                                                                # 나누기를 몇 번 시도해 볼지

KEEP = ["object_id", "source", "split", "class17", "class_id", "verdict", "reason", "item", "group",
        "image", "label_zip", "method", "day", "device", "x", "y", "w", "h", "img_w", "img_h"]


def add_groups(df: pd.DataFrame) -> pd.DataFrame:
    """출처별 묶음(group) 열을 붙인다."""
    df = df.copy()
    day = df["day"].astype(str)
    hour = df["date"].astype(str).str[11:13]
    method = df["method"].astype(str)
    group = pd.Series("", index=df.index, dtype=object)
    for src, key in {
        "B": "B:" + method + "_" + day + "_" + hour,
        "C": "C:" + df["device"].astype(str) + "_" + day,
        "A": "A:" + method.str.replace(r"[LCR]$", "", regex=True) + "_" + day + "_" + hour,
    }.items():
        mask = df["source"] == src
        group[mask] = key[mask]
    df["group"] = group
    return df


def _folds(sub: pd.DataFrame, n_splits: int, seed: int) -> np.ndarray:
    """묶음 단위로 n_splits개 조각에 나눈다(클래스 비율을 맞춤). 조각 번호를 돌려준다."""
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold = np.empty(len(sub), dtype=int)
    for k, (_, idx) in enumerate(sgkf.split(sub, sub["class17"], sub["group"])):
        fold[idx] = k
    return fold


def _shortage(split: np.ndarray, cls: pd.Series, caps: dict) -> float:
    """세트·클래스마다 목표 개수(caps)보다 모자란 비율의 합.
    작은 세트(test 225개)에서 100개 모자란 것이 큰 세트(train 1,050개)에서 100개 모자란 것보다 더 나쁘게 계산된다."""
    counts = pd.crosstab(cls, split)
    total = 0.0
    for name, cap in caps.items():
        col = counts[name] if name in counts.columns else pd.Series(0, index=counts.index)
        total += float(((cap - col).clip(lower=0) / cap).sum())
    return round(total, 3)


def _best_split(sub: pd.DataFrame, n_splits: int, to_split, caps: dict, seed: int, tries: int):
    """seed를 바꿔 tries번 나눠 보고 가장 덜 모자란 결과를 고른다. (split 배열, 고른 seed, 모자란 합)"""
    if len(sub) == 0:
        return np.array([], dtype=object), seed, 0
    best = None
    for t in range(tries):
        split = to_split(_folds(sub, n_splits, seed + t))
        short = _shortage(split, sub["class17"], caps)
        if best is None or short < best[2]:
            best = (split, seed + t, short)
        if short == 0:
            break
    return best


def _cap(sub: pd.DataFrame, caps: dict, seed: int) -> tuple[pd.DataFrame, dict]:
    """세트·클래스마다 최대 caps[세트]개만 뽑는다. 모자라면 있는 만큼 쓰고 기록한다."""
    parts, short = [], {}
    for (split, cls), g in sub.groupby(["split", "class17"], sort=True):
        n = caps[split]
        if len(g) < n:
            short[(split, cls)] = len(g)
        parts.append(g.sample(n=min(n, len(g)), random_state=seed))
    return (pd.concat(parts) if parts else sub.iloc[0:0]), short


def make_split(df: pd.DataFrame, app_images: set[str], seed: int = 42,
               b_caps: dict = B_CAPS, c_caps: dict = C_CAPS, a_caps: dict = A_CAPS, tries: int = TRIES):
    """manifest(뽑힌 물체 목록, split 열 포함), 모자란 칸 기록(short), 고른 seed 정보(info)를 돌려준다."""
    df = add_groups(df).sort_values("object_id", kind="mergesort").reset_index(drop=True)
    out, short, info = [], {}, {}

    # B: 20조각 → 3조각 test(15%), 3조각 val(15%), 14조각 train(70%)
    b = df[df["source"] == "B"].reset_index(drop=True)
    b_split = lambda f: np.where(f < 3, "test", np.where(f < 6, "val", "train"))
    b["split"], info["B_seed"], info["B_모자란합"] = _best_split(b, 20, b_split, b_caps, seed, tries)
    picked, s = _cap(b, b_caps, seed); out.append(picked); short.update(s)

    # C: 이미지가 있는 것만. 4조각 → 1 조정용(25%), 2 최종 평가용(50%), 1 학습 추가용(25%)
    c = df[(df["source"] == "C") & df["image"].isin(app_images)].reset_index(drop=True)
    c_split = lambda f: np.select([f == 0, f <= 2], ["phone_tune", "phone_final"], "phone_train")
    c["split"], info["C_seed"], info["C_모자란합"] = _best_split(c, 4, c_split, c_caps, seed, tries)
    picked, s = _cap(c, c_caps, seed); out.append(picked); short.update(s)

    # A: 선별장은 추가 실험의 '학습에 더할 데이터'로만 쓴다
    a = df[df["source"] == "A"].reset_index(drop=True)
    a["split"] = "extra_a"
    picked, s = _cap(a, a_caps, seed); out.append(picked); short.update(s)

    manifest = pd.concat(out, ignore_index=True)
    manifest = manifest[[c for c in KEEP if c in manifest.columns]]
    manifest = manifest.sort_values(["split", "class_id", "object_id"], kind="mergesort").reset_index(drop=True)
    return manifest, short, info


def check_no_leak(manifest: pd.DataFrame) -> dict:
    """같은 묶음·같은 사진이 두 세트 이상에 들어갔는지 센다 (0이어야 정상)."""
    return {
        "여러 세트에 걸친 묶음": int((manifest.groupby("group")["split"].nunique() > 1).sum()),
        "여러 세트에 걸친 사진": int((manifest.groupby(["source", "image"])["split"].nunique() > 1).sum()),
    }
