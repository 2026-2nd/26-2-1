"""단일 물건 crop 이미지에서 수작업 수치 피처를 뽑는다 (전통적 기계학습용).

피처 묶음 (README 실험 1 "색상 vs 색상+질감·형태"를 위해 나눔):
    color          RGB·HSV 채널 평균/표준편차, HSV 히스토그램, colorfulness        (33개)
    texture_shape  밝기·대비·선명도·에지 밀도·종횡비, LBP 히스토그램, HOG           (5 + 10 + 324개)

모든 피처는 이미지 한 장에서만 계산하므로 train 정보를 쓰지 않는다 (누수 없음).
스케일링은 학습 단계의 Pipeline에서 train에만 fit 한다.
"""
from __future__ import annotations

import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern

SIZE = 128          # 색상·질감 계산용 크기
HOG_SIZE = 64       # HOG 계산용 크기
H_BINS, S_BINS, V_BINS = 12, 4, 4
LBP_P, LBP_R = 8, 1  # uniform LBP → P+2 = 10 bins


def _color_names() -> list[str]:
    names = [f"{c}_{s}" for c in ("r", "g", "b") for s in ("mean", "std")]
    names += [f"{c}_{s}" for c in ("h", "s", "v") for s in ("mean", "std")]
    names += [f"hist_h{i}" for i in range(H_BINS)]
    names += [f"hist_s{i}" for i in range(S_BINS)]
    names += [f"hist_v{i}" for i in range(V_BINS)]
    names += ["colorfulness"]
    return names


def _texture_shape_names() -> list[str]:
    names = ["brightness", "contrast", "sharpness_log", "edge_density", "aspect_ratio_log"]
    names += [f"lbp_{i}" for i in range(LBP_P + 2)]
    n_hog = len(_hog(np.zeros((HOG_SIZE, HOG_SIZE), np.uint8)))
    names += [f"hog_{i}" for i in range(n_hog)]
    return names


def _hog(gray64: np.ndarray) -> np.ndarray:
    return hog(gray64, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2),
               block_norm="L2-Hys", feature_vector=True)


def color_features(bgr: np.ndarray) -> np.ndarray:
    img = cv2.resize(bgr, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    rgb = img[:, :, ::-1].reshape(-1, 3).astype(np.float32) / 255.0
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.float32)
    hsv /= np.array([180.0, 255.0, 255.0], np.float32)   # OpenCV H 범위는 0~179
    feats = []
    for arr in (rgb, hsv):
        for c in range(3):
            feats += [arr[:, c].mean(), arr[:, c].std()]
    n = len(hsv)
    feats += list(np.histogram(hsv[:, 0], bins=H_BINS, range=(0, 1))[0] / n)
    feats += list(np.histogram(hsv[:, 1], bins=S_BINS, range=(0, 1))[0] / n)
    feats += list(np.histogram(hsv[:, 2], bins=V_BINS, range=(0, 1))[0] / n)
    # Hasler & Süsstrunk (2003) colorfulness
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    rg, yb = r - g, 0.5 * (r + g) - b
    feats.append(np.hypot(rg.std(), yb.std()) + 0.3 * np.hypot(rg.mean(), yb.mean()))
    return np.asarray(feats, np.float32)


def texture_shape_features(bgr: np.ndarray, box_w: float | None = None, box_h: float | None = None) -> np.ndarray:
    h0, w0 = bgr.shape[:2]
    w = box_w or w0
    h = box_h or h0
    img = cv2.resize(bgr, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g = gray.astype(np.float32) / 255.0
    edges = cv2.Canny(gray, 100, 200)
    feats = [
        g.mean(),                                               # 밝기
        g.std(),                                                # 대비
        np.log1p(cv2.Laplacian(gray, cv2.CV_64F).var()),        # 선명도
        (edges > 0).mean(),                                     # 에지 밀도
        np.log(max(w, 1) / max(h, 1)),                          # 원본 box 종횡비
    ]
    lbp = local_binary_pattern(gray, LBP_P, LBP_R, method="uniform")
    feats += list(np.histogram(lbp, bins=LBP_P + 2, range=(0, LBP_P + 2))[0] / lbp.size)
    gray64 = cv2.resize(gray, (HOG_SIZE, HOG_SIZE), interpolation=cv2.INTER_AREA)
    return np.concatenate([np.asarray(feats, np.float32), _hog(gray64).astype(np.float32)])


COLOR_NAMES = _color_names()
TEXTURE_SHAPE_NAMES = _texture_shape_names()
FEATURE_GROUPS = {
    "color": COLOR_NAMES,
    "texture_shape": TEXTURE_SHAPE_NAMES,
}


def extract(bgr: np.ndarray, box_w: float | None = None, box_h: float | None = None) -> np.ndarray:
    """color + texture_shape 순서로 이어 붙인 1차원 피처 벡터."""
    return np.concatenate([color_features(bgr), texture_shape_features(bgr, box_w, box_h)])


def feature_names() -> list[str]:
    return COLOR_NAMES + TEXTURE_SHAPE_NAMES
