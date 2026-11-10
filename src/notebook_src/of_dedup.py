import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree


def iou(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    width = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    height = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    intersection = width * height
    return intersection / max(a[2] * a[3] + b[2] * b[3] - intersection, 1e-12)


def close_pairs(hashes, max_dist=10):
    hashes = np.asarray(hashes, dtype=np.uint64)
    bits = np.unpackbits(hashes.view(np.uint8).reshape(-1, 8), axis=1)
    tree = BallTree(bits, metric="hamming")
    rows = []
    for start in range(0, len(bits), 512):
        neighbors, distances = tree.query_radius(bits[start:start + 512],
                                                 r=(max_dist + 0.01) / 64,
                                                 return_distance=True)
        for offset, (indices, values) in enumerate(zip(neighbors, distances)):
            left = start + offset
            rows.extend((left, int(right), int(round(distance * 64)))
                        for right, distance in zip(indices, values) if right > left)
    return pd.DataFrame(rows, columns=["i", "j", "hamming"])


def pixel_check(pairs, thumbnails, boxes, sizes):
    pairs = pairs.copy()
    mae_bg, mae_all = [], []
    for left, right in pairs[["i", "j"]].itertuples(index=False, name=None):
        first, second = thumbnails[left], thumbnails[right]
        difference = np.abs(first.astype(float) - second.astype(float))
        mask = np.ones(difference.shape[:2], bool)
        for index in [left, right]:
            width, height = sizes[index]
            sy, sx = mask.shape[0] / height, mask.shape[1] / width
            for x, y, w, h in boxes[index]:
                x0, y0 = max(0, int(np.floor(x * sx))), max(0, int(np.floor(y * sy)))
                x1, y1 = min(mask.shape[1], int(np.ceil((x + w) * sx))), min(mask.shape[0], int(np.ceil((y + h) * sy)))
                mask[y0:y1, x0:x1] = False
        mae_bg.append(float(difference[mask].mean()) if mask.any() else np.nan)
        mae_all.append(float(difference.mean()))
    pairs["mae_bg"], pairs["mae_all"] = mae_bg, mae_all
    return pairs


def groups_from_pairs(n, pairs):
    parent = np.arange(n)
    def find(value):
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value
    for left, right in pairs[["i", "j"]].itertuples(index=False, name=None):
        a, b = find(left), find(right)
        parent[max(a, b)] = min(a, b)
    return np.array([find(i) for i in range(n)])

def bitwise_count(values):
    values = np.asarray(values, dtype=np.uint64)
    values = values - ((values >> np.uint64(1)) & np.uint64(0x5555555555555555))
    values = (values & np.uint64(0x3333333333333333)) + ((values >> np.uint64(2)) & np.uint64(0x3333333333333333))
    values = (values + (values >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    with np.errstate(over="ignore"):
        return ((values * np.uint64(0x0101010101010101)) >> np.uint64(56)).astype(np.uint8)
