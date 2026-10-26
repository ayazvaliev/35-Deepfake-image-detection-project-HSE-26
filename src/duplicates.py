import hashlib
from io import BytesIO

import numpy as np
import pandas as pd
from PIL import Image
from scipy.fft import dctn
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from src.config import PROCESSED_DIR, RAW_DIR

THUMB = 64
MAX_HAMMING = 8
# chosen by eye: above ~15 different people with similar pose and background appear
MAE_THRESHOLD = 15
SPLIT_ORDER = ["train", "valid", "test"]
OF_TEST = RAW_DIR / "openforensics" / "Dataset" / "Test"


def file_list():
    index = pd.read_csv(PROCESSED_DIR / "index.csv")
    big = index[index["dataset"] == "140k"]
    of_rows = [
        {"path": p.relative_to(RAW_DIR).as_posix(), "dataset": "openforensics", "split": "test", "label": label}
        for label, cls in enumerate(["Real", "Fake"])
        for p in sorted((OF_TEST / cls).glob("*.jpg"))
    ]
    return pd.concat([big, pd.DataFrame(of_rows)], ignore_index=True)


def phash(gray):
    small = np.asarray(gray.resize((32, 32), Image.LANCZOS), dtype=np.float64)
    low = dctn(small, norm="ortho")[:8, :8]
    return np.packbits(low > np.median(low)).tobytes().hex()


def build_hashes():
    files = file_list()
    cache = PROCESSED_DIR / "phash.csv"
    thumbs_path = PROCESSED_DIR / "thumbs64.npy"
    if cache.exists() and thumbs_path.exists():
        hashes = pd.read_csv(cache, dtype={"phash": str})
        thumbs = np.load(thumbs_path, mmap_mode="r")
        if hashes["path"].tolist() == files["path"].tolist() and len(thumbs) == len(hashes):
            return hashes, thumbs

    md5s, phashes = [], []
    thumbs = np.empty((len(files), THUMB, THUMB), dtype=np.uint8)
    for i, path in enumerate(files["path"]):
        data = (RAW_DIR / path).read_bytes()
        md5s.append(hashlib.md5(data).hexdigest())
        with Image.open(BytesIO(data)) as im:
            gray = im.convert("L")
        phashes.append(phash(gray))
        thumbs[i] = np.asarray(gray.resize((THUMB, THUMB), Image.LANCZOS))

    hashes = files.assign(md5=md5s, phash=phashes)
    hashes.to_csv(cache, index=False)
    np.save(thumbs_path, thumbs)
    return hashes, thumbs


def hamming_scan(hashes, max_dist, block=1000):
    codes = np.array([int(h, 16) for h in hashes["phash"]], dtype=np.uint64)
    nearest = np.empty(len(codes), dtype=np.uint8)
    found = []
    for start in range(0, len(codes), block):
        dist = np.bitwise_count(codes[start:start + block, None] ^ codes[None, :])
        rows = np.arange(len(dist))
        dist[rows, start + rows] = 255
        nearest[start:start + block] = dist.min(axis=1)
        i, j = np.nonzero(dist <= max_dist)
        keep = j > start + i
        found.append(pd.DataFrame({"i": start + i[keep], "j": j[keep], "hamming": dist[i[keep], j[keep]]}))
    return nearest, pd.concat(found, ignore_index=True)


def pair_mae(pairs, thumbs, chunk=10_000):
    mae = np.empty(len(pairs))
    for start in range(0, len(pairs), chunk):
        part = pairs.iloc[start:start + chunk]
        a = thumbs[part["i"].to_numpy()].astype(np.int16)
        b = thumbs[part["j"].to_numpy()].astype(np.int16)
        mae[start:start + chunk] = np.abs(a - b).mean(axis=(1, 2))
    return mae


def describe_pairs(pairs, hashes):
    a = hashes.iloc[pairs["i"]].reset_index(drop=True).add_suffix("_a")
    b = hashes.iloc[pairs["j"]].reset_index(drop=True).add_suffix("_b")
    out = pd.concat([pairs.reset_index(drop=True), a, b], axis=1)
    out["same_md5"] = out["md5_a"] == out["md5_b"]
    return out


def pair_category(pairs):
    cross = pairs["dataset_a"] != pairs["dataset_b"]
    same_split = pairs["split_a"] == pairs["split_b"]
    first = np.where(pairs["split_a"].map(SPLIT_ORDER.index) <= pairs["split_b"].map(SPLIT_ORDER.index), pairs["split_a"], pairs["split_b"])
    second = np.where(first == pairs["split_a"], pairs["split_b"], pairs["split_a"])
    between = pairs["dataset_a"] + " " + first + "–" + second
    within = pairs["dataset_a"] + " внутри " + pairs["split_a"]
    return pd.Series(np.where(cross, "140k–openforensics test", np.where(same_split, within, between)), index=pairs.index)


def dup_groups(hashes, confirmed):
    n = len(hashes)
    graph = coo_matrix((np.ones(len(confirmed)), (confirmed["i"], confirmed["j"])), shape=(n, n))
    _, group_id = connected_components(graph, directed=False)
    groups = pd.DataFrame({"path": hashes["path"], "group_id": group_id})
    groups["group_size"] = groups.groupby("group_id")["path"].transform("size")
    groups.to_csv(PROCESSED_DIR / "dup_groups.csv", index=False)
    return groups


# for each image: splits of the same dataset where it has a near-copy
def partner_splits(hashes, confirmed):
    both = pd.concat([
        confirmed[["i", "split_b", "dataset_b"]].set_axis(["idx", "split", "dataset"], axis=1),
        confirmed[["j", "split_a", "dataset_a"]].set_axis(["idx", "split", "dataset"], axis=1),
    ])
    both = both[both["dataset"] == hashes["dataset"].to_numpy()[both["idx"]]]
    return both.groupby("idx")["split"].agg(set).reindex(range(len(hashes))).apply(lambda x: x if isinstance(x, set) else set())


def leak_table(hashes, confirmed):
    splits = partner_splits(hashes, confirmed)
    big = hashes["dataset"] == "140k"
    rows = []
    for target, source in [("valid", "train"), ("test", "train"), ("test", "valid")]:
        for label, cls in [(0, "real"), (1, "fake")]:
            mask = big & (hashes["split"] == target) & (hashes["label"] == label)
            leaked = splits[mask].apply(lambda s: source in s).sum()
            rows.append({"target": target, "source": source, "class": cls, "leaked": int(leaked),
                         "of_split": int(mask.sum()), "percent": round(100 * leaked / mask.sum(), 2)})
    return pd.DataFrame(rows)


def clean_test(hashes, confirmed):
    splits = partner_splits(hashes, confirmed)
    return (hashes["split"] == "test") & ~splits.apply(lambda s: bool(s & {"train", "valid"}))
