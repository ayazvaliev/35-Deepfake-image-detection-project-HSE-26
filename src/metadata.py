import hashlib

import numpy as np
import pandas as pd
from PIL import Image, JpegImagePlugin
from sklearn.metrics import roc_auc_score

from src.config import PROCESSED_DIR, RAW_DIR

# IJG standard luminance table (JPEG Annex K), natural order
STD_LUMA = np.array([
    16, 11, 10, 16, 24, 40, 51, 61,
    12, 12, 14, 19, 26, 58, 60, 55,
    14, 13, 16, 24, 40, 57, 69, 56,
    14, 17, 22, 29, 51, 87, 80, 62,
    18, 22, 37, 56, 68, 109, 103, 77,
    24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101,
    72, 92, 95, 98, 112, 100, 103, 99,
])
MAX_QUALITY_ERROR = 1.0
# |AUC - 0.5| or TVD above this counts as a real/fake difference
DIFF_THRESHOLD = 0.05
SUBSAMPLING = {0: "4:4:4", 1: "4:2:2", 2: "4:2:0"}


def ijg_table(quality):
    scale = 5000 / quality if quality < 50 else 200 - 2 * quality
    return np.clip((STD_LUMA * scale + 50) // 100, 1, 255)


def jpeg_quality(quant):
    luma = np.array(quant[0])
    errors = [np.abs(luma - ijg_table(q)).mean() for q in range(1, 101)]
    best = int(np.argmin(errors))
    error = float(errors[best])
    quality = best + 1 if error <= MAX_QUALITY_ERROR else None
    return quality, error


def file_metadata(path):
    try:
        with Image.open(path) as im:
            meta = {
                "format": im.format,
                "width": im.width,
                "height": im.height,
                "mode": im.mode,
                "progressive": bool(im.info.get("progressive", False)),
                "has_exif": "exif" in im.info,
                "has_icc": "icc_profile" in im.info,
            }
            quant = getattr(im, "quantization", None)
            if quant:
                meta["quality"], meta["quality_error"] = jpeg_quality(quant)
                meta["qtable_hash"] = hashlib.md5(str(sorted(quant.items())).encode()).hexdigest()[:8]
                meta["subsampling"] = SUBSAMPLING.get(JpegImagePlugin.get_sampling(im))
            im.load()
        meta["opens"] = True
    except Exception:
        meta = {"opens": False}
    meta["file_size"] = path.stat().st_size
    return meta


def build_metadata():
    index = pd.read_csv(PROCESSED_DIR / "index.csv")
    cache = PROCESSED_DIR / "metadata.csv"
    if cache.exists():
        metadata = pd.read_csv(cache)
        if metadata["path"].tolist() == index["path"].tolist():
            return metadata

    rows = [file_metadata(RAW_DIR / p) for p in index["path"]]
    metadata = pd.concat([index, pd.DataFrame(rows)], axis=1)
    metadata.to_csv(cache, index=False)
    return metadata


def value_shares(values):
    shares = values.astype(str).value_counts(normalize=True).round(3).head(5)
    return ", ".join(f"{k}: {v}" for k, v in shares.items())


def compare_classes(df, numeric, categorical):
    rows = []
    for field in numeric + categorical:
        real = df.loc[df["label"] == 0, field]
        fake = df.loc[df["label"] == 1, field]
        row = {"field": field, "n_unique": df[field].nunique(dropna=False)}
        if field in numeric:
            known = df[field].notna()
            if df.loc[known, field].nunique() > 1:
                row["auc"] = round(roc_auc_score(df.loc[known, "label"], df.loc[known, field]), 3)
            row["real"] = f"median {real.median()}, [{real.min()}, {real.max()}]"
            row["fake"] = f"median {fake.median()}, [{fake.min()}, {fake.max()}]"
        else:
            real_shares = real.astype(str).value_counts(normalize=True)
            fake_shares = fake.astype(str).value_counts(normalize=True)
            # total variation distance between value distributions of the two classes
            row["tvd"] = round(real_shares.sub(fake_shares, fill_value=0).abs().sum() / 2, 3)
            row["real"] = value_shares(real)
            row["fake"] = value_shares(fake)
        score = abs(row.get("auc", 0.5) - 0.5) if field in numeric else row["tvd"]
        row["differs"] = score >= DIFF_THRESHOLD
        rows.append(row)
    return pd.DataFrame(rows)
