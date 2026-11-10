from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

from . import df40_eda as E

QUALITY_BINS = ("<=49", "50-74", "75-89", "90-94", "95", "96-99", "100", "unknown", "not_jpeg")
JPEG_FEATURES = ("source_jpeg_quality_level", "source_jpeg_quality_bin", "source_jpeg_subsampling")


def _records(frame, unit="images"):
    required = ("dataset", "target", "source_format", "source_jpeg_quality", *JPEG_FEATURES)
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError("Сначала извлеките JPEG headers: " + ", ".join(missing))
    if unit not in ("images", "faces"):
        raise ValueError("unit должен быть images или faces")
    data = frame.copy()
    data["class"] = data["target"].map({0: "real", 1: "fake"}).fillna("unknown")
    if unit == "faces":
        return data.loc[data["face_index"].notna()].copy() if "face_index" in data else data
    key = next((name for name in ("image_sample_id", "input_path", "image_sha256", "sample_id") if name in data), None)
    if key is None:
        raise ValueError("Нет ключа исходного изображения")
    keys = ["dataset", key]
    grouped = data.groupby(keys, dropna=False)
    inconsistent = grouped[list(JPEG_FEATURES)].nunique(dropna=False).gt(1).any(axis=1)
    if inconsistent.any():
        raise ValueError("JPEG metadata одного исходника различаются между лицами")
    classes = grouped["target"].agg(["min", "max"]).reset_index()
    classes["class"] = np.select(
        [classes["min"].eq(0) & classes["max"].eq(0),
         classes["min"].eq(1) & classes["max"].eq(1),
         classes["min"].eq(0) & classes["max"].eq(1)],
        ["real", "fake", "mixed"], default="unknown",
    )
    data = data.drop_duplicates(keys).drop(columns="class")
    return data.merge(classes[keys + ["class"]], on=keys, how="left", validate="one_to_one")


def jpeg_quality_summary(frame, unit="images"):
    data = _records(frame, unit)
    rows = []
    for (dataset, label), group in data.groupby(["dataset", "class"], dropna=False):
        jpeg = group["source_format"].fillna("").astype(str).str.upper().isin(["JPEG", "JPG", "MPO"])
        quality = pd.to_numeric(group["source_jpeg_quality"], errors="coerce")
        known = jpeg & quality.between(1, 100)
        rows.append({
            "dataset": dataset, "class": label, "unit": unit, "records": len(group),
            "jpeg_records": int(jpeg.sum()), "not_jpeg_records": int(group["source_jpeg_quality_bin"].eq("not_jpeg").sum()),
            "unknown_source_format": int(group["source_format"].fillna("unknown").astype(str).str.lower().isin(["", "unknown"]).sum()),
            "known_quality": int(known.sum()), "unknown_quality": int((jpeg & ~known).sum()),
            "known_share_of_jpeg": float(known.sum() / jpeg.sum()) if jpeg.any() else np.nan,
            "quality_95": int((jpeg & quality.eq(95)).sum()),
            "quality_100": int((jpeg & quality.eq(100)).sum()),
            "distinct_known_levels": int(quality.loc[known].nunique()),
        })
    return pd.DataFrame(rows)


def jpeg_quality_values(frame, unit="images"):
    data = _records(frame, unit)
    columns = ["dataset", "class", "source_jpeg_quality_level", "source_jpeg_quality_bin"]
    if "source_jpeg_quality_method" in data:
        columns.append("source_jpeg_quality_method")
    if "source_jpeg_quality_status" in data:
        columns.append("source_jpeg_quality_status")
    return data.groupby(columns, dropna=False).size().rename("records").reset_index()


def jpeg_header_audit(frame, unit="images"):
    data = _records(frame, unit)
    result = {}
    for column in ("source_jpeg_quality_method", "source_jpeg_quality_status", "source_jpeg_subsampling",
                   "source_jpeg_progressive"):
        if column in data:
            result[column] = data.groupby(["dataset", column], dropna=False).size().rename("records").reset_index()
    if "source_jpeg_qtable_hash" in data:
        present = data["source_jpeg_qtable_hash"].notna() & data["source_jpeg_qtable_hash"].astype(str).ne("")
        result["distinct_qtables"] = (
            data.loc[present].groupby("dataset")["source_jpeg_qtable_hash"]
            .nunique().rename("distinct_qtable_hashes").to_frame()
        )
    return result


def _quality_axis(axis, values):
    quality = pd.to_numeric(pd.Series(values), errors="coerce")
    quality = quality.loc[quality.between(1, 100)]
    minimum = float(quality.min()) if len(quality) else 1.0
    lower = minimum - 0.5
    step = 5 if lower >= 50 else 10
    start = int(np.ceil(lower / step) * step)
    ticks = sorted({*range(start, 101, step), int(minimum), 95, 100})
    axis.set_xticks(ticks)
    axis.set_xlim(lower, 100.8)


def plot_source_jpeg_quality(frame, unit="images"):
    data = _records(frame, unit)
    fig, axes = plt.subplots(1, 2, figsize=(17, 5.5), layout="constrained")
    table = pd.crosstab(data["dataset"], data["source_jpeg_quality_bin"], normalize="index")
    table = table.reindex(columns=QUALITY_BINS, fill_value=0)
    table.T.plot.bar(ax=axes[0], colormap="Set2", width=0.8, rot=40)
    axes[0].set_title("Quality-бин, unknown и исходники других форматов")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("доля исходников" if unit == "images" else "доля лиц")
    axes[0].legend(title="", frameon=False)
    colors = {**E.COLORS, "mixed": "#9A60AD", "unknown": "#A3A9B5"}
    for label in ("real", "fake", "mixed", "unknown"):
        values = pd.to_numeric(data.loc[data["class"].eq(label), "source_jpeg_quality"], errors="coerce")
        values = values.loc[values.between(1, 100)]
        if len(values):
            counts = values.value_counts(normalize=True).sort_index()
            axes[1].step(counts.index, counts.values, where="mid", color=colors[label], lw=2, label=label)
            axes[1].scatter(counts.index, counts.values, color=colors[label], s=20)
    _quality_axis(axes[1], data["source_jpeg_quality"])
    axes[1].set_title("Точные совпавшие IJG quality-levels; 95 и 100 отдельно")
    axes[1].set_xlabel("оценка quality по исходным quantization tables")
    axes[1].set_ylabel("доля среди измеренных quality внутри класса")
    if axes[1].get_legend_handles_labels()[0]:
        axes[1].legend(frameon=False)
    fig.suptitle("JPEG-сжатие исходников сохраняется как диагностический признак", fontsize=16)
    return fig


def plot_split_jpeg_quality(parts):
    fig, axes = plt.subplots(1, 2, figsize=(17, 5.5), layout="constrained")
    bins = {}
    known_quality = []
    for partition in ("train", "validate", "test"):
        data = _records(parts[partition], "faces")
        bins[partition] = data["source_jpeg_quality_bin"].value_counts(normalize=True)
        quality = pd.to_numeric(data["source_jpeg_quality"], errors="coerce")
        quality = quality.loc[quality.between(1, 100)]
        known_quality.append(quality)
        if len(quality):
            counts = quality.value_counts(normalize=True).sort_index()
            axes[1].step(counts.index, counts.values, where="mid", lw=2,
                         label=partition, color=E.COLORS[partition])
            axes[1].scatter(counts.index, counts.values, s=20, color=E.COLORS[partition])
    table = pd.DataFrame(bins).reindex(QUALITY_BINS).fillna(0)
    table.plot.bar(ax=axes[0], color=[E.COLORS[name] for name in table], rot=40)
    axes[0].set_xlabel("")
    axes[0].set_ylabel("доля лиц внутри части")
    axes[0].set_title("Глобальные quality-бины всех исходников")
    axes[0].legend(title="", frameon=False)
    _quality_axis(axes[1], pd.concat(known_quality, ignore_index=True))
    axes[1].set_xlabel("точный совпавший quality-level")
    axes[1].set_ylabel("доля среди измеренных JPEG quality")
    axes[1].set_title("Распределение exact quality, включая 95 и 100")
    if axes[1].get_legend_handles_labels()[0]:
        axes[1].legend(frameon=False)
    fig.suptitle("Source JPEG quality в train / validate / test", fontsize=16)
    return fig


def plot_conditional_split_jpeg_quality(parts):
    combined = pd.concat(
        [parts[name].assign(checked_split=name) for name in ("train", "validate", "test")],
        ignore_index=True,
    )
    _records(combined, "faces")
    datasets = [name for name in E.DATASETS if combined["dataset"].astype(str).str.lower().eq(name).any()]
    fig, axes = plt.subplots(2, len(datasets), figsize=(4.6 * len(datasets), 9),
                             squeeze=False, layout="constrained")
    jpeg = combined["source_format"].fillna("").astype(str).str.upper().isin(["JPEG", "JPG", "MPO"])
    for row, target in enumerate((0, 1)):
        for column, dataset in enumerate(datasets):
            ax = axes[row, column]
            mask = jpeg & combined["target"].eq(target)
            mask &= combined["dataset"].astype(str).str.lower().eq(dataset)
            data = combined.loc[mask]
            if data.empty:
                ax.text(0.5, 0.5, "Нет JPEG-лиц", ha="center", va="center", transform=ax.transAxes)
                ax.set_axis_off()
                continue
            counts = {
                name: group["source_jpeg_quality_bin"].value_counts(normalize=True)
                for name, group in data.groupby("checked_split")
            }
            table = pd.DataFrame(counts).reindex(index=QUALITY_BINS[:-1],
                                                 columns=["train", "validate", "test"]).fillna(0)
            table.plot.bar(ax=ax, color=[E.COLORS[name] for name in table], rot=45, width=0.85)
            ax.set_title(dataset + " | " + ("real" if target == 0 else "fake"))
            ax.set_xlabel("")
            ax.set_ylabel("доля среди JPEG-лиц")
            ax.tick_params(axis="x", labelsize=8)
            ax.legend(title="", fontsize=8, frameon=False)
    fig.suptitle("JPEG quality внутри dataset + target; 95 и 100 остаются разными бинами", fontsize=16)
    return fig


def migrate_pixel_metadata_binding(root, proof_path):
    return E.migrate_pixel_metadata_binding(root, proof_path)
