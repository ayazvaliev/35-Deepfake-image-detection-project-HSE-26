from __future__ import annotations

from collections import Counter
import json
import os
from pathlib import Path

import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import seaborn as sns

COLORS = {"real": "#198D91", "fake": "#EF8354", "train": "#4077C8", "validate": "#9A60AD", "test": "#A3A9B5"}
DATASETS = ("df40", "140k", "openforensics", "mffi")
POSE_ORDER = ("frontal", "three_quarter", "profile", "unknown")
LIGHT_ORDER = ("dark", "medium", "bright", "unknown")


def style():
    sns.set_theme(style="whitegrid", context="notebook", font="DejaVu Sans")
    mpl.rcParams.update({
        "figure.facecolor": "#F6F8FC", "axes.facecolor": "white",
        "axes.spines.top": False, "axes.spines.right": False, "grid.alpha": 0.25,
        "axes.titleweight": "bold", "axes.titlesize": 11, "font.size": 10,
        "figure.dpi": 115, "savefig.dpi": 140, "axes.labelcolor": "#334155",
        "text.color": "#17243B", "axes.edgecolor": "#D7DFEB",
    })


def _first(paths):
    for path in paths:
        if path.exists():
            return path
    raise FileNotFoundError("Missing prepared artifact: " + " or ".join(map(str, paths)))


def load_df40(root):
    raw_root = Path(os.environ.get("DEEPFAKE_DATA_ROOT", str(root / "data/raw"))).expanduser()
    if not raw_root.is_absolute():
        raw_root = root / raw_root
    raw = raw_root / "df40"
    snapshot = root / "reports/generated/acquisition/df40"
    inventory = pd.read_parquet(_first([raw / "inventory.parquet", root / "data/manifests/df40_inventory.parquet"]))
    faces = pd.read_parquet(root / "data/processed/df40/metadata.parquet")
    sample_paths = [raw / "sample.parquet", raw / "selected.parquet", snapshot / "sample.parquet"]
    if any(path.is_file() for path in sample_paths):
        sample = pd.read_parquet(_first(sample_paths))
    else:
        if "image_sample_id" not in faces:
            raise FileNotFoundError("Нет acquisition sample и image_sample_id для восстановления metadata-выборки")
        sample = faces.drop_duplicates("image_sample_id").copy()
        sample["sample_id"] = sample["image_sample_id"]
        sample.attrs["artifact_source"] = "reconstructed_from_processed_image_metadata"
    sampling = json.loads(_first([raw / "sampling.json", raw / "metadata/sampling.json",
                                  snapshot / "sampling.json"]).read_text())
    audit = json.loads((root / "data/processed/df40/preprocessing_audit.json").read_text())
    return inventory, sample, faces, sampling, audit


def labels(frame):
    result = frame.copy()
    result["class"] = result["target"].map({0: "real", 1: "fake"}).fillna("unknown")
    return result


def stage_table(inventory, sample, faces):
    rows = []
    for name, frame in (("Инвентарь native stills", inventory), ("Выбранные изображения", sample), ("Строки обработки (включая ошибки)", faces)):
        rows.append({
            "этап": name, "строк": len(frame), "real": int(frame["target"].eq(0).sum()),
            "fake": int(frame["target"].eq(1).sum()),
            "генераторов fake": frame.loc[frame["target"].eq(1), "generator"].nunique(),
            "записей bbox (face_index)": int(frame["face_index"].notna().sum()) if "face_index" in frame else pd.NA,
            "сохранённых кропов": int(frame["crop_path"].notna().sum()) if "crop_path" in frame else pd.NA,
            "eligible": int(frame["eligible"].fillna(False).sum()) if "eligible" in frame else pd.NA,
        })
    return pd.DataFrame(rows).set_index("этап")


def divergence(left, right):
    categories = sorted(set(left.index.astype(str)) | set(right.index.astype(str)))
    left, right = left.copy(), right.copy()
    left.index, right.index = left.index.astype(str), right.index.astype(str)
    p = left.reindex(categories, fill_value=0).to_numpy(float)
    q = right.reindex(categories, fill_value=0).to_numpy(float)
    if not p.sum() or not q.sum():
        return {"TV": float("nan"), "JS_bits": float("nan"), "max_gap_pp": float("nan")}
    p = p / p.sum()
    q = q / q.sum()
    midpoint = (p + q) / 2
    a = p > 0
    b = q > 0
    js = 0.5 * np.sum(p[a] * np.log2(p[a] / midpoint[a]))
    js += 0.5 * np.sum(q[b] * np.log2(q[b] / midpoint[b]))
    return {"TV": float(np.abs(p - q).sum() / 2), "JS_bits": float(js),
            "max_gap_pp": float(np.max(np.abs(p - q)) * 100) if len(p) else 0.0}


def sampling_fidelity(inventory, sample):
    rows = []
    for column in ("target", "generator", "native_method", "native_family", "manipulation", "domain", "author_split", "source_format"):
        if column in inventory and column in sample:
            values = divergence(inventory[column].fillna("unknown").astype(str).value_counts(),
                                sample[column].fillna("unknown").astype(str).value_counts())
            rows.append({"признак": column, **values})
    return pd.DataFrame(rows).set_index("признак")


def plot_sampling(inventory, sample):
    columns = [column for column in ("native_method", "domain", "author_split", "source_format") if column in inventory and column in sample]
    fig, axes = plt.subplots(2, 2, figsize=(15, 11), layout="constrained")
    for ax, column in zip(axes.ravel(), columns):
        p = inventory[column].fillna("unknown").astype(str).value_counts(normalize=True)
        q = sample[column].fillna("unknown").astype(str).value_counts(normalize=True)
        order = p.sort_values().index
        y = np.arange(len(order))
        ax.barh(y - 0.18, 100 * p.reindex(order, fill_value=0), height=0.34, color="#B3C4DB", label="инвентарь")
        ax.barh(y + 0.18, 100 * q.reindex(order, fill_value=0), height=0.34, color=COLORS["train"], label="сабсет")
        ax.set_yticks(y, order)
        ax.set_xlabel("доля изображений, %")
        ax.set_title(column)
        ax.legend(frameon=False, fontsize=8)
    for ax in axes.ravel()[len(columns):]:
        ax.axis("off")
    fig.suptitle("Стратифицированная выборка сохраняет доступные метаданные", fontsize=16)
    return fig


def plot_composition(frame):
    frame = labels(frame)
    columns = ["manipulation", "domain", "author_split"]
    if "native_family" in frame:
        columns.insert(0, "native_family")
    if len(columns) == 4:
        fig, axes = plt.subplots(2, 2, figsize=(15, 8.5), layout="constrained")
    else:
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), layout="constrained")
    for ax, column in zip(np.asarray(axes).ravel(), columns):
        table = pd.crosstab(frame[column].fillna("unknown"), frame["class"]).reindex(columns=["real", "fake"], fill_value=0)
        table.plot.bar(ax=ax, stacked=True, color=[COLORS["real"], COLORS["fake"]], width=0.7, rot=0)
        ax.set_title(column)
        ax.set_ylabel("изображений")
        ax.set_xlabel("")
        ax.legend(title="", frameon=False)
        ax.ticklabel_format(axis="y", style="plain")
    fig.suptitle("Native-семья архива и фактический тип изображения", fontsize=16)
    return fig


def image_table(faces):
    key = "image_sample_id" if "image_sample_id" in faces else "input_path"
    return faces.drop_duplicates(key).copy()


def plot_examples(faces, seed=42):
    candidates = faces.loc[faces["eligible"] & faces["crop_path"].notna()].copy()
    selected = []
    for target in (0, 1):
        part = candidates.loc[candidates["target"].eq(target)].sample(frac=1, random_state=seed)
        spread = part.drop_duplicates("generator")
        chosen = pd.concat([spread, part]).drop_duplicates("sample_id").head(3)
        selected.extend(chosen.to_dict("records"))
    if not selected:
        raise ValueError("No eligible face crops available for examples")
    fig, axes = plt.subplots(2, len(selected), figsize=(3.2 * len(selected), 6.8), squeeze=False, layout="constrained")
    for column, row in enumerate(selected):
        raw = cv2.imread(str(row["input_path"]))
        crop = cv2.imread(str(row["crop_path"]))
        if raw is None or crop is None:
            raise FileNotFoundError(f"Cannot decode example {row['sample_id']}")
        axes[0, column].imshow(cv2.cvtColor(raw, cv2.COLOR_BGR2RGB))
        if row.get("bbox") is not None:
            x, y, width, height = row["bbox"]
            axes[0, column].add_patch(Rectangle((x, y), width, height, fill=False, lw=2,
                                               edgecolor=COLORS["fake" if row["target"] else "real"]))
        axes[1, column].imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        name = str(row["generator"])
        axes[0, column].set_title(name[:28] + "\n" + ("fake" if row["target"] else "real"), fontsize=9)
        axes[1, column].set_title(f"{row.get('source_format', 'unknown')} → PNG\n{row.get('yaw_bin', 'unknown')} | {row.get('light_bin', 'unknown')}", fontsize=9)
        axes[0, column].axis("off")
        axes[1, column].axis("off")
    fig.suptitle("Исходные изображения и единый кроп 256 × 256", fontsize=16)
    return fig


def quality_sample(faces, maximum=3000, seed=42):
    eligible = faces.loc[faces["eligible"] & faces["crop_path"].notna()]
    selected = eligible.sample(min(maximum, len(eligible)), random_state=seed)
    rows = []
    for row in selected.to_dict("records"):
        image = cv2.imread(str(row["crop_path"]))
        if image is None:
            raise FileNotFoundError(row["crop_path"])
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        side = 64
        gray_small = cv2.resize(gray, (side, side), interpolation=cv2.INTER_AREA).astype(float)
        energy = np.abs(np.fft.fftshift(np.fft.fft2(gray_small - gray_small.mean()))) ** 2
        y, x = np.ogrid[-side // 2:side // 2, -side // 2:side // 2]
        high = np.sqrt(x * x + y * y) >= side / 3
        rows.append({"sample_id": row["sample_id"], "target": row["target"],
                     "sharpness_laplacian": laplacian, "hf_ratio": float(energy[high].sum() / max(energy.sum(), 1)),
                     "source_kib": Path(row["input_path"]).stat().st_size / 1024})
    return pd.DataFrame(rows, columns=["sample_id", "target", "sharpness_laplacian", "hf_ratio", "source_kib"])


def plot_image_properties(faces, quality):
    images = labels(image_table(faces))
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), layout="constrained")
    for target, name in ((0, "real"), (1, "fake")):
        selected = images.loc[images["target"].eq(target)]
        if "image_width" in selected:
            axes[0, 0].hist(selected["image_width"].dropna(), bins=35, alpha=0.55, color=COLORS[name], label=name)
        if "image_height" in selected:
            axes[0, 1].hist(selected["image_height"].dropna(), bins=35, alpha=0.55, color=COLORS[name], label=name)
    table = pd.crosstab(images["source_format"], images["class"]).reindex(columns=["real", "fake"], fill_value=0)
    table.plot.bar(ax=axes[0, 2], color=[COLORS["real"], COLORS["fake"]], rot=25)
    axes[0, 0].set_title("Ширина исходника")
    axes[0, 1].set_title("Высота исходника")
    axes[0, 2].set_title("Формат по заголовку исходника")
    for ax in axes[0, :2]:
        ax.set_xlabel("px")
        ax.set_ylabel("изображений")
        if ax.get_legend_handles_labels()[0]:
            ax.legend(frameon=False)
    axes[0, 2].set_xlabel("")
    axes[0, 2].set_ylabel("изображений")
    axes[0, 2].legend(title="", frameon=False)
    for ax, column, title in zip(axes[1, :2], ("sharpness_laplacian", "hf_ratio"), ("Чёткость: дисперсия Лапласиана", "Доля высоких частот")):
        for target, name in ((0, "real"), (1, "fake")):
            values = quality.loc[quality["target"].eq(target), column].dropna()
            if column == "sharpness_laplacian":
                values = np.log1p(values)
            ax.hist(values, bins=35, density=True, alpha=0.55, color=COLORS[name], label=name)
        ax.set_title(title)
        ax.set_xlabel("log(1 + variance)" if column == "sharpness_laplacian" else "доля энергии")
        ax.set_ylabel("плотность")
        if ax.get_legend_handles_labels()[0]:
            ax.legend(frameon=False)
    for target, name in ((0, "real"), (1, "fake")):
        values = faces.loc[faces["target"].eq(target), "padding_fraction"].dropna()
        axes[1, 2].hist(values, bins=np.linspace(0, 1, 26), alpha=0.55, color=COLORS[name], label=name)
    axes[1, 2].set_title("Фактический padding после adaptive-кропа")
    axes[1, 2].set_xlabel("доля площади")
    axes[1, 2].set_ylabel("лиц")
    axes[1, 2].legend(frameon=False)
    fig.suptitle("Геометрия, кодек исходника и простые признаки качества", fontsize=16)
    return fig


def plot_attributes(faces):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), layout="constrained")
    for ax, column in zip(axes.ravel()[:4], ("yaw", "pitch", "roll", "lightness_lstar")):
        for target, name in ((0, "real"), (1, "fake")):
            values = pd.to_numeric(faces.loc[faces["target"].eq(target), column], errors="coerce").dropna()
            if len(values):
                ax.hist(values, bins=np.linspace(0, 100, 35) if column == "lightness_lstar" else np.linspace(-90, 90, 50) if column == "yaw" else np.linspace(-180, 180, 65),
                        density=True, alpha=0.55, color=COLORS[name], label=name)
        ax.set_title(column)
        ax.set_xlabel("L*" if column == "lightness_lstar" else "градусы")
        ax.set_ylabel("плотность")
        if ax.get_legend_handles_labels()[0]:
            ax.legend(frameon=False)
    for ax, column, order in ((axes[1, 1], "yaw_bin", POSE_ORDER), (axes[1, 2], "light_bin", LIGHT_ORDER)):
        data = labels(faces)
        table = pd.crosstab(data[column].fillna("unknown"), data["class"], normalize="columns")
        table.reindex(index=order, columns=["real", "fake"], fill_value=0).plot.bar(
            ax=ax, color=[COLORS["real"], COLORS["fake"]], rot=20,
        )
        ax.set_title(column)
        ax.set_xlabel("")
        ax.set_ylabel("доля внутри класса")
        ax.legend(title="", frameon=False)
    fig.suptitle("Поза и освещение измерены одинаковым инструментом", fontsize=16)
    return fig


def flag_counts(frame, column="flags"):
    counts = Counter()
    if column in frame:
        for flags in frame[column]:
            if isinstance(flags, (list, tuple, np.ndarray, set)):
                counts.update(flags)
    return pd.Series(counts, dtype=int).sort_values(ascending=False).rename("строк")


def missing_table(faces):
    rows = []
    for column in ("yaw", "pitch", "roll", "lightness_lstar", "face_mask_fraction_of_crop", "source_format"):
        for target, part in faces.groupby("target", dropna=False):
            missing = part[column].isna() | part[column].astype(str).isin(["", "unknown"])
            rows.append({"target": target, "признак": column, "не измерено": int(missing.sum()),
                         "доля": float(missing.mean())})
    return pd.DataFrame(rows)


def plot_flags(faces):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.4), layout="constrained")
    for ax, column, title in ((axes[0], "flags", "Все флаги, включая информационные"),
                               (axes[1], "exclusion_flags", "Причины исключения до сплитов")):
        values = flag_counts(faces, column).head(14).sort_values()
        if len(values):
            values.plot.barh(ax=ax, color=COLORS["fake"] if column == "exclusion_flags" else COLORS["train"])
        else:
            ax.text(0.5, 0.5, "Флагов нет", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        ax.set_xlabel("строк обработки")
    return fig


def load_composite(root, relative="data/splits/composite"):
    output = root / relative
    parts = {partition: pd.read_parquet(output / f"{partition}.parquet")
             for partition in ("train", "validate", "test", "excluded")}
    audit = json.loads((output / "audit.json").read_text())
    return parts, audit


def split_divergences(parts):
    rows = []
    for left_name, right_name in (("train", "validate"), ("train", "test"), ("validate", "test")):
        data = pd.concat([parts[left_name].assign(split=left_name), parts[right_name].assign(split=right_name)], ignore_index=True)
        for feature in ("generator", "manipulation", "domain", "source_format", "yaw_bin", "pitch_bin", "roll_bin", "light_bin", "face_size_bin", "padding_bin", "source_resolution_bin", "face_mask_fraction_of_crop_bin"):
            condition = ["dataset", "target"]
            if feature == "source_format":
                condition.append("generator")
            if feature not in data:
                continue
            for key, group in data.groupby(condition, dropna=False, sort=True):
                key = key if isinstance(key, tuple) else (key,)
                left = group.loc[group["split"].eq(left_name), feature].fillna("unknown").astype(str).value_counts()
                right = group.loc[group["split"].eq(right_name), feature].fillna("unknown").astype(str).value_counts()
                rows.append({**dict(zip(condition, key)), "feature": feature,
                             "pair": left_name + " / " + right_name,
                             "left_n": int(left.sum()), "right_n": int(right.sum()), **divergence(left, right)})
        if "transformation_tags" in data:
            for key, group in data.groupby(["dataset", "target"], dropna=False, sort=True):
                tags = group["transformation_tags"].map(
                    lambda value: set(map(str, value)) if isinstance(value, (list, tuple, set, np.ndarray)) and len(value) else {"unknown"}
                )
                for tag in sorted(set().union(*tags)):
                    present = tags.map(lambda values: tag in values)
                    left_mask = group["split"].eq(left_name)
                    right_mask = group["split"].eq(right_name)
                    left_total, right_total = int(left_mask.sum()), int(right_mask.sum())
                    left_present = int((present & left_mask).sum())
                    right_present = int((present & right_mask).sum())
                    left = pd.Series({"present": left_present, "absent": left_total - left_present})
                    right = pd.Series({"present": right_present, "absent": right_total - right_present})
                    rows.append({
                        "dataset": key[0], "target": key[1], "feature": "transformation_tags",
                        "tag": tag, "pair": left_name + " / " + right_name,
                        "left_n": left_total, "right_n": right_total, **divergence(left, right),
                    })
    return pd.DataFrame(rows)


def plot_split_counts(parts):
    data = pd.concat([parts["train"].assign(split="train"), parts["validate"].assign(split="validate"), parts["test"].assign(split="test")], ignore_index=True)
    data["dataset"] = data["dataset"].str.lower()
    data = labels(data)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), layout="constrained")
    table = pd.crosstab([data["dataset"], data["split"]], data["class"]).reindex(columns=["real", "fake"], fill_value=0)
    table.plot.bar(ax=axes[0], stacked=True, color=[COLORS["real"], COLORS["fake"]], rot=50)
    axes[0].set_title("Число лиц по источнику и классу")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("лиц")
    axes[0].legend(title="", frameon=False)
    for ax, column in zip(axes[1:], ("dataset", "source_format")):
        table = pd.crosstab(data[column].fillna("unknown"), data["split"], normalize="columns")
        table.reindex(columns=["train", "validate", "test"], fill_value=0).plot.bar(
            ax=ax, color=[COLORS["train"], COLORS["validate"], COLORS["test"]], rot=30,
        )
        ax.set_title("Доли: " + column)
        ax.set_ylabel("доля внутри сплита")
        ax.set_xlabel("")
        ax.legend(title="", frameon=False)
    fig.suptitle("Итоговый композит: train, validate и test", fontsize=16)
    return fig


def plot_conditional_bins(parts, target, features=("source_format", "yaw_bin", "light_bin")):
    data = pd.concat([parts["train"].assign(split="train"), parts["validate"].assign(split="validate"), parts["test"].assign(split="test")], ignore_index=True)
    data["dataset"] = data["dataset"].str.lower()
    data = data.loc[data["target"].eq(target)]
    datasets = [dataset for dataset in DATASETS if dataset in set(data["dataset"])]
    fig, axes = plt.subplots(len(datasets), 3, figsize=(16, 3.2 * len(datasets)), squeeze=False, layout="constrained")
    for row, dataset in enumerate(datasets):
        group = data.loc[data["dataset"].eq(dataset)]
        for ax, feature in zip(axes[row], features):
            table = pd.crosstab(group[feature].fillna("unknown"), group["split"], normalize="columns")
            if feature in ("yaw_bin", "light_bin"):
                table = table.reindex(POSE_ORDER if feature == "yaw_bin" else LIGHT_ORDER, fill_value=0)
            table.reindex(columns=["train", "validate", "test"], fill_value=0).plot.bar(
                ax=ax, color=[COLORS["train"], COLORS["validate"], COLORS["test"]], rot=15, width=0.75,
            )
            ax.set_title(f"{dataset} | {feature}")
            ax.set_xlabel("")
            ax.set_ylabel("доля внутри dataset + target")
            ax.legend(title="", frameon=False, fontsize=8)
    fig.suptitle("Условные распределения: " + ("real" if target == 0 else "fake"), fontsize=16)
    return fig


def plot_split_continuous(parts):
    data = pd.concat([parts["train"].assign(split="train"), parts["validate"].assign(split="validate"), parts["test"].assign(split="test")], ignore_index=True)
    data["dataset"] = data["dataset"].str.lower()
    datasets = [dataset for dataset in DATASETS if dataset in set(data["dataset"])]
    fig, axes = plt.subplots(len(datasets), 2, figsize=(14, 3 * len(datasets)), squeeze=False, layout="constrained")
    for row, dataset in enumerate(datasets):
        group = data.loc[data["dataset"].eq(dataset)]
        for ax, feature in zip(axes[row], ("yaw", "lightness_lstar")):
            for partition in ("train", "validate", "test"):
                values = pd.to_numeric(group.loc[group["split"].eq(partition), feature], errors="coerce").dropna()
                if feature == "yaw":
                    values = values.abs()
                if len(values):
                    ax.hist(values, bins=np.linspace(0, 90 if feature == "yaw" else 100, 35),
                            density=True, histtype="step", lw=2, color=COLORS[partition], label=partition)
            ax.set_title(f"{dataset} | " + ("|yaw|" if feature == "yaw" else "L*"))
            ax.set_xlabel("градусы" if feature == "yaw" else "L*")
            ax.set_ylabel("плотность")
            if ax.get_legend_handles_labels()[0]:
                ax.legend(frameon=False)
    fig.suptitle("Непрерывные признаки дополняют сравнение бинов", fontsize=16)
    return fig


def plot_divergences(table):
    summary = table.groupby("feature").agg(TV_mean=("TV", "mean"), TV_max=("TV", "max")).sort_values("TV_max")
    fig, ax = plt.subplots(figsize=(11, 5), layout="constrained")
    positions = np.arange(len(summary))
    ax.hlines(positions, summary["TV_mean"], summary["TV_max"], color="#B5BFD2", lw=3)
    ax.scatter(summary["TV_mean"], positions, s=70, color=COLORS["train"], label="средний TV по стратам", zorder=3)
    ax.scatter(summary["TV_max"], positions, s=70, color=COLORS["validate"], label="максимальный TV", zorder=3)
    ax.set_yticks(positions, summary.index)
    ax.set_xlabel("Total variation: 0 = распределения совпадают")
    ax.set_title("Где остаётся дисбаланс train / validate / test")
    if ax.get_legend_handles_labels()[0]:
        ax.legend(frameon=False)
    return fig


def overlap_audit(parts):
    combined = pd.concat([parts[key].assign(checked_split=key) for key in ("train", "validate", "test")], ignore_index=True)
    rows = []
    for column in ("group_id", "image_sha256", "decoded_sha256", "crop_sha256"):
        if column not in combined:
            continue
        present = combined[column].notna() & combined[column].astype(str).ne("")
        group = combined.loc[present].groupby(column)["checked_split"].nunique()
        rows.append({"ключ": column, "групп": len(group), "между сплитами": int(group.gt(1).sum()),
                     "без ключа": int((~present).sum())})
    return pd.DataFrame(rows).set_index("ключ")


def exclusion_table(parts):
    rows = []
    for dataset, group in parts["excluded"].groupby("dataset"):
        for flag, count in flag_counts(group).items():
            rows.append({"dataset": dataset, "reason": flag, "records": int(count)})
    return pd.DataFrame(rows, columns=["dataset", "reason", "records"])


def plot_exclusions(table):
    fig, ax = plt.subplots(figsize=(13, 5.2), layout="constrained")
    if table.empty:
        ax.text(0.5, 0.5, "Исключённых лиц нет", ha="center", va="center", transform=ax.transAxes)
    else:
        matrix = table.pivot_table(index="reason", columns="dataset", values="records", aggfunc="sum", fill_value=0)
        order = matrix.sum(axis=1).sort_values().tail(12).index
        matrix.loc[order].plot.barh(ax=ax, stacked=True, colormap="Set2")
    ax.set_title("Почему записи не попали в обучение")
    ax.set_xlabel("строк обработки; одна строка может иметь несколько причин")
    ax.set_ylabel("")
    return fig


def plot_source_files(quality):
    fig, ax = plt.subplots(figsize=(10, 4.2), layout="constrained")
    for target, name in ((0, "real"), (1, "fake")):
        values = quality.loc[quality["target"].eq(target), "source_kib"].dropna()
        if len(values):
            ax.hist(np.log1p(values), bins=40, density=True, alpha=0.55, color=COLORS[name], label=name)
    ax.set_xlabel("log(1 + размер исходного файла в KiB)")
    ax.set_ylabel("плотность")
    ax.set_title("Размер исходного файла тоже может выдавать класс")
    if ax.get_legend_handles_labels()[0]:
        ax.legend(frameon=False)
    return fig


def _boundary_flags(value):
    if isinstance(value, (list, tuple, np.ndarray, set)):
        return set(map(str, value))
    if value is None or value is pd.NA:
        return set()
    return {item for item in str(value).split(";") if item and item != "nan"}


def _face_area(value):
    try:
        box = np.asarray(value, dtype=float)
        if box.shape == (4,) and np.isfinite(box).all() and min(box[2:]) > 0:
            return float(box[2] * box[3])
    except (TypeError, ValueError):
        pass
    return -1.0


def _boundary_metrics(images):
    total = len(images)
    measured = images["primary_padding_measured"]
    values = images.loc[measured, "primary_padding_fraction"]
    known = len(values)
    positive = values.loc[values.gt(0)]
    dimensions = images["raw_dimensions_known"]
    square = images["raw_square_input"]
    result = {
        "input_images": total,
        "measured_primary_images": known,
        "unmeasured_primary_images": total - known,
        "primary_bad_images": int((~images["primary_eligible"]).sum()),
        "images_with_any_bad_face": int(images["any_bad_face"].sum()),
        "no_face_images": int(images["no_face_flag"].sum()),
        "decoding_failed_images": int(images["decoding_failed_flag"].sum()),
        "images_with_multiple_face_records": int(images["face_records"].gt(1).sum()),
        "images_any_face_padded": int(images["any_face_padding_fraction_max"].gt(0).sum()),
        "actual_images_padded": int(images["actual_primary_padding_fraction"].gt(0).sum()),
        "actual_measured_images": int(images["actual_primary_padding_fraction"].notna().sum()),
        "actual_share_images_padded_all": float(images["actual_primary_padding_fraction"].gt(0).sum() / total) if total else np.nan,
        "raw_dimensions_known_images": int(dimensions.sum()),
        "raw_square_images": int(square.sum()),
        "share_square_inputs_all": float(square.sum() / total) if total else np.nan,
        "share_square_inputs_known_dimensions": float(square.sum() / dimensions.sum()) if dimensions.sum() else np.nan,
        "raw_aspect_ratio_median": float(images.loc[dimensions, "raw_aspect_ratio"].median()),
        "padding_area_median": float(values.median()) if known else np.nan,
        "padding_area_p95": float(values.quantile(0.95)) if known else np.nan,
        "padding_area_median_among_padded": float(positive.median()) if len(positive) else np.nan,
        "padding_area_p95_among_padded": float(positive.quantile(0.95)) if len(positive) else np.nan,
    }
    for suffix, threshold in (("padded", 0.0), ("padding_gt05", 0.05), ("padding_gt10", 0.10)):
        count = int(values.gt(threshold).sum())
        result[f"images_{suffix}"] = count
        result[f"share_images_{suffix}_all"] = count / total if total else np.nan
        result[f"share_images_{suffix}_measured"] = count / known if known else np.nan
    return result


def crop_boundary_summary(faces):
    frame = faces.copy()
    if "image_sample_id" in frame:
        key = frame["image_sample_id"].astype("string")
    else:
        key = pd.Series(pd.NA, index=frame.index, dtype="string")
    for fallback in ("input_path", "sample_id"):
        if fallback in frame:
            key = key.fillna(frame[fallback].astype("string"))
    if key.isna().any():
        raise ValueError("Input image identity is required for crop-boundary statistics")
    frame["input_image_key"] = key
    frame["_face_area"] = frame["bbox"].map(_face_area) if "bbox" in frame else -1.0
    frame["_face_order"] = pd.to_numeric(frame.get("face_index", pd.Series(0, index=frame.index)), errors="coerce").fillna(0)
    actual_padding = pd.to_numeric(frame.get("padding_fraction", pd.Series(np.nan, index=frame.index)), errors="coerce")
    if "proposed_padding_fraction" in frame:
        padding = pd.to_numeric(frame["proposed_padding_fraction"], errors="coerce")
        legacy = frame.get("crop_policy", pd.Series("", index=frame.index)).ne("adaptive")
        padding = padding.combine_first(actual_padding.where(legacy))
        frame["_padding_basis"] = np.where(frame["proposed_padding_fraction"].notna(), "proposed_1.3", "legacy_actual")
    else:
        padding = actual_padding
        frame["_padding_basis"] = "legacy_actual"
    valid_face = frame["_face_area"].gt(0)
    frame["_measured_padding"] = padding.where(padding.between(0, 1) & valid_face)
    frame["_actual_padding"] = actual_padding.where(actual_padding.between(0, 1) & valid_face)
    frame["_bad_face"] = ~frame.get("eligible", pd.Series(False, index=frame.index)).fillna(False).astype(bool)
    flags = frame["flags"].map(_boundary_flags) if "flags" in frame else pd.Series([set() for _ in range(len(frame))], index=frame.index)
    frame["_no_face"] = flags.map(lambda value: "no_face" in value)
    frame["_decoding_failed"] = flags.map(lambda value: "decoding_failed" in value)
    primary = frame.sort_values(
        ["input_image_key", "_face_area", "_face_order"], ascending=[True, False, True], kind="stable",
    ).drop_duplicates("input_image_key").copy()
    grouped = frame.groupby("input_image_key", sort=False)
    image_features = grouped.agg(
        face_records=("_face_area", "size"),
        any_face_padding_fraction_max=("_measured_padding", "max"),
        any_bad_face=("_bad_face", "any"),
        no_face_flag=("_no_face", "any"),
        decoding_failed_flag=("_decoding_failed", "any"),
    )
    primary = primary.merge(image_features, left_on="input_image_key", right_index=True, how="left", validate="one_to_one")
    primary["primary_padding_fraction"] = primary["_measured_padding"]
    primary["padding_diagnostic_basis"] = primary["_padding_basis"]
    primary["actual_primary_padding_fraction"] = primary["_actual_padding"]
    primary["primary_padding_measured"] = primary["primary_padding_fraction"].notna()
    primary["primary_eligible"] = ~primary["_bad_face"]
    width = pd.to_numeric(primary.get("image_width", pd.Series(np.nan, index=primary.index)), errors="coerce")
    height = pd.to_numeric(primary.get("image_height", pd.Series(np.nan, index=primary.index)), errors="coerce")
    primary["raw_dimensions_known"] = width.gt(0) & height.gt(0) & np.isfinite(width) & np.isfinite(height)
    primary["raw_aspect_ratio"] = (width / height).where(primary["raw_dimensions_known"])
    primary["raw_square_input"] = width.eq(height) & primary["raw_dimensions_known"]
    primary = primary.drop(columns=[column for column in primary if column.startswith("_")])
    result = {"images": primary.reset_index(drop=True), "overall": pd.DataFrame([_boundary_metrics(primary)])}
    for column in ("native_method", "target", "source_format"):
        if column not in primary:
            result["by_" + column] = pd.DataFrame()
            continue
        rows = []
        for value, images in primary.groupby(column, dropna=False, sort=True):
            rows.append({column: value, **_boundary_metrics(images)})
        result["by_" + column] = pd.DataFrame(rows)
    return result


def plot_crop_boundaries(summary):
    overall = summary["overall"].iloc[0]
    images = summary["images"]
    fig, axes = plt.subplots(2, 2, figsize=(16, 10.5), layout="constrained")
    suffixes = ("padded", "padding_gt05", "padding_gt10")
    labels_ = ("хотя бы 1 px", "площадь > 5%", "площадь > 10%")
    positions = np.arange(3)
    axes[0, 0].bar(positions - 0.18, [100 * overall[f"share_images_{key}_all"] for key in suffixes],
                   width=0.34, color=COLORS["train"], label="от всех исходников")
    axes[0, 0].bar(positions + 0.18, [100 * overall[f"share_images_{key}_measured"] for key in suffixes],
                   width=0.34, color=COLORS["validate"], label="от измеренных primary-кропов")
    axes[0, 0].set_xticks(positions, labels_, rotation=10)
    axes[0, 0].set_ylabel("доля исходных изображений, %")
    axes[0, 0].set_title("Как часто квадрат выходит за границы")
    axes[0, 0].legend(frameon=False, fontsize=9)
    values = images.loc[images["primary_padding_measured"], "primary_padding_fraction"].sort_values().to_numpy()
    if len(values):
        axes[0, 1].step(values, np.arange(1, len(values) + 1) / len(values), where="post", color=COLORS["train"], lw=2)
    axes[0, 1].set_xlim(0, 1)
    axes[0, 1].set_ylim(0, 1.02)
    axes[0, 1].set_xlabel("доля недостающей площади одного primary-кропа")
    axes[0, 1].set_ylabel("накопленная доля измеренных изображений")
    axes[0, 1].set_title("Площадь padding внутри кропа: отдельная метрика")
    methods = summary["by_native_method"]
    if len(methods):
        methods = methods.sort_values("share_images_padded_measured")
        axes[1, 0].barh(methods["native_method"].astype(str), methods["share_images_padded_measured"] * 100,
                       color=COLORS["fake"], height=0.7)
    axes[1, 0].set_xlabel("изображений с padding, % от измеренных")
    axes[1, 0].set_title("Выход за границу по native-методам")
    ratios = images.loc[images["raw_dimensions_known"], "raw_aspect_ratio"].dropna()
    if len(ratios):
        axes[1, 1].hist(ratios, bins=35, color="#598B9C", alpha=0.85)
    axes[1, 1].axvline(1, color="#334155", ls="--", lw=1)
    axes[1, 1].set_xlabel("ширина исходника / высота исходника")
    axes[1, 1].set_ylabel("исходных изображений")
    axes[1, 1].set_title(f"Квадратных исходников: {overall['share_square_inputs_known_dimensions']:.1%}")
    fig.suptitle("Предложенный квадрат 1.3 × bbox до adaptive-ограничения", fontsize=16)
    return fig


def taxonomy_table(parts):
    data = pd.concat([parts[name] for name in ("train", "validate", "test", "excluded")], ignore_index=True)
    tables = []
    for field in ("generator", "manipulation", "transformation"):
        raw = field + "_raw"
        if field not in data or raw not in data:
            continue
        table = data[["dataset", raw, field]].copy()
        for column in (raw, field):
            table[column] = table[column].map(
                lambda value: json.dumps(value.tolist() if isinstance(value, np.ndarray) else value,
                                         ensure_ascii=False, sort_keys=True, default=str)
                if isinstance(value, (list, tuple, dict, np.ndarray)) else str(value)
            )
        table = table.groupby(["dataset", raw, field], dropna=False).size().rename("records").reset_index()
        table = table.rename(columns={raw: "raw", field: "canonical"})
        table["field"] = field
        tables.append(table[["dataset", "field", "raw", "canonical", "records"]])
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame(
        columns=["dataset", "field", "raw", "canonical", "records"],
    )


def plot_face_mask_fraction(faces):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), layout="constrained")
    for target, name in ((0, "real"), (1, "fake")):
        values = pd.to_numeric(faces.loc[faces["target"].eq(target), "face_mask_fraction_of_crop"], errors="coerce").dropna()
        if len(values):
            axes[0].hist(values, bins=np.linspace(0, 1, 31), density=True, histtype="step", lw=2,
                         color=COLORS[name], label=name)
    for edge in (0.25, 0.5, 0.75):
        axes[0].axvline(edge, color="#A3A9B5", ls=":", lw=1)
    axes[0].set_xlabel("площадь выбранной маски внутри кропа / площадь кропа")
    axes[0].set_ylabel("плотность")
    axes[0].set_title("Какую долю фактического кропа занимает лицо")
    if axes[0].get_legend_handles_labels()[0]:
        axes[0].legend(frameon=False)
    data = labels(faces)
    sources = data.get("mask_source", pd.Series("missing", index=data.index)).fillna("missing")
    table = pd.crosstab(sources, data["class"], normalize="columns").reindex(columns=["real", "fake"], fill_value=0)
    table.plot.bar(ax=axes[1], color=[COLORS["real"], COLORS["fake"]], rot=15)
    axes[1].set_title("Источник маски: только аудит покрытия")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("доля внутри класса")
    axes[1].legend(title="", frameon=False)
    return fig


def plot_split_mask_fraction(parts):
    data = pd.concat([parts[name].assign(split=name) for name in ("train", "validate", "test")], ignore_index=True)
    data["dataset"] = data["dataset"].str.lower()
    datasets = [name for name in DATASETS if name in set(data["dataset"])]
    fig, axes = plt.subplots(len(datasets), 2, figsize=(14, 3 * len(datasets)), squeeze=False, layout="constrained")
    for row, dataset in enumerate(datasets):
        for target in (0, 1):
            ax = axes[row, target]
            group = data.loc[data["dataset"].eq(dataset) & data["target"].eq(target)]
            for partition in ("train", "validate", "test"):
                values = pd.to_numeric(group.loc[group["split"].eq(partition), "face_mask_fraction_of_crop"], errors="coerce").dropna()
                if len(values):
                    ax.hist(values, bins=np.linspace(0, 1, 31), density=True, histtype="step", lw=2,
                            color=COLORS[partition], label=partition)
            for edge in (0.25, 0.5, 0.75):
                ax.axvline(edge, color="#A3A9B5", ls=":", lw=0.8)
            ax.set_title(f"{dataset} | " + ("real" if target == 0 else "fake"))
            ax.set_xlabel("маска лица / фактический кроп")
            ax.set_ylabel("плотность")
            if ax.get_legend_handles_labels()[0]:
                ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Доля маски лица: условные распределения во всех трёх сплитах", fontsize=16)
    return fig


def format_target_table(frame):
    data = frame.loc[frame["target"].isin([0, 1])].copy()
    data["source_format"] = data["source_format"].fillna("unknown").astype(str)
    table = pd.crosstab([data["dataset"], data["source_format"]], data["target"])
    table = table.reindex(columns=[0, 1], fill_value=0).rename(columns={0: "real", 1: "fake"}).reset_index()
    table["total"] = table["real"] + table["fake"]
    table["fake_share"] = table["fake"] / table["total"]
    table["only_target"] = np.select(
        [table["real"].eq(0), table["fake"].eq(0)], ["fake", "real"], default="mixed",
    )
    return table.sort_values(["dataset", "total"], ascending=[True, False]).reset_index(drop=True)


def format_target_finding(table):
    findings = []
    known = table.loc[~table["source_format"].str.lower().isin(["unknown", "none", "nan", ""])]
    for dataset, rows in known.groupby("dataset", sort=True):
        exclusive = rows.loc[rows["only_target"].ne("mixed")]
        if len(exclusive):
            values = [
                f"{row.source_format} только у {row.only_target} (лиц: {int(row.total):,})"
                for row in exclusive.itertuples(index=False)
            ]
            findings.append(f"**{dataset}**: " + "; ".join(values) + ".")
        else:
            findings.append(f"**{dataset}**: каждый представленный формат встречается у обоих таргетов.")
    return "\n\n".join(findings)


def mask_quality_summary(faces, threshold=0.25):
    rows = []
    for (dataset, target), group in faces.groupby(["dataset", "target"], dropna=False, sort=True):
        values = pd.to_numeric(group["face_mask_fraction_of_crop"], errors="coerce")
        known = values.notna() & values.between(0, 1)
        small = known & values.le(threshold)
        missing = ~known
        eligible = group["eligible"].fillna(False).astype(bool)
        exclusions = group.get("exclusion_flags", pd.Series([[] for _ in range(len(group))], index=group.index)).map(_boundary_flags)
        generator = group.get("generator", pd.Series("unknown", index=group.index)).fillna("unknown").astype(str).str.lower()
        rows.append({
            "dataset": dataset, "target": target, "processing_records": len(group),
            "known_mask_fraction": int(known.sum()), "missing_mask_fraction": int(missing.sum()),
            "mask_fraction_le_threshold": int(small.sum()),
            "excluded_face_mask_too_small": int(exclusions.map(lambda value: "face_mask_too_small" in value).sum()),
            "excluded_face_mask_missing": int(exclusions.map(lambda value: "face_mask_missing" in value).sum()),
            "eligible_current": int(eligible.sum()),
            "eligible_violating_mask_rule": int((eligible & (small | missing)).sum()),
            "eligible_unknown_generator": int((eligible & generator.isin(["", "unknown", "none", "nan"])).sum()),
        })
    return pd.DataFrame(rows)


def detector_coverage_summary(faces):
    images = image_table(faces).copy()
    native = pd.to_numeric(images.get("detector_native_faces", pd.Series(np.nan, index=images.index)), errors="coerce")
    downscaled = pd.to_numeric(images.get("detector_downscaled_faces", pd.Series(np.nan, index=images.index)), errors="coerce")
    merged = pd.to_numeric(images.get("detected_faces", pd.Series(np.nan, index=images.index)), errors="coerce")
    images["native_has_face"] = native.gt(0)
    images["downscaled_has_face"] = downscaled.gt(0)
    images["merged_has_face"] = merged.gt(0)
    images["detection_measured"] = native.notna() & merged.notna()
    images["rescued_by_downscale"] = native.eq(0) & merged.gt(0)
    images["native_face_lost"] = native.gt(0) & merged.eq(0)

    def summarize(group):
        total = len(group)
        return {
            "input_images": total,
            "measured_images": int(group["detection_measured"].sum()),
            "native_face_images": int(group["native_has_face"].sum()),
            "downscaled_face_images": int(group["downscaled_has_face"].sum()),
            "merged_face_images": int(group["merged_has_face"].sum()),
            "rescued_images": int(group["rescued_by_downscale"].sum()),
            "native_face_lost_images": int(group["native_face_lost"].sum()),
            "native_coverage_all": float(group["native_has_face"].sum() / total) if total else np.nan,
            "merged_coverage_all": float(group["merged_has_face"].sum() / total) if total else np.nan,
        }

    result = {"images": images, "overall": pd.DataFrame([summarize(images)])}
    for column in ("native_method", "target", "source_format"):
        rows = []
        if column in images:
            for key, group in images.groupby(column, sort=True, dropna=False):
                rows.append({column: key, **summarize(group)})
        result["by_" + column] = pd.DataFrame(rows)
    return result


def plot_detector_coverage(summary):
    table = summary["by_native_method"].sort_values("merged_coverage_all")
    fig, ax = plt.subplots(figsize=(13, 7.2), layout="constrained")
    positions = np.arange(len(table))
    ax.barh(positions - 0.18, table["native_coverage_all"] * 100, height=0.34,
            color="#AEC3DC", label="только native-проход")
    ax.barh(positions + 0.18, table["merged_coverage_all"] * 100, height=0.34,
            color=COLORS["train"], label="native + downscaled 320 после NMS")
    ax.set_yticks(positions, table["native_method"].astype(str))
    ax.set_xlim(0, 102)
    ax.set_xlabel("доля исходных изображений хотя бы с одной YuNet-детекцией, %")
    ax.set_title("Покрытие детектора по native-методам на одном фактическом прогоне")
    ax.legend(frameon=False)
    return fig


def near_duplicate_audit(audit):
    report = audit.get("near_duplicate_verification", {})
    if report.get("mode") != "pixel_verification":
        raise ValueError("Нужен актуальный аудит pixel_verification; пересоберите композитные сплиты")
    keys = ("candidate_pairs", "verified_pairs", "rejected_pairs", "unreadable_pairs")
    counts = {key: int(report[key]) for key in keys}
    if any(value < 0 for value in counts.values()):
        raise ValueError("Число пар не может быть отрицательным")
    if sum(counts[key] for key in keys[1:]) != counts["candidate_pairs"]:
        raise ValueError("Исходы проверки не совпадают с числом пар-кандидатов")
    outcomes = pd.DataFrame([
        {"outcome": label, "pairs": counts[key],
         "share": counts[key] / counts["candidate_pairs"] if counts["candidate_pairs"] else np.nan}
        for key, label in (("verified_pairs", "подтверждены пикселями / точным хешем"),
                           ("rejected_pairs", "отклонены проверкой"),
                           ("unreadable_pairs", "нет доступной совместимой пары изображений"))
    ])
    details = pd.DataFrame([
        {"metric": label, "count": int(report[key]), "unit": unit}
        for key, label, unit in (
            ("candidate_pairs", "все пары-кандидаты pHash", "пар"),
            ("exact_metadata_pairs", "подтверждены точным хешем", "пар"),
            ("verified_original_pairs", "подтверждены по исходнику", "пар"),
            ("verified_crop_pairs", "подтверждены по кропу", "пар"),
            ("low_texture_rejections", "отклонены с low-texture условием", "пар"),
            ("unreadable_rows", "строки без доступного сравнения", "строк"),
            ("excluded_rows", "исключены из-за недоступной проверки", "строк"),
        )
    ])
    settings = pd.Series(report["settings"], name="value")
    if "rgb_mae_max" in settings:
        settings.loc["rgb_mae_max_0_255"] = float(settings["rgb_mae_max"]) * 255
    if "min_gray_std" in settings:
        settings.loc["min_gray_std_0_255"] = float(settings["min_gray_std"]) * 255
    return {"outcomes": outcomes, "details": details, "settings": settings.to_frame()}


def plot_near_duplicate_audit(table):
    fig, ax = plt.subplots(figsize=(12, 3.8), layout="constrained")
    positions = np.arange(len(table))
    ax.barh(positions, table["pairs"], color=["#198D91", "#9BAEC8", "#EF8354"])
    ax.set_yticks(positions, table["outcome"])
    total = table["pairs"].sum()
    for position, row in table.iterrows():
        label = f"{int(row['pairs']):,}" + (f" ({row['share']:.1%})" if total else "")
        ax.annotate(label, (row["pairs"], position), xytext=(5, 0), textcoords="offset points",
                    ha="left", va="center")
    ax.set_xlim(0, max(1, table["pairs"].max()) * 1.25)
    ax.set_xlabel("пар-кандидатов; одна строка может участвовать в нескольких парах")
    ax.set_title("pHash даёт кандидатов, объединение требует проверки пикселей")
    ax.invert_yaxis()
    return fig


def lineage_table(frame):
    columns = ["source_group_basis", "lineage_status", "processing_records", "source_groups",
               "known_identity_records", "generation_batches_audit_only"]
    if "dataset" not in frame:
        return pd.DataFrame(columns=columns)
    data = frame.loc[frame["dataset"].astype(str).str.lower().eq("df40")].copy()
    if data.empty:
        return pd.DataFrame(columns=columns)
    for column in ("source_group_basis", "lineage_status"):
        if column not in data:
            data[column] = "unknown"
        data[column] = data[column].fillna("unknown").replace("", "unknown")
    for column in ("source_group", "identity_group", "generation_batch"):
        if column not in data:
            data[column] = ""
        data[column] = data[column].fillna("").astype(str)
    rows = []
    for (basis, status), group in data.groupby(["source_group_basis", "lineage_status"], dropna=False):
        rows.append({
            "source_group_basis": basis, "lineage_status": status, "processing_records": len(group),
            "source_groups": group.loc[group["source_group"].ne(""), "source_group"].nunique(),
            "known_identity_records": int(group["identity_group"].ne("").sum()),
            "generation_batches_audit_only": group.loc[group["generation_batch"].ne(""), "generation_batch"].nunique(),
        })
    return pd.DataFrame(rows, columns=columns).sort_values("processing_records", ascending=False)


def _sha256(path):
    import hashlib

    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pixel_illustrations(root, faces, metadata_path):
    cache = root / "data/eda_cache/df40"
    paths = faces.loc[faces["eligible"].fillna(False), ["input_path", "crop_path"]]
    pixels_available = len(paths) > 0 and all(
        value is not None and Path(str(value)).is_file() for value in paths.to_numpy().ravel()
    )
    if pixels_available:
        cache.mkdir(parents=True, exist_ok=True)
        quality = quality_sample(faces, maximum=3000, seed=42)
        figures = {
            "examples": plot_examples(faces, seed=42),
            "image_properties": plot_image_properties(faces, quality),
            "source_files": plot_source_files(quality),
        }
        artifacts = {}
        for name, figure in figures.items():
            path = cache / f"{name}.png"
            figure.savefig(path, dpi=140)
            plt.close(figure)
            artifacts[name] = {"path": path.name, "sha256": _sha256(path)}
        summary = quality.groupby("target").agg(
            лиц=("sample_id", "size"), чёткость_медиана=("sharpness_laplacian", "median"),
            HF_медиана=("hf_ratio", "median"), размер_KiB_медиана=("source_kib", "median"),
        ).rename(index={0: "real", 1: "fake"})
        summary_path = cache / "quality_summary.html"
        summary_path.write_text(summary.to_html())
        artifacts["quality_summary"] = {"path": summary_path.name, "sha256": _sha256(summary_path)}
        notebook = root / "notebooks/eda_df40.ipynb"
        manifest = {
            "schema_version": 1, "scope": "preprocessing_pixel_illustrations_only",
            "origin": "computed_from_real_pixels", "metadata_path": str(Path(metadata_path).relative_to(root)),
            "metadata_sha256": _sha256(metadata_path),
            "source_notebook_path": "notebooks/eda_df40.ipynb",
            "source_notebook_sha256": _sha256(notebook) if notebook.is_file() else None,
            "source_commit": None, "quality_sample_size": len(quality), "seed": 42,
            "artifacts": artifacts,
        }
        (cache / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        mode = "fresh_pixels"
    else:
        manifest_path = cache / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError("Нет raw/crop изображений и проверенного pixel-illustration cache")
        manifest = json.loads(manifest_path.read_text())
        mode = "historical_pixel_artifacts"
    if manifest.get("schema_version") != 1 or manifest.get("scope") != "preprocessing_pixel_illustrations_only":
        raise ValueError("Неизвестный формат pixel-illustration cache")
    if manifest["metadata_sha256"] != _sha256(metadata_path):
        raise ValueError("Pixel-illustration cache относится к другой версии metadata")
    source_commit = manifest.get("source_commit")
    if source_commit:
        import hashlib
        import subprocess

        snapshot_path = manifest.get("source_notebook_snapshot")
        snapshot = root / snapshot_path if snapshot_path else None
        if snapshot is not None and snapshot.is_file():
            saved = snapshot.read_bytes()
        else:
            try:
                saved = subprocess.run(
                    ["git", "show", f"{source_commit}:{manifest['source_notebook_path']}"],
                    cwd=root, check=True, capture_output=True,
                ).stdout
            except (OSError, subprocess.CalledProcessError) as error:
                raise ValueError("Нет проверяемого source notebook snapshot или исходного git blob") from error
        if hashlib.sha256(saved).hexdigest() != manifest["source_notebook_sha256"]:
            raise ValueError("SHA исходного выполненного notebook не совпадает с provenance")
    checked = {}
    for name, artifact in manifest["artifacts"].items():
        path = cache / artifact["path"]
        if _sha256(path) != artifact["sha256"]:
            raise ValueError(f"Повреждён pixel-illustration artifact: {name}")
        checked[name] = path
    if not {"examples", "image_properties", "source_files", "quality_summary"} <= checked.keys():
        raise ValueError("Неполный pixel-illustration cache")
    return {
        "mode": mode, "manifest": manifest,
        "figure_paths": {name: checked[name] for name in ("examples", "image_properties", "source_files")},
        "quality_summary_html": checked["quality_summary"].read_text(),
    }


def global_split_divergences(parts):
    features = ("dataset", "target", "generator", "manipulation", "domain", "source_format",
                "yaw_bin", "pitch_bin", "roll_bin", "light_bin", "face_size_bin",
                "padding_bin", "source_resolution_bin", "face_mask_fraction_of_crop_bin")
    rows = []
    for first, second in (("train", "validate"), ("train", "test"), ("validate", "test")):
        for feature in features:
            if feature not in parts[first] or feature not in parts[second]:
                continue
            left = parts[first][feature].fillna("unknown").astype(str).value_counts()
            right = parts[second][feature].fillna("unknown").astype(str).value_counts()
            rows.append({"pair": f"{first} / {second}", "feature": feature,
                         **divergence(left, right)})
    return pd.DataFrame(rows, columns=["pair", "feature", "TV", "JS_bits", "max_gap_pp"])


def plot_global_marginals(parts):
    features = ("dataset", "target", "source_format")
    fig, axes = plt.subplots(1, 3, figsize=(17, 5), layout="constrained")
    for ax, feature in zip(axes, features):
        values = {
            name: frame[feature].fillna("unknown").astype(str).value_counts(normalize=True)
            for name, frame in parts.items() if name in ("train", "validate", "test")
        }
        table = pd.DataFrame(values).fillna(0)
        table.plot.bar(ax=ax, color=[COLORS[name] for name in table], rot=25)
        ax.set_title(feature)
        ax.set_xlabel("")
        ax.set_ylabel("доля внутри части")
        ax.legend(title="", frameon=False)
    fig.suptitle("Глобальные доли источников, классов и исходных форматов", fontsize=16)
    return fig


def balance_before_after(baseline, updated):
    rows = []
    for name, parts in (("author_baseline", baseline), ("global_balanced", updated)):
        global_table = global_split_divergences(parts)
        conditional = split_divergences(parts)
        for scope, table in (("global", global_table), ("conditional", conditional)):
            for feature, values in table.groupby("feature"):
                measured = values["TV"].dropna()
                rows.append({
                    "version": name, "scope": scope, "feature": feature,
                    "comparisons": len(measured),
                    "TV_mean": measured.mean(), "TV_max": measured.max(),
                    "max_gap_pp": values["max_gap_pp"].max(),
                })
    return pd.DataFrame(rows)


def plot_balance_before_after(table):
    features = ("dataset", "target", "generator", "domain", "source_format",
                "face_size_bin", "source_resolution_bin", "yaw_bin", "light_bin",
                "face_mask_fraction_of_crop_bin")
    fig, axes = plt.subplots(1, 2, figsize=(17, 8), layout="constrained")
    for ax, scope, title in zip(axes, ("global", "conditional"),
                                ("Глобальные маргинальные доли", "Условные распределения")):
        subset = table.loc[table["scope"].eq(scope) & table["feature"].isin(features)]
        pivot = subset.pivot(index="feature", columns="version", values="TV_max")
        pivot = pivot.reindex([feature for feature in features if feature in pivot.index])
        positions = np.arange(len(pivot))
        for version, color, label, marker in (
                ("author_baseline", "#A3A9B5", "авторские части", "s"),
                ("global_balanced", COLORS["train"], "global 70/15/15", "o")):
            if version in pivot:
                ax.scatter(pivot[version], positions, s=70, color=color, label=label, marker=marker)
        if {"author_baseline", "global_balanced"} <= set(pivot):
            for position, row in enumerate(pivot.to_dict("records")):
                if pd.notna(row["author_baseline"]) and pd.notna(row["global_balanced"]):
                    ax.plot([row["author_baseline"], row["global_balanced"]], [position, position],
                            color="#CBD5E1", zorder=0)
        ax.set_yticks(positions, pivot.index)
        ax.set_xlabel("максимальный TV между частями")
        ax.set_title(title)
        ax.legend(frameon=False)
    fig.suptitle("Как новый split изменил расхождения распределений", fontsize=16)
    return fig
