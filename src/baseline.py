import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.cheater import FEATURE_GROUPS, MODELS
from src.config import PROCESSED_DIR, SEED
from src.face_stats import build_face_stats

C_GRID = np.logspace(-2, 2, 9)
POSE_COLUMNS = ["n_faces", "face_area", "face_cx", "face_cy", "yaw", "roll"]


def train_sample(per_class=10_000):
    index = pd.read_csv(PROCESSED_DIR / "index.csv")
    groups = pd.read_csv(PROCESSED_DIR / "dup_groups.csv")
    train = index[(index["dataset"] == "140k") & (index["split"] == "train")].merge(groups, on="path")
    # one random image per group of near-copies
    one_per_group = train.sample(frac=1, random_state=SEED).drop_duplicates("group_id")
    return one_per_group.groupby("label").sample(per_class, random_state=SEED).reset_index(drop=True)


def add_pose(sample):
    known = build_face_stats()[["path"] + POSE_COLUMNS]
    missing = sample[~sample["path"].isin(known["path"])][["path", "dataset", "split", "label"]]
    extra = build_face_stats(missing, "face_stats_train")[["path"] + POSE_COLUMNS]
    pose = sample.merge(pd.concat([known, extra]), on="path", how="left")
    return pose[pose["n_faces"] > 0].reset_index(drop=True)


def add_bins(df):
    df = df.copy()
    df["yaw_bin"] = pd.cut(df["yaw"].abs(), [0, 5, 15, np.inf], right=False)
    df["roll_bin"] = pd.cut(df["roll"].abs(), [0, 2, np.inf], right=False)
    df["area_bin"] = pd.qcut(df["face_area"], 3)
    return df


def balance(df, bins):
    parts = []
    for _, cell in df.groupby(bins, observed=True):
        n = cell["label"].value_counts().reindex([0, 1], fill_value=0).min()
        if n:
            parts.append(cell.groupby("label").sample(n, random_state=SEED))
    return pd.concat(parts, ignore_index=True)


def cv_cheater(df):
    folds = StratifiedKFold(5, shuffle=True, random_state=SEED)
    features = FEATURE_GROUPS["face"]
    return {
        name: round(float(cross_val_score(make(), df[features], df["label"], cv=folds, scoring="roc_auc").mean()), 3)
        for name, make in MODELS.items()
    }


def make_baseline(c):
    return make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=5000, random_state=SEED))


def fit_baseline(x_train, y_train, x_valid, y_valid):
    rows = []
    for c in C_GRID:
        model = make_baseline(c).fit(x_train, y_train)
        rows.append({"C": c, "valid_auc": roc_auc_score(y_valid, model.predict_proba(x_valid)[:, 1])})
    grid = pd.DataFrame(rows)
    best_c = grid.loc[grid["valid_auc"].idxmax(), "C"]
    model = make_baseline(best_c).fit(x_train, y_train)
    valid_score = model.predict_proba(x_valid)[:, 1]
    # threshold at 1% FPR on clean valid
    threshold = np.quantile(valid_score[np.asarray(y_valid) == 0], 0.99)
    return model, grid, threshold


def bootstrap_auc(y, score, n=1000):
    y, score = np.asarray(y), np.asarray(score)
    rng = np.random.default_rng(SEED)
    aucs = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        aucs.append(roc_auc_score(y[i], score[i]))
    return tuple(np.round(np.quantile(aucs, [0.025, 0.975]), 3))


def evaluate(y, score, threshold):
    y, score = np.asarray(y), np.asarray(score)
    fpr, tpr, _ = roc_curve(y, score)
    eer_at = np.argmin(np.abs((1 - tpr) - fpr))
    return {
        "n_real": int((y == 0).sum()), "n_fake": int((y == 1).sum()),
        "roc_auc": round(roc_auc_score(y, score), 3),
        "tpr@fpr1%": round(tpr[fpr <= 0.01].max(), 3),
        "fpr (порог valid)": round((score[y == 0] > threshold).mean(), 4),
        "tpr (порог valid)": round((score[y == 1] > threshold).mean(), 3),
        "eer": round((fpr[eer_at] + 1 - tpr[eer_at]) / 2, 3),
    }
