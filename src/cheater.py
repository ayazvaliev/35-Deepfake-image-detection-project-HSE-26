import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.config import SEED
from src.face_stats import build_face_stats
from src.metadata import build_metadata

FEATURE_GROUPS = {
    "metadata": ["file_size"],
    "face": ["face_area", "face_cx", "face_cy", "yaw", "roll", "n_faces"],
    "image": ["brightness", "bg_brightness", "sharpness", "chroma", "is_gray"],
}
FEATURE_GROUPS["all"] = [f for features in FEATURE_GROUPS.values() for f in features]
DEVIATION_FEATURES = ["face_cx_dev", "face_cy_dev", "yaw_abs", "roll_abs"]
MODELS = {
    "logreg": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=SEED)),
    "boosting": lambda: HistGradientBoostingClassifier(random_state=SEED),
}


def cheater_data():
    stats = build_face_stats()
    sizes = build_metadata()[["path", "file_size"]]
    df = stats.merge(sizes, on="path")
    df = df[df["n_faces"] > 0].reset_index(drop=True)
    df["is_gray"] = df["is_gray"].astype(int)
    return df


def add_deviations(df):
    df = df.copy()
    # median from train only, so valid does not leak into the feature
    cy_median = df[df["split"] == "train"].groupby("dataset")["face_cy"].median()
    df["face_cx_dev"] = (df["face_cx"] - 0.5).abs()
    df["face_cy_dev"] = (df["face_cy"] - df["dataset"].map(cy_median)).abs()
    df["yaw_abs"] = df["yaw"].abs()
    df["roll_abs"] = df["roll"].abs()
    return df


def xy(df, features, split):
    part = df[df["split"] == split]
    return part[features], part["label"]


def cheater_auc(df, groups=FEATURE_GROUPS, models=MODELS):
    rows = []
    for ds, group in df.groupby("dataset"):
        for group_name, features in groups.items():
            for model_name, make_model in models.items():
                model = make_model().fit(*xy(group, features, "train"))
                x_valid, y_valid = xy(group, features, "valid")
                auc = roc_auc_score(y_valid, model.predict_proba(x_valid)[:, 1])
                rows.append({"dataset": ds, "group": group_name, "model": model_name, "auc": round(auc, 3)})
    return pd.DataFrame(rows)


def feature_auc(df):
    rows = []
    for ds, group in df.groupby("dataset"):
        x_valid, y_valid = xy(group, FEATURE_GROUPS["all"], "valid")
        for feature in FEATURE_GROUPS["all"]:
            row = {"dataset": ds, "feature": feature}
            if x_valid[feature].nunique() > 1:
                auc = roc_auc_score(y_valid, x_valid[feature])
                row["auc"] = round(max(auc, 1 - auc), 3)
                row["higher"] = "fake" if auc > 0.5 else "real"
            rows.append(row)
    return pd.DataFrame(rows)


def cheater_importance(df, ds):
    group = df[df["dataset"] == ds]
    features = FEATURE_GROUPS["all"]
    model = MODELS["boosting"]().fit(*xy(group, features, "train"))
    x_valid, y_valid = xy(group, features, "valid")
    result = permutation_importance(model, x_valid, y_valid, scoring="roc_auc", n_repeats=10, random_state=SEED)
    importance = pd.DataFrame({"mean": result.importances_mean, "std": result.importances_std}, index=features)
    return importance.sort_values("mean", ascending=False).round(4)
