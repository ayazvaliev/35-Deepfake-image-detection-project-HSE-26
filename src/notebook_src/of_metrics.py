import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve


def basic(target, scores):
    target, scores = np.asarray(target), np.asarray(scores)
    if len(np.unique(target)) < 2:
        return {"auc": np.nan, "tpr@1%": np.nan, "eer": np.nan}
    fpr, tpr, _ = roc_curve(target, scores)
    index = np.argmin(np.abs(fpr - (1 - tpr)))
    return {"auc": float(roc_auc_score(target, scores)),
            "tpr@1%": float(np.interp(0.01, fpr, tpr)),
            "eer": float((fpr[index] + 1 - tpr[index]) / 2)}


def boot_ci(target, scores, groups, n_boot=200, seed=42):
    target, scores, groups = np.asarray(target), np.asarray(scores), np.asarray(groups)
    codes, unique = pd.factorize(groups)
    members = [np.flatnonzero(codes == index) for index in range(len(unique))]
    rng, values = np.random.default_rng(seed), []
    for _ in range(n_boot):
        indices = np.concatenate([members[index] for index in rng.integers(0, len(members), len(members))])
        if np.unique(target[indices]).size == 2:
            values.append(roc_auc_score(target[indices], scores[indices]))
    return np.quantile(values, [0.025, 0.975]) if values else np.array([np.nan, np.nan])


def ece(target, scores, n_bins=10):
    target, scores = np.asarray(target), np.asarray(scores)
    bins = np.minimum((scores * n_bins).astype(int), n_bins - 1)
    return float(sum(np.mean(bins == index) * abs(target[bins == index].mean() - scores[bins == index].mean())
                     for index in range(n_bins) if np.any(bins == index)))


def report(data, by, min_class=200):
    rows = []
    for name, group in data.groupby(by, observed=True, dropna=False):
        n_real, n_fake = int((group["y_fake"] == 0).sum()), int((group["y_fake"] == 1).sum())
        metrics = basic(group["y_fake"], group["score"]) if min(n_real, n_fake) >= min_class else {
            "auc": np.nan, "tpr@1%": np.nan, "eer": np.nan}
        rows.append({"группа": name, "n_real": n_real, "n_fake": n_fake, **metrics})
    return pd.DataFrame(rows)
