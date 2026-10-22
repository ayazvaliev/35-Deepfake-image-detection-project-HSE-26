import pandas as pd

from src.config import PROCESSED_DIR, RAW_DIR, SEED

LABELS = {"real": 0, "fake": 1}
SPLITS_140K = ["train", "valid", "test"]
SPLITS_OF = {"Train": "train", "Validation": "valid", "Test": "test"}
DIR_140K = "140k/real_vs_fake/real-vs-fake"


def index_140k():
    frames = []
    for split in SPLITS_140K:
        df = pd.read_csv(RAW_DIR / "140k" / f"{split}.csv")
        df["split"] = split
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)

    folder = df["path"].str.split("/").str[1]
    label = folder.map(LABELS)
    # in the source CSV label is inverted: 1 = real, 0 = fake
    mismatches = {
        "label_str": int((folder != df["label_str"]).sum()),
        "label": int((label != 1 - df["label"]).sum()),
    }

    index = pd.DataFrame({
        "path": DIR_140K + "/" + df["path"],
        "dataset": "140k",
        "split": df["split"],
        "label": label,
    })
    return index, mismatches


def index_openforensics(n=10_000):
    root = RAW_DIR / "openforensics" / "Dataset"
    files = {
        (split, cls): sorted(p.relative_to(RAW_DIR).as_posix() for p in (root / split / cls).glob("*.jpg"))
        for split in SPLITS_OF
        for cls in ["Real", "Fake"]
    }
    total = sum(len(paths) for paths in files.values())

    frames = []
    for (split, cls), paths in files.items():
        split_total = len(files[(split, "Real")]) + len(files[(split, "Fake")])
        n_class = round(n * split_total / total / 2)
        sample = pd.Series(paths).sample(n_class, random_state=SEED)
        frames.append(pd.DataFrame({
            "path": sample.values,
            "dataset": "openforensics",
            "split": SPLITS_OF[split],
            "label": LABELS[cls.lower()],
        }))
    return pd.concat(frames, ignore_index=True)


def build_index():
    index_140, mismatches = index_140k()
    index = pd.concat([index_140, index_openforensics()], ignore_index=True)
    index.to_csv(PROCESSED_DIR / "index.csv", index=False)
    return index, mismatches
