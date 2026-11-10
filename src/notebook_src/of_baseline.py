import hashlib
import json
import os
import random

import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights, resnet18

from .of_crops import match_detector
from .of_data import CACHE, RESULTS, ROOT, SEED, SPLITS, crop_face, load_all, read_rgb
from .of_metrics import basic


MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
DEGRADATIONS = ["clean", "jpeg50", "down50"]


def device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resize(image, size):
    method = cv2.INTER_AREA if max(image.shape[:2]) > size else cv2.INTER_CUBIC
    return cv2.resize(image, (size, size), interpolation=method)


def make_view(image):
    return resize(np.asarray(image), 224)


def jpeg(image, quality):
    bgr = cv2.cvtColor(np.ascontiguousarray(image), cv2.COLOR_RGB2BGR)
    success, encoded = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not success:
        raise ValueError("JPEG augmentation encoding failed")
    return cv2.cvtColor(cv2.imdecode(encoded, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def color_jitter(image, rng):
    values = image.astype(np.float32) / 255
    values = (values - 0.5) * rng.uniform(0.8, 1.2) + 0.5
    values *= rng.uniform(0.8, 1.2)
    gray = cv2.cvtColor(values, cv2.COLOR_RGB2GRAY)[:, :, None]
    values = gray + (values - gray) * rng.uniform(0.8, 1.2)
    return np.clip(values * 255, 0, 255).astype(np.uint8)


def to_tensor(images):
    values = np.stack(images).astype(np.float32) / 255
    values = (values - MEAN) / STD
    return torch.from_numpy(values.transpose(0, 3, 1, 2).copy())


def resnet(pretrained=True):
    model = resnet18(weights=None)
    if pretrained:
        weights = ROOT / "models" / "resnet18-f37072fd.pth"
        if not weights.exists():
            weights = ROOT / "models" / "resnet18-weights.pth"
        if weights.exists():
            model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
        else:
            model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    return model


def degraded(image, name):
    if name == "jpeg50":
        return resize(jpeg(image, 50), 224)
    if name == "down50":
        return resize(resize(image, 112), 224)
    return make_view(image)


def augment(image, name, rng):
    image = np.asarray(image).copy()
    if name != "none" and rng.random() < 0.5:
        image = image[:, ::-1].copy()
    if name in ["jpeg_scale", "full", "color"]:
        side = int(rng.integers(192, 257))
        top, left = [int(rng.integers(0, 257 - side)) for _ in range(2)]
        image = image[top:top + side, left:left + side]
        if rng.random() < 0.5:
            image = jpeg(image, int(rng.integers(40, 101)))
    if name in ["full", "color"] and rng.random() < 0.5:
        image = cv2.GaussianBlur(image, (0, 0), float(rng.uniform(0.2, 1.5)))
    image = resize(image, 224)
    return color_jitter(image, rng) if name == "color" else image


class FaceDataset(Dataset):
    def __init__(self, crops, indices, targets, augmentation=None, degradation="clean", seed=42):
        self.crops = crops
        self.indices = np.asarray(indices)
        self.targets = targets
        self.augmentation = augmentation
        self.degradation = degradation
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        row = self.indices[index]
        image = self.crops[row]
        view = augment(image, self.augmentation, self.rng) if self.augmentation else degraded(image, self.degradation)
        return to_tensor([view])[0], torch.tensor(self.targets[row], dtype=torch.float32)


def loader(crops, indices, targets, augmentation=None, degradation="clean", seed=42, shuffle=False):
    dataset = FaceDataset(crops, indices, targets, augmentation, degradation, seed)
    batch_size = int(os.environ.get("DEEPFAKE_BASELINE_BATCH", 64))
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, generator=generator)


def prepare_crops(data, limit):
    frame_table, _ = load_all()
    paths = frame_table.set_index(["split", "image_id"])["path"].to_dict()
    detections = pd.concat([pd.read_parquet(CACHE / f"yunet_frames_{split}.parquet").assign(split=split)
                            for split in SPLITS], ignore_index=True)
    selected = data[(data["face_side_px"] >= 32) & ~data["ann_conflict"] & ~data["neighbor_opposite_1.3"]].copy()
    if "leak_person" in selected:
        selected = selected[~selected["leak_person"]]
    selected = selected[selected["scene_n_splits"] == 1]
    selected = selected.join(match_detector(selected, detections))
    selected = selected[(selected["det_idx"] >= 0) & (selected[["det_w", "det_h"]].min(axis=1) >= 32)]
    parts = []
    for split, share in [("Train", 0.4), ("Val", 0.2), ("Test-Dev", 0.4)]:
        candidates = selected[selected["split"] == split]
        quotas = int(limit * share / 2)
        for label in ["real", "fake"]:
            group = candidates[candidates["label"] == label]
            parts.append(group.sample(min(quotas, len(group)), random_state=SEED))
    index = pd.concat(parts, ignore_index=True)
    index["part"] = index["split"].map({"Train": "fit", "Val": "val", "Test-Dev": "test"})
    train_rows = index.index[index["part"] == "fit"]
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED)
    _, holdout = next(splitter.split(index.loc[train_rows], groups=index.loc[train_rows, "scene_id"]))
    index.loc[train_rows[holdout], "part"] = "holdout"
    if index.groupby("source_id")["part"].nunique().max() > 1:
        raise AssertionError("Source image leakage in OpenForensics baseline")
    index.to_parquet(CACHE / "crops_index.parquet", index=False)
    arrays = {kind: np.lib.format.open_memmap(CACHE / f"crops_{kind}.npy", mode="w+", dtype=np.uint8,
                                             shape=(len(index), 256, 256, 3)) for kind in ["det", "ann"]}
    for key, group in index.groupby(["split", "image_id"], sort=False):
        image = read_rgb(paths[key])
        for row, face in group.iterrows():
            for kind, columns in [("ann", ["x", "y", "w", "h"]), ("det", ["det_x", "det_y", "det_w", "det_h"])]:
                arrays[kind][row] = crop_face(image, *face[columns].to_numpy(float), out=256)
    for array in arrays.values():
        array.flush()
    return index


def fft_features(crops, degradation):
    yy, xx = np.indices((224, 224))
    radius = np.sqrt((xx - 112) ** 2 + (yy - 112) ** 2)
    bins = np.clip((radius / (112 * np.sqrt(2)) * 64).astype(int), 0, 63)
    counts = np.bincount(bins.ravel(), minlength=64)
    result = np.empty((len(crops), 64), np.float32)
    for row, crop in enumerate(crops):
        image = degraded(crop, degradation)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
        spectrum = np.abs(np.fft.fftshift(np.fft.fft2(gray - gray.mean()))) ** 2
        radial = np.bincount(bins.ravel(), weights=spectrum.ravel(), minlength=64) / counts
        result[row] = np.log1p(radial)
    return result


def feature_embeddings(crops, targets):
    model = resnet().to(device()).eval()
    result = np.empty((len(crops), 960), np.float32)
    start = 0
    with torch.inference_mode():
        for batch, _ in loader(crops, np.arange(len(crops)), targets):
            value = batch.to(device())
            value = model.maxpool(model.relu(model.bn1(model.conv1(value))))
            features = []
            for layer in [model.layer1, model.layer2, model.layer3, model.layer4]:
                value = layer(value)
                features.append(value.mean(dim=(2, 3)))
            block = torch.cat(features, dim=1).cpu().numpy()
            result[start:start + len(block)] = block
            start += len(block)
    return result


def save_scores(name, seed, degradation, crop, indices, probabilities):
    pd.DataFrame({"row": indices, "score": probabilities, "model": name,
                  "seed": seed, "degrade": degradation, "eval_crop": crop}).to_parquet(
        RESULTS / "baseline" / f"scores_{name}_{seed}_{degradation}_{crop}.parquet", index=False)


def linear_baselines(index, crops):
    targets = index["y_fake"].to_numpy()
    fit = index["part"].eq("fit").to_numpy()
    evaluation = index.index[index["part"].isin(["val", "test"])].to_numpy()
    for kind, images in crops.items():
        features = fft_features(images, "clean")
        np.save(CACHE / f"fft_{kind}_clean.npy", features)
        model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000, random_state=SEED))
        model.fit(features[fit], targets[fit])
        for degradation in DEGRADATIONS:
            values = features[evaluation] if degradation == "clean" else fft_features(images[evaluation], degradation)
            save_scores(f"fft_lr_{kind}", SEED, degradation, kind, evaluation, model.predict_proba(values)[:, 1])
        embeddings = feature_embeddings(images, targets)
        np.save(CACHE / f"probe_{kind}.npy", embeddings)
        boundaries = [0, 64, 192, 448, 960]
        for layer in range(4):
            features = embeddings[:, boundaries[layer]:boundaries[layer + 1]]
            model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000, random_state=SEED))
            model.fit(features[fit], targets[fit])
            save_scores(f"probe_layer{layer + 1}_{kind}", SEED, "clean", kind, evaluation,
                        model.predict_proba(features[evaluation])[:, 1])


def predict(model, batches, embeddings=False):
    scores, vectors = [], []
    model.eval()
    with torch.inference_mode():
        for batch, _ in batches:
            value = batch.to(device())
            if embeddings:
                value = model.maxpool(model.relu(model.bn1(model.conv1(value))))
                for layer in [model.layer1, model.layer2, model.layer3, model.layer4]:
                    value = layer(value)
                vector = value.mean(dim=(2, 3))
                vectors.append(vector.cpu().numpy())
                logits = model.fc(vector).squeeze(1)
            else:
                logits = model(value).squeeze(1)
            scores.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(scores), np.concatenate(vectors) if embeddings else None


def finetune(index, crops, kind, augmentation, seed, epochs):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    targets = index["y_fake"].to_numpy()
    fit = index.index[index["part"] == "fit"].to_numpy()
    holdout = index.index[index["part"] == "holdout"].to_numpy()
    evaluation = index.index[index["part"].isin(["val", "test"])].to_numpy()
    model = resnet()
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.layer4.parameters():
        parameter.requires_grad = True
    model.fc = torch.nn.Linear(512, 1)
    model.to(device())
    optimizer = torch.optim.AdamW([parameter for parameter in model.parameters() if parameter.requires_grad], lr=1e-4, weight_decay=1e-4)
    criterion = torch.nn.BCEWithLogitsLoss()
    train_batches = loader(crops[kind], fit, targets, augmentation=augmentation, seed=seed, shuffle=True)
    holdout_batches = loader(crops[kind], holdout, targets)
    best, best_state, logs = -np.inf, None, []
    name = f"{kind}_{augmentation}"
    for epoch in range(epochs):
        model.eval()
        model.layer4.train()
        model.fc.train()
        losses = []
        for batch, target in train_batches:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch.to(device())).squeeze(1)
            loss = criterion(logits, target.to(device()))
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        probabilities, _ = predict(model, holdout_batches)
        auc = basic(targets[holdout], probabilities)["auc"]
        logs.append({"model": f"ft_{name}", "seed": seed, "epoch": epoch + 1,
                     "train_loss": float(np.mean(losses)), "holdout_auc": auc})
        if auc > best:
            best = auc
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        print(f"OF baseline {name} seed={seed} epoch={epoch + 1}/{epochs} holdout_auc={auc:.4f}", flush=True)
    model.load_state_dict(best_state)
    torch.save(best_state, CACHE / f"ft_{name}_{seed}.pt")
    pd.DataFrame(logs).to_parquet(RESULTS / "baseline" / f"log_ft_{name}_{seed}.parquet", index=False)
    for evaluation_kind in ["det", "ann"] if augmentation == "flip" else [kind]:
        for degradation in DEGRADATIONS:
            batches = loader(crops[evaluation_kind], evaluation, targets, degradation=degradation)
            need_embeddings = name == "det_flip" and seed == 42 and degradation == "clean" and evaluation_kind == "det"
            probabilities, vectors = predict(model, batches, need_embeddings)
            save_scores(f"ft_{name}", seed, degradation, evaluation_kind, evaluation, probabilities)
            if need_embeddings:
                np.save(CACHE / "emb_ft_det_flip_42.npy", vectors)


def ensure_baseline(data):
    epochs = int(os.environ.get("DEEPFAKE_BASELINE_EPOCHS", 6))
    limit = int(os.environ.get("DEEPFAKE_OF_MAX_FACES", 50000))
    if epochs < 1 or limit < 100:
        raise ValueError("Baseline epochs must be >=1 and face limit must be >=100")
    signature = hashlib.sha256(pd.util.hash_pandas_object(data[["split", "face_id", "source_id", "label"]], index=False).values.tobytes()).hexdigest()
    config = {"schema": 1, "epochs": epochs, "max_faces": limit, "source_sha256": signature,
              "crop_scale": 1.3, "crop_size": 256, "seeds": [42, 43]}
    marker = CACHE / "baseline_complete.json"
    if marker.exists() and json.loads(marker.read_text()) == config:
        return
    (RESULTS / "baseline").mkdir(parents=True, exist_ok=True)
    print(f"Preparing original OpenForensics baseline: {config}, device={device()}", flush=True)
    index = prepare_crops(data, limit)
    crops = {kind: np.load(CACHE / f"crops_{kind}.npy", mmap_mode="r") for kind in ["det", "ann"]}
    linear_baselines(index, crops)
    for seed in [42, 43]:
        for kind, augmentation in [("det", "none"), ("det", "flip"), ("ann", "flip"),
                                   ("det", "jpeg_scale"), ("det", "full"), ("det", "color")]:
            finetune(index, crops, kind, augmentation, seed, epochs)
    marker.write_text(json.dumps(config, indent=2) + "\n")
