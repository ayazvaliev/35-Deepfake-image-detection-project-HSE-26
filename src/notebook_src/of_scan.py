import hashlib
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from src.notebook_src.of_data import CACHE, SPLITS, crop_face, dhash, load_all, mark_cache, read_rgb
from src.notebook_src.metadata import jpeg_quality


def scan(split):
    frames, faces = load_all()
    frames = frames[frames["split"] == split]
    faces = faces[faces["split"] == split]
    face_groups = dict(tuple(faces.groupby("image_id", sort=False)))
    frame_rows, face_rows, thumbs, face_thumbs = [], [], [], []
    for frame in frames.to_dict("records"):
        path = frame["path"]
        image = read_rgb(path)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        with Image.open(path) as source:
            source.load()
            exif = source.getexif()
            quantization = getattr(source, "quantization", None)
            quality, error = jpeg_quality(quantization) if quantization else (None, np.nan)
            row = {"image_id": frame["image_id"], "exists": True,
                   "w_file": source.width, "h_file": source.height, "mode": source.mode,
                   "n_exif_tags": len(exif), "exif_orient": exif.get(274, 1),
                   "jpeg_q": quality, "jpeg_std_table": bool(error <= 1)}
        with open(path, "rb") as stream:
            content = stream.read()
        row.update(bytes=len(content), md5=hashlib.md5(content).hexdigest(), dhash=dhash(gray))
        frame_rows.append(row)
        thumbs.append(cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA))
        for face in face_groups[frame["image_id"]].to_dict("records"):
            crop = crop_face(gray, face["x"], face["y"], face["w"], face["h"], out=32)
            face_rows.append({"face_id": face["face_id"], "face_dhash": dhash(crop)})
            face_thumbs.append(crop)
    pd.DataFrame(frame_rows).to_parquet(CACHE / f"scan_frames_{split}.parquet", index=False)
    face_table = pd.DataFrame(face_rows).set_index("face_id").loc[faces["face_id"]].reset_index()
    order = {row["face_id"]: index for index, row in enumerate(face_rows)}
    pd.DataFrame(face_table).to_parquet(CACHE / f"scan_faces_{split}.parquet", index=False)
    np.save(CACHE / f"thumbs_{split}.npy", np.asarray(thumbs, dtype=np.uint8))
    np.save(CACHE / f"face_thumbs_{split}.npy", np.asarray([face_thumbs[order[value]] for value in faces["face_id"]], dtype=np.uint8))
    mark_cache("scan_frames", split, frames, faces)
    print(f"Scanned {split}: {len(frame_rows)} frames, {len(face_rows)} faces", flush=True)


if __name__ == "__main__":
    for selected in sys.argv[1:] or SPLITS:
        scan(selected)
