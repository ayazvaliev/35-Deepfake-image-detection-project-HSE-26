import numpy as np
import pandas as pd

from src.notebook_src.of_dedup import iou


def match_detector(faces, detections, min_iou=0.2):
    groups = dict(tuple(detections.groupby(["split", "image_id"], sort=False)))
    rows = []
    for index, face in faces.iterrows():
        group = groups.get((face["split"], face["image_id"]))
        row = {"det_idx": -1, "det_x": np.nan, "det_y": np.nan,
               "det_w": np.nan, "det_h": np.nan, "det_score": np.nan}
        if group is not None:
            overlaps = [iou(face[["x", "y", "w", "h"]], candidate)
                        for candidate in group[["det_x", "det_y", "det_w", "det_h"]].to_numpy()]
            best = int(np.argmax(overlaps))
            if overlaps[best] >= min_iou:
                found = group.iloc[best]
                row = {"det_idx": int(group.index[best]), **found[["det_x", "det_y", "det_w", "det_h", "det_score"]].to_dict()}
        rows.append(row)
    return pd.DataFrame(rows, index=faces.index)
