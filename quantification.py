import numpy as np
import glob
import tifffile as tiff
from PIL import Image

def get_cells(frame):
    ids = np.unique(frame)
    ids = ids[ids != 0] # 0 = background

    return {i: (frame == i) for i in ids} # bool mask

def iou(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()

    if union <= 0:
        return 0
    return inter / union # jaccard

def match_cells(cells_t, cells_t_next, threshold=0.3):
    matches = {}

    for id_t, mask_t in cells_t.items():
        best_score = 0
        best_id = None

        for id_t_next, mask_t_next in cells_t_next.items():
            iou_score = iou(mask_t, mask_t_next)

            if iou_score > best_score:
                best_score = iou_score
                best_id = id_t_next

        if best_score > threshold:
            matches[id_t] = best_id

    return matches

def create_overlay(frame_t, frame_t_next, threshold=0.05):
    cells_t = get_cells(frame_t)
    cells_t_next = get_cells(frame_t_next)

    matches = match_cells(cells_t, cells_t_next)

    overlay = np.zeros((*frame_t.shape, 4), dtype=np.uint8)

    for id_t, id_t_next in matches.items():
        shape_0 = cells_t[id_t]
        shape_1 = cells_t_next[id_t_next]

        expansion = np.logical_and(shape_1, ~shape_0)
        contraction = np.logical_and(shape_0, ~shape_1)

        size = shape_0.sum()

        if size == 0:
            continue

        expansion_ratio = expansion.sum() / size
        contraction_ratio = contraction.sum() / size

        if expansion_ratio > threshold:
            overlay[expansion] = [0, 255, 0, 255]

        if contraction_ratio > threshold:
            overlay[contraction] = [255, 0, 0, 255]

    return overlay

image_files = sorted(glob.glob("*_cp_masks.png"))
mask_files = sorted(glob.glob("*.npy"))

results = []

for i in range(len(mask_files) - 1):
    masks_t = np.load(mask_files[i], allow_pickle=True).item()["masks"]
    masks_t1 = np.load(mask_files[i+1], allow_pickle=True).item()["masks"]

    img = Image.open(image_files[i+1])

    overlay = create_overlay(img, masks_t, masks_t1)
    results.append(overlay)

results = np.stack(results)

for i, frame in enumerate(results):
    Image.fromarray(frame).save(f"overlay_{i:04d}.png")