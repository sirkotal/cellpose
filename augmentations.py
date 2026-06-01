import os
from typing import List, Union
import albumentations as A
import cv2
import numpy as np
import tifffile as tiff

def augment_data(input_items, input_masks, output_flip, output_app, flip_p=1.0, app_p=1.0):
    transform_flip = A.HorizontalFlip(p=flip_p)
    transform_app = A.Compose([
        A.ElasticTransform(alpha=800,sigma=7,border_mode=4,p=1.0), # 420
        A.RandomBrightnessContrast(brightness_range=(0, 0.1), contrast_range=(0, 0.1), p=app_p),
        A.GaussianBlur(blur_range=(2, 7), p=app_p),
    ])

    os.makedirs(output_flip, exist_ok=True)
    os.makedirs(output_app, exist_ok=True)

    frame = 0
    mask_idx = 0

    for idx, item in enumerate(input_items):
        if frame % 11 == 0 and frame != 0:
            frame = 0
            mask_idx += 1

        if isinstance(item, str):
            img = cv2.imread(item, cv2.IMREAD_UNCHANGED)
            base = os.path.splitext(os.path.basename(item))[0]
        elif isinstance(item, np.ndarray):
            img = item
            base = f"img_{idx:04d}"
        else:
            raise ValueError("must be file paths or numpy arrays")
        
        mask = tiff.imread(input_masks[mask_idx])[frame].astype(np.uint16)

        flipped = transform_flip(image=img, mask=mask)
        os.makedirs(os.path.dirname(os.path.join(output_flip, f"{base}_flip.png")), exist_ok=True)
        cv2.imwrite(os.path.join(output_flip, f"{base}_flip.png"), flipped["image"])
        cv2.imwrite(os.path.join(output_flip, f"{base}_flip_mask.png"), flipped["mask"])

        app = transform_app(image=img, mask=mask)
        os.makedirs(os.path.dirname(os.path.join(output_app, f"{base}_app.png")), exist_ok=True)
        cv2.imwrite(os.path.join(output_app, f"{base}_app.png"), app["image"])
        cv2.imwrite(os.path.join(output_app, f"{base}_app_mask.png"), app["mask"])

        frame += 1


if __name__ == "__main__":
    input_dir = "/home/miguel/Desktop/test/"
    files = [os.path.join(input_dir, f) for f in sorted(os.listdir(input_dir)) if f.lower().endswith(('.png'))]
    masks = [os.path.join(input_dir, "masks/", f) for f in sorted(os.listdir(os.path.join(input_dir, "masks/"))) if f.lower().endswith(('.tif'))]
    augment_data(files, masks, "/home/miguel/Desktop/test/flip", "/home/miguel/Desktop/test/app", 1, 1)