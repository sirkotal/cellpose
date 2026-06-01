import os
from typing import List, Union
import albumentations as A
import cv2
import numpy as np

def augment_data(input_items, output_flip, output_app, flip_p=1.0, app_p=1.0):
    transform_flip = A.HorizontalFlip(p=flip_p)
    transform_app = A.Compose([
        A.ElasticTransform(alpha=800,sigma=7,border_mode=4,p=1.0), # 420
        A.RandomBrightnessContrast(brightness_range=(0, 0.1), contrast_range=(0, 0.1), p=app_p),
        A.GaussianBlur(blur_range=(2, 7), p=app_p),
    ])

    os.makedirs(output_flip, exist_ok=True)
    os.makedirs(output_app, exist_ok=True)

    for idx, item in enumerate(input_items):
        if isinstance(item, str):
            img = cv2.imread(item, cv2.IMREAD_UNCHANGED)
            base = os.path.splitext(os.path.basename(item))[0]
        elif isinstance(item, np.ndarray):
            img = item
            base = f"img_{idx:06d}"
        else:
            raise ValueError("must be file paths or numpy arrays")

        flipped = transform_flip(image=img)["image"]
        os.makedirs(os.path.dirname(os.path.join(output_flip, f"{base}_flip.png")), exist_ok=True)
        cv2.imwrite(os.path.join(output_flip, f"{base}_flip.png"), flipped)

        app = transform_app(image=img)["image"]
        os.makedirs(os.path.dirname(os.path.join(output_app, f"{base}_app.png")), exist_ok=True)
        cv2.imwrite(os.path.join(output_app, f"{base}_app.png"), app)


if __name__ == "__main__":
    input_dir = "/home/miguel/Desktop/test/"
    files = [os.path.join(input_dir, f) for f in sorted(os.listdir(input_dir)) if f.lower().endswith(('.png'))]
    augment_data(files, "/home/miguel/Desktop/test/flip", "/home/miguel/Desktop/test/app", 1, 1)