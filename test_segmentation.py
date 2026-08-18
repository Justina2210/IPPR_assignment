import os

import cv2

from preprocessing import load_image, preprocess_image
from segmentation import segment_glove


DATASET_ROOT = os.path.join("dataset")
OUTPUT_ROOT = os.path.join("outputs", "segmentation_preview")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

image_paths = []
for root, _, files in os.walk(DATASET_ROOT):
    for filename in files:
        if os.path.splitext(filename)[1].lower() in IMAGE_EXTENSIONS:
            image_paths.append(os.path.join(root, filename))

if not image_paths:
    raise FileNotFoundError(f"No image files found in dataset folder: {DATASET_ROOT}")

print(f"Found {len(image_paths)} images for segmentation testing.")

for index, image_path in enumerate(image_paths, start=1):
    try:
        image = load_image(image_path)
        processed = preprocess_image(image)
        result = segment_glove(processed)

        relative_path = os.path.relpath(image_path, DATASET_ROOT)
        image_output_dir = os.path.join(OUTPUT_ROOT, os.path.dirname(relative_path))
        os.makedirs(image_output_dir, exist_ok=True)

        base_name = os.path.splitext(os.path.basename(relative_path))[0]

        with open(os.path.join(image_output_dir, f"{base_name}_segmentation_result.txt"), "w", encoding="utf-8") as f:
            f.write(f"glove_area={result['glove_area']}\n")
            f.write(f"cuff_detected={result['cuff_detected']}\n")
            f.write(f"cuff_y={result['cuff_y']}\n")

        cv2.imwrite(os.path.join(image_output_dir, f"{base_name}_raw_mask.png"), result["raw_mask"])
        cv2.imwrite(os.path.join(image_output_dir, f"{base_name}_glove_mask.png"), result["glove_mask"])

        print(f"[{index}/{len(image_paths)}] Saved segmentation for: {image_path}")
        print("  glove_area:", result["glove_area"])
        print("  cuff_detected:", result["cuff_detected"])

    except Exception as exc:
        print(f"[{index}/{len(image_paths)}] Failed for: {image_path}")
        print("Reason:", exc)

print("All segmentation batch test tasks completed.")
