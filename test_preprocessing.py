import os

import cv2

from preprocessing import load_image, preprocess_image


# Root folder containing all dataset material folders.
# This lets you process every image in the dataset without editing the path each time.
DATASET_ROOT = os.path.join("datasets")
OUTPUT_ROOT = os.path.join("outputs", "preprocessing_preview")

# Supported image extensions.
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


# Use os.walk() so every image under the dataset tree is processed automatically.
image_paths = []
for root, _, files in os.walk(DATASET_ROOT):
    for filename in files:
        if os.path.splitext(filename)[1].lower() in IMAGE_EXTENSIONS:
            image_paths.append(os.path.join(root, filename))

if not image_paths:
    raise FileNotFoundError(f"No image files found in dataset folder: {DATASET_ROOT}")

print(f"Found {len(image_paths)} images to preprocess.")

for index, image_path in enumerate(image_paths, start=1):
    try:
        image = load_image(image_path)
        processed = preprocess_image(image)

        # Create a folder for each image so the outputs stay organised.
        relative_path = os.path.relpath(image_path, DATASET_ROOT)
        image_output_dir = os.path.join(OUTPUT_ROOT, os.path.dirname(relative_path))
        os.makedirs(image_output_dir, exist_ok=True)

        base_name = os.path.splitext(os.path.basename(relative_path))[0]

        cv2.imwrite(os.path.join(image_output_dir, f"{base_name}_original.jpg"), processed["original"])
        cv2.imwrite(os.path.join(image_output_dir, f"{base_name}_denoised.jpg"), processed["denoised"])
        cv2.imwrite(os.path.join(image_output_dir, f"{base_name}_gray.jpg"), processed["gray"])
        cv2.imwrite(os.path.join(image_output_dir, f"{base_name}_gray_enhanced.jpg"), processed["gray_enhanced"])
        cv2.imwrite(os.path.join(image_output_dir, f"{base_name}_hsv.jpg"), cv2.cvtColor(processed["hsv"], cv2.COLOR_BGR2RGB))
        cv2.imwrite(os.path.join(image_output_dir, f"{base_name}_lab.jpg"), cv2.cvtColor(processed["lab"], cv2.COLOR_BGR2RGB))

        print(f"[{index}/{len(image_paths)}] Saved previews for: {image_path}")
    except Exception as exc:
        print(f"[{index}/{len(image_paths)}] Failed for: {image_path}")
        print("Reason:", exc)

print("All preprocessing preview tasks completed.")