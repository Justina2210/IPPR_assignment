import os

import cv2
import numpy as np

# Longest side is resized to this value; aspect ratio is preserved so
# glove geometry isn't distorted.
TARGET_LONG_SIDE = 1000


def load_image(image_path):
    """Load an image from disk as a BGR array; raises ValueError if it can't be read."""
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"Unable to load image: {image_path}")

    return image


def resize_image(image, target_long_side=TARGET_LONG_SIDE):
    """Resize so the longest side equals target_long_side, preserving aspect ratio."""
    if image is None or image.size == 0:
        raise ValueError("Input image is empty or invalid.")

    height, width = image.shape[:2]
    longest_side = max(height, width)

    if longest_side == 0:
        raise ValueError("Image dimensions are invalid.")

    scale = target_long_side / longest_side
    new_width = max(1, int(width * scale))
    new_height = max(1, int(height * scale))

    resized = cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA
    )

    return resized


def denoise_image(image):
    """Mild bilateral filtering - kept light so small defects like spotting/tearing aren't smoothed away."""
    denoised = cv2.bilateralFilter(
        image,
        d=5,
        sigmaColor=35,
        sigmaSpace=35
    )

    return denoised


def convert_to_grayscale(image):
    """Convert BGR to grayscale, for edge/contour/texture analysis."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def convert_to_hsv(image):
    """Convert BGR to HSV; useful for colour-based defects like dirty, stain, spotting."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


def convert_to_lab(image):
    """Convert BGR to LAB; separates brightness from colour, useful for discoloration/stain/dirty."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2LAB)


def enhance_grayscale(gray):
    """CLAHE-enhance grayscale contrast; applied only to the gray copy so colour stays untouched."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def preprocess_image(image):
    """Run the full pipeline: resize -> denoise -> grayscale/HSV/LAB -> CLAHE enhancement."""
    if image is None or image.size == 0:
        raise ValueError("Input image is empty or invalid.")

    resized = resize_image(image)
    denoised = denoise_image(resized)
    gray = convert_to_grayscale(denoised)
    gray_enhanced = enhance_grayscale(gray)
    hsv = convert_to_hsv(denoised)
    lab = convert_to_lab(denoised)

    # Detectors pick whichever representation suits their own defect logic.
    return {
        "original": resized,
        "denoised": denoised,
        "gray": gray,
        "gray_enhanced": gray_enhanced,
        "hsv": hsv,
        "lab": lab,
    }


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_root = os.path.join(base_dir, "datasets")

    candidate_folders = [
        os.path.join(dataset_root, "nitrile"),
        os.path.join(dataset_root, "latex"),
        os.path.join(dataset_root, "cotton"),
    ]

    sample_path = None

    for folder in candidate_folders:
        if not os.path.isdir(folder):
            continue

        for root, _, files in os.walk(folder):
            for file in files:
                name_lower = file.lower()
                if name_lower.endswith((".jpg", ".jpeg", ".png", ".bmp")):
                    sample_path = os.path.join(root, file)
                    break
            if sample_path is not None:
                break
        if sample_path is not None:
            break

    if sample_path is None:
        print("No sample images found in the dataset folders.")
    else:
        try:
            image = load_image(sample_path)
            processed = preprocess_image(image)

            print(f"Loaded image: {sample_path}")
            print(f"Original size: {processed['original'].shape[:2]}")
            print(f"Gray size: {processed['gray'].shape}")
            print(f"HSV size: {processed['hsv'].shape}")
            print(f"LAB size: {processed['lab'].shape}")
            print(f"Enhanced grayscale size: {processed['gray_enhanced'].shape}")
            print("Preprocessing pipeline completed successfully.")
        except Exception as exc:
            print(f"Preprocessing test failed: {exc}")
