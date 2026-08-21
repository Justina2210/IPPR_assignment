"""
preprocessing.py
----------------
Shared preprocessing module for the Glove Defect Detection System.

This module is used by all detectors before segmentation and
per-defect processing. It is intentionally limited to general image
preparation and does not perform any defect detection or background
removal logic.

Responsibilities:
1. Load image from disk
2. Resize while preserving aspect ratio
3. Apply mild noise reduction
4. Generate grayscale image
5. Generate enhanced grayscale image
6. Generate HSV image
7. Generate LAB image

Important:
- No defect detection is performed here.
- No background removal is performed here.
- No cuff detection is performed here.
- No heavy colour normalisation is applied because colour information
  is important for defects such as dirty, stain, spotting and
  discoloration.
"""

import os

import cv2
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

# All images will have their longest side resized to this value.
# Aspect ratio is preserved to avoid distorting glove shape.
TARGET_LONG_SIDE = 1000


# ============================================================
# IMAGE LOADING
# ============================================================

def load_image(image_path):
    """
    Load an image from a file path.

    Parameters
    ----------
    image_path : str
        Path to the image file.

    Returns
    -------
    numpy.ndarray
        Image in OpenCV BGR format.

    Raises
    ------
    ValueError
        If the image cannot be loaded.
    """
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"Unable to load image: {image_path}")

    return image


# ============================================================
# RESIZE
# ============================================================

def resize_image(image, target_long_side=TARGET_LONG_SIDE):
    """
    Resize an image while preserving its original aspect ratio.

    The longest side of the image is resized to a fixed value.
    This helps maintain consistent image dimensions while preventing
    distortion of glove geometry.

    Parameters
    ----------
    image : numpy.ndarray
        Input BGR image.
    target_long_side : int
        Desired size of the longest image dimension.

    Returns
    -------
    numpy.ndarray
        Resized image.
    """
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


# ============================================================
# DENOISING
# ============================================================

def denoise_image(image):
    """
    Apply mild bilateral filtering.

    Bilateral filtering reduces small image noise while preserving
    important object boundaries and defect edges. The filtering is
    intentionally mild because excessive smoothing may remove small
    defects such as spotting or tearing.

    Parameters
    ----------
    image : numpy.ndarray
        Resized BGR image.

    Returns
    -------
    numpy.ndarray
        Denoised BGR image.
    """
    denoised = cv2.bilateralFilter(
        image,
        d=5,
        sigmaColor=35,
        sigmaSpace=35
    )

    return denoised


# ============================================================
# COLOUR SPACE CONVERSION
# ============================================================

def convert_to_grayscale(image):
    """
    Convert BGR image to grayscale.

    Grayscale images are useful for:
    - edge detection
    - contour analysis
    - texture analysis
    - wrinkle/fold analysis
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def convert_to_hsv(image):
    """
    Convert BGR image to HSV.

    HSV separates:
    - hue
    - saturation
    - brightness

    This can be useful for colour-based defects such as dirty,
    stain and spotting.
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


def convert_to_lab(image):
    """
    Convert BGR image to LAB.

    LAB separates brightness from colour information and is useful
    for detecting colour abnormalities such as discoloration,
    stain and dirty regions.
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2LAB)


# ============================================================
# CONTRAST ENHANCEMENT
# ============================================================

def enhance_grayscale(gray):
    """
    Enhance local grayscale contrast using CLAHE.

    CLAHE can improve visibility of:
    - edges
    - folds
    - wrinkles
    - texture changes
    - surface irregularities

    CLAHE is applied only to the grayscale copy so that the original
    colour information remains unchanged.

    Parameters
    ----------
    gray : numpy.ndarray
        Grayscale image.

    Returns
    -------
    numpy.ndarray
        Contrast-enhanced grayscale image.
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


# ============================================================
# MAIN PREPROCESSING PIPELINE
# ============================================================

def preprocess_image(image):
    """
    Run the complete shared preprocessing pipeline.

    Flow:
        Input Image
            ↓
        Resize while maintaining aspect ratio
            ↓
        Mild noise reduction
            ↓
        Generate grayscale, HSV and LAB representations
            ↓
        CLAHE grayscale enhancement

    Parameters
    ----------
    image : numpy.ndarray
        Original BGR image.

    Returns
    -------
    dict
        Dictionary containing all prepared image representations.
    """
    if image is None or image.size == 0:
        raise ValueError("Input image is empty or invalid.")

    # Step 1: Resize image while preserving aspect ratio.
    resized = resize_image(image)

    # Step 2: Mild noise reduction.
    denoised = denoise_image(resized)

    # Step 3: Generate grayscale
    gray = convert_to_grayscale(denoised)

    # Step 4: Generate enhanced grayscale copy
    gray_enhanced = enhance_grayscale(gray)

    # Step 5: Generate HSV representation
    hsv = convert_to_hsv(denoised)

    # Step 6: Generate LAB representation
    lab = convert_to_lab(denoised)

    # Individual detectors can choose the best representation for
    # their defect-specific logic.
    return {
        "original": resized,
        "denoised": denoised,
        "gray": gray,
        "gray_enhanced": gray_enhanced,
        "hsv": hsv,
        "lab": lab,
    }


# ============================================================
# SIMPLE DEVELOPMENT CHECK
# ============================================================

if __name__ == "__main__":
    """
    Run a quick demonstration when the file is executed directly.

    This helps confirm the preprocessing pipeline works and prints
    useful output without requiring any other project files.
    """

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
