"""Synthetic plant-camera degradations: low light, haze/smoke, motion blur.

Each function takes a BGR uint8 image and a severity (1 = moderate, 2 = severe)
and returns a degraded BGR uint8 image of the same shape. They are meant to be
applied at the model's input resolution so that, e.g., blur length is comparable
across images.
"""

import cv2
import numpy as np


def low_light(img: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    """Underexposed camera: scale brightness down, then add sensor noise."""
    scale, noise_sigma = {1: (0.35, 4.0), 2: (0.15, 8.0)}[severity]
    out = img.astype(np.float32) * scale
    out += rng.normal(0.0, noise_sigma, img.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


def haze(img: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    """Dust/smoke/steam via the atmospheric scattering model: I = J*t + A*(1-t)."""
    t = {1: 0.55, 2: 0.3}[severity]
    airlight = 200.0 + rng.uniform(-15, 15)  # light grey smoke
    out = img.astype(np.float32) * t + airlight * (1.0 - t)
    return np.clip(out, 0, 255).astype(np.uint8)


def motion_blur(img: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    """Camera vibration / fast-moving worker: linear blur at a random angle."""
    length = {1: 9, 2: 21}[severity]
    kernel = np.zeros((length, length), np.float32)
    kernel[length // 2, :] = 1.0
    angle = rng.uniform(0, 180)
    rot = cv2.getRotationMatrix2D((length / 2 - 0.5, length / 2 - 0.5), angle, 1.0)
    kernel = cv2.warpAffine(kernel, rot, (length, length))
    kernel /= kernel.sum()
    return cv2.filter2D(img, -1, kernel)


CONDITIONS = {"low_light": low_light, "haze": haze, "motion_blur": motion_blur}
SEVERITIES = (1, 2)
