"""
Synthetic image degradation for CCTV robustness testing.

Simulates real-world conditions common in IP camera footage:
  - blur      : camera out-of-focus, fast motion
  - fog       : weather, dust, dirty lens
  - noise     : low-light, cheap sensors, high ISO
  - low_res   : distant cameras, digital zoom
  - compression: heavy H.264/MJPEG compression artifacts (IP cameras)

Each function takes an RGB uint8 ndarray and returns a degraded copy.
"""
import cv2
import numpy as np


def apply_blur(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """
    Gaussian blur — simulates out-of-focus camera or motion blur.
    kernel_size: 3=mild, 7=moderate, 15=heavy, 25=severe
    """
    k = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    return cv2.GaussianBlur(image, (k, k), 0)


def apply_fog(image: np.ndarray, intensity: float = 0.4) -> np.ndarray:
    """
    Fog/haze overlay — white veil mixed over image.
    intensity: 0.0=no fog, 0.4=moderate, 0.7=heavy, 0.9=dense fog
    """
    intensity = float(np.clip(intensity, 0.0, 1.0))
    fog_layer = np.full_like(image, 255, dtype=np.uint8)
    return cv2.addWeighted(image, 1.0 - intensity, fog_layer, intensity, 0)


def apply_noise(image: np.ndarray, sigma: float = 20.0) -> np.ndarray:
    """
    Gaussian noise — simulates low-light sensor noise.
    sigma: 10=mild, 25=moderate, 50=heavy, 80=severe
    """
    noise = np.random.normal(0, sigma, image.shape).astype(np.float32)
    noisy = np.clip(image.astype(np.float32) + noise, 0, 255)
    return noisy.astype(np.uint8)


def apply_low_res(image: np.ndarray, scale_factor: float = 0.25) -> np.ndarray:
    """
    Downscale then upscale — simulates distant camera or digital zoom pixelation.
    scale_factor: 0.5=mild, 0.25=moderate, 0.125=severe (like 8x zoom on cheap camera)
    """
    h, w = image.shape[:2]
    small_w = max(1, int(w * scale_factor))
    small_h = max(1, int(h * scale_factor))
    small = cv2.resize(image, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def apply_compression(image: np.ndarray, quality: int = 10) -> np.ndarray:
    """
    JPEG compression artifacts — simulates IP camera H.264/MJPEG at low bitrate.
    quality: 90=almost clean, 30=moderate, 10=heavy, 5=severe (typical cheap IP cam)
    """
    quality = int(np.clip(quality, 1, 95))
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encoded = cv2.imencode(".jpg", image, encode_params)
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR)


def apply_combined(image: np.ndarray,
                   blur_kernel: int = 5,
                   fog_intensity: float = 0.3,
                   noise_sigma: float = 15.0,
                   jpeg_quality: int = 20) -> np.ndarray:
    """
    Combined degradation — closest to real-world bad CCTV footage.
    Applies blur + fog + noise + compression in sequence.
    """
    out = apply_blur(image, blur_kernel)
    out = apply_fog(out, fog_intensity)
    out = apply_noise(out, noise_sigma)
    out = apply_compression(out, jpeg_quality)
    return out


# Preset degradation levels for benchmarking
DEGRADATION_LEVELS = {
    # name → (function, kwargs, description)
    "clean": (lambda img: img.copy(), {}, "No degradation — baseline"),

    # Blur levels
    "blur_mild":   (apply_blur, {"kernel_size": 3},  "Mild blur (k=3)"),
    "blur_moderate": (apply_blur, {"kernel_size": 7}, "Moderate blur (k=7)"),
    "blur_heavy":  (apply_blur, {"kernel_size": 15}, "Heavy blur (k=15)"),
    "blur_severe": (apply_blur, {"kernel_size": 25}, "Severe blur (k=25) — extreme defocus"),

    # Fog levels
    "fog_mild":    (apply_fog, {"intensity": 0.2}, "Mild fog (20%)"),
    "fog_moderate": (apply_fog, {"intensity": 0.4}, "Moderate fog (40%)"),
    "fog_heavy":   (apply_fog, {"intensity": 0.6}, "Heavy fog (60%)"),
    "fog_dense":   (apply_fog, {"intensity": 0.8}, "Dense fog (80%) — almost whiteout"),

    # Noise levels
    "noise_mild":  (apply_noise, {"sigma": 10},  "Mild noise (σ=10)"),
    "noise_moderate": (apply_noise, {"sigma": 25}, "Moderate noise (σ=25)"),
    "noise_heavy": (apply_noise, {"sigma": 50},  "Heavy noise (σ=50)"),

    # Low resolution (distant camera)
    "lowres_half":    (apply_low_res, {"scale_factor": 0.5},   "Half resolution"),
    "lowres_quarter": (apply_low_res, {"scale_factor": 0.25},  "Quarter resolution — 50m camera"),
    "lowres_eighth":  (apply_low_res, {"scale_factor": 0.125}, "Eighth resolution — 100m+ camera"),

    # Compression
    "compression_moderate": (apply_compression, {"quality": 30}, "JPEG quality 30"),
    "compression_heavy":    (apply_compression, {"quality": 10}, "JPEG quality 10 — cheap IP cam"),
    "compression_severe":   (apply_compression, {"quality": 5},  "JPEG quality 5 — very low bitrate"),

    # Combined (realistic bad CCTV scenario)
    "combined_mild":   (apply_combined, {"blur_kernel": 3, "fog_intensity": 0.2, "noise_sigma": 10, "jpeg_quality": 30}, "Mild combined — slightly bad conditions"),
    "combined_typical": (apply_combined, {"blur_kernel": 5, "fog_intensity": 0.3, "noise_sigma": 20, "jpeg_quality": 20}, "Typical combined — common CCTV scenario"),
    "combined_severe": (apply_combined, {"blur_kernel": 9, "fog_intensity": 0.5, "noise_sigma": 40, "jpeg_quality": 10}, "Severe combined — worst case CCTV"),
}


def degrade(image: np.ndarray, level_name: str) -> np.ndarray:
    """Apply a named degradation level from DEGRADATION_LEVELS."""
    if level_name not in DEGRADATION_LEVELS:
        raise ValueError(f"Unknown degradation level: {level_name}. Choose from: {list(DEGRADATION_LEVELS)}")
    fn, kwargs, _ = DEGRADATION_LEVELS[level_name]
    return fn(image, **kwargs)
