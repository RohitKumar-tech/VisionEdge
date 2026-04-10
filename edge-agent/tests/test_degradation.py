"""
Tests for the degradation pipeline itself.
Verifies each degradation produces valid output and measurable change.
"""
import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "benchmark"))
from degradation import (
    apply_blur, apply_fog, apply_noise, apply_low_res,
    apply_compression, apply_combined, degrade, DEGRADATION_LEVELS
)


@pytest.fixture
def sample_image():
    """A synthetic BGR image with a gradient — varied content so degradations are measurable."""
    rng = np.random.default_rng(seed=42)
    img = rng.integers(50, 200, (112, 112, 3), dtype=np.uint8)
    return img


def test_blur_reduces_sharpness(sample_image):
    blurred = apply_blur(sample_image, kernel_size=15)
    assert blurred.shape == sample_image.shape
    assert blurred.dtype == np.uint8
    # Blurred image has lower variance (less detail)
    assert blurred.std() < sample_image.std()


def test_blur_even_kernel_corrected(sample_image):
    """Even kernel sizes must be auto-corrected to odd."""
    result = apply_blur(sample_image, kernel_size=4)
    assert result.shape == sample_image.shape  # should not crash


def test_fog_shifts_pixels_toward_white(sample_image):
    fogged = apply_fog(sample_image, intensity=0.6)
    assert fogged.shape == sample_image.shape
    assert fogged.dtype == np.uint8
    # Foggy image has higher mean (shifted toward white=255)
    assert fogged.mean() > sample_image.mean()


def test_fog_intensity_zero_is_unchanged(sample_image):
    result = apply_fog(sample_image, intensity=0.0)
    np.testing.assert_array_equal(result, sample_image)


def test_fog_intensity_clipped_above_one(sample_image):
    result = apply_fog(sample_image, intensity=2.0)
    assert result.shape == sample_image.shape  # should not crash or produce wrong shape


def test_noise_changes_pixel_values(sample_image):
    noisy = apply_noise(sample_image, sigma=30)
    assert noisy.shape == sample_image.shape
    assert noisy.dtype == np.uint8
    assert not np.array_equal(noisy, sample_image)


def test_noise_stays_in_valid_range(sample_image):
    noisy = apply_noise(sample_image, sigma=200)  # extreme noise
    assert noisy.min() >= 0
    assert noisy.max() <= 255


def test_low_res_preserves_shape(sample_image):
    result = apply_low_res(sample_image, scale_factor=0.25)
    assert result.shape == sample_image.shape  # shape restored after upscale


def test_low_res_reduces_detail(sample_image):
    result = apply_low_res(sample_image, scale_factor=0.125)
    # After severe downscale+upscale, laplacian variance (sharpness) drops
    import cv2
    orig_sharpness = cv2.Laplacian(sample_image, cv2.CV_64F).var()
    result_sharpness = cv2.Laplacian(result, cv2.CV_64F).var()
    assert result_sharpness < orig_sharpness


def test_compression_produces_artifacts(sample_image):
    result = apply_compression(sample_image, quality=5)
    assert result.shape == sample_image.shape
    assert result.dtype == np.uint8
    # Heavy compression changes pixel values
    assert not np.array_equal(result, sample_image)


def test_combined_applies_all_effects(sample_image):
    result = apply_combined(sample_image,
                             blur_kernel=5, fog_intensity=0.3,
                             noise_sigma=20, jpeg_quality=20)
    assert result.shape == sample_image.shape
    assert result.dtype == np.uint8
    # Result should differ significantly from original
    assert np.abs(result.astype(float) - sample_image.astype(float)).mean() > 5.0


def test_degrade_all_levels_do_not_crash(sample_image):
    """Every degradation level must produce valid output without exception."""
    for level_name in DEGRADATION_LEVELS:
        result = degrade(sample_image, level_name)
        assert result.shape == sample_image.shape, f"Shape mismatch for level: {level_name}"
        assert result.dtype == np.uint8, f"Dtype wrong for level: {level_name}"


def test_degrade_unknown_level_raises():
    img = np.zeros((112, 112, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="Unknown degradation level"):
        degrade(img, "totally_fake_level")


def test_clean_level_returns_exact_copy(sample_image):
    result = degrade(sample_image, "clean")
    np.testing.assert_array_equal(result, sample_image)
