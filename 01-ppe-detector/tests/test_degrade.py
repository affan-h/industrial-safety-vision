import numpy as np
import pytest

from degrade import CONDITIONS, SEVERITIES, haze, low_light


@pytest.fixture
def img():
    return np.random.default_rng(0).integers(0, 256, (120, 160, 3), dtype=np.uint8)


@pytest.mark.parametrize("name", CONDITIONS)
@pytest.mark.parametrize("severity", SEVERITIES)
def test_shape_and_dtype_preserved(img, name, severity):
    out = CONDITIONS[name](img, severity, np.random.default_rng(1))
    assert out.shape == img.shape and out.dtype == np.uint8


def test_low_light_darkens_more_when_severe(img):
    rng = lambda: np.random.default_rng(1)
    assert low_light(img, 2, rng()).mean() < low_light(img, 1, rng()).mean() < img.mean()


def test_haze_reduces_contrast(img):
    assert haze(img, 2, np.random.default_rng(1)).std() < haze(img, 1, np.random.default_rng(1)).std() < img.std()
