import numpy as np

from failure_gallery import size_bin


def test_size_bins_and_overshooting_box():
    assert size_bin(np.array([0, 0, 5, 5]), 100, 100) == "small"     # 0.25% of the image
    assert size_bin(np.array([0, 0, 20, 20]), 100, 100) == "medium"  # 4%
    assert size_bin(np.array([0, 0, 90, 90]), 100, 100) == "large"
    assert size_bin(np.array([-10, -10, 110, 110]), 100, 100) == "large"  # label larger than the frame
