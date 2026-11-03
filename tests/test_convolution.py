import numpy as np
import pytest

from cv_lib.convolution import convolve2d, pad2d


def test_pad2d():
    img = np.array([[1, 2], [3, 4]])
    expected = np.array([[0, 0, 0, 0],
                         [0, 1, 2, 0],
                         [0, 3, 4, 0],
                         [0, 0, 0, 0]])
    assert np.array_equal(pad2d(img, 1), expected)


def test_identity_kernel_returns_input():
    img = np.arange(25, dtype=np.float64).reshape(5, 5)
    identity = np.array([[0, 0, 0],
                         [0, 1, 0],
                         [0, 0, 0]], dtype=np.float64)
    assert np.array_equal(convolve2d(img, identity), img)


def test_matches_opencv_interior():
    cv2 = pytest.importorskip("cv2")
    rng = np.random.default_rng(0)
    img = rng.integers(-50, 50, (10, 10)).astype(np.float64)
    kernel = rng.random((3, 3)) - 0.5

    ours = convolve2d(img, kernel)
    ref = cv2.filter2D(img, -1, kernel[::-1, ::-1], borderType=cv2.BORDER_CONSTANT)

    assert np.allclose(ours[1:-1, 1:-1], ref[1:-1, 1:-1], atol=1e-9)