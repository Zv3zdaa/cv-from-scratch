import numpy as np
from cv_lib.convolution import convolve2d


def main() -> None:
    matrix = np.random.randint(-11, 11, size=(5, 5))
    kernel = np.array([[0, 1, -9], [5, 7, -10], [8, -9, 7]])

    print("Input matrix:")
    print(matrix)
    print("\nKernel:")
    print(kernel)

    result = convolve2d(matrix, kernel)
    print("\nConvolution result:")
    print(result)

    try:
        import cv2
    except ImportError:
        print("\n(cv2 not installed — skipping reference check)")
        return

    # OpenCV filter2D выполняет корреляцию, поэтому для свертки ядро переворачивают.
    reference = cv2.filter2D(
        matrix.astype(np.float64), -1, kernel[::-1, ::-1],
        borderType=cv2.BORDER_CONSTANT,
    )
    print("\nOpenCV reference:")
    print(reference)

    # Сравниваем только внутреннюю область — края могут отличаться из-за border handling.
    interior_match = np.allclose(result[1:-1, 1:-1], reference[1:-1, 1:-1])
    print(f"\nInterior matches OpenCV: {interior_match}")


if __name__ == "__main__":
    main()