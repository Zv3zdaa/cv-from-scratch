import numpy as np

#Дополняет 2D-массив нулями по краям на `pad` пикселей
def pad2d(image: np.ndarray, pad: int) -> np.ndarray:
    h, w = image.shape
    padded = np.zeros((h + 2 * pad, w + 2 * pad), dtype=image.dtype)
    padded[pad:pad + h, pad:pad + w] = image
    return padded

#2D-свёртка методом скользящего окна с нулевым padding
def convolve2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    if image.ndim != 2:
        raise ValueError("image must be 2D")
    if kernel.shape[0] != kernel.shape[1]:
        raise ValueError("kernel must be square")
    if kernel.shape[0] % 2 == 0:
        raise ValueError("kernel size must be odd")
    
    kernel = kernel[::-1, ::-1]

    kh = kernel.shape[0]
    pad = kh // 2
    padded = pad2d(image, pad)

    h, w = image.shape
    out = np.zeros((h, w), dtype=np.float64)

    for y in range(h):
        for x in range(w):
            window = padded[y:y + kh, x:x + kh]
            out[y, x] = np.sum(window * kernel)

    return out