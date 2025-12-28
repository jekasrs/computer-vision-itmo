import cv2
import numpy as np


def read_image(path: str) -> np.ndarray:
    """ Чтение изображения и преобразование в бинарный вид."""
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError("Не удалось загрузить изображение")

    _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    return binary


def save_image(path: str, image: np.ndarray) -> None:
    """Сохранение изображения в файл."""
    cv2.imwrite(path, image)


def erosion(binary_image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Морфологическая эрозии с квадратным структурным элементом kernel_size. """
    if kernel_size % 2 == 0 or kernel_size < 3:
        raise ValueError("kernel_size должен быть нечётным и >= 3")

    pad = kernel_size // 2
    height, width = binary_image.shape
    result = np.zeros_like(binary_image)

    for y in range(pad, height - pad):
        for x in range(pad, width - pad):
            window = binary_image[
                y - pad : y + pad + 1,
                x - pad : x + pad + 1
            ]
            if np.all(window == 255):
                result[y, x] = 255

    return result


def erosion_opencv(binary_image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.erode(binary_image, kernel)
