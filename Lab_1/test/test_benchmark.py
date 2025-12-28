import time

import matplotlib.pyplot as plt

from src.utils import erosion_opencv, erosion, read_image


def test_benchmark(image_path: str) -> None:
    kernel_sizes = [3, 5, 7, 9, 11, 13, 15, 17]
    native_times = []
    opencv_times = []

    test_image = read_image(image_path)

    for k in kernel_sizes:
        start = time.time()
        erosion(test_image, k)
        native_times.append(time.time() - start)

        start = time.time()
        erosion_opencv(test_image, k)
        opencv_times.append(time.time() - start)

    print("native times: ", native_times)
    print("opencv times: ", opencv_times)

    plt.figure()
    plt.plot(kernel_sizes, native_times, marker='o', label="Native")
    plt.plot(kernel_sizes, opencv_times, marker='o', label="OpenCV")
    plt.xticks(kernel_sizes)
    plt.xlabel("Kernel size")
    plt.ylabel("Execution time (seconds)")
    plt.title("Comparison of performance")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    image_path = "../images/rectangle.png"
    test_benchmark(image_path)
