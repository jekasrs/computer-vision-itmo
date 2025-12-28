from src.utils import erosion_opencv
from utils import read_image, erosion, save_image

if __name__ == '__main__':
    file_name = "composition"
    input_path = f"images/{file_name}.png"
    output_path_native = f"images/{file_name}_native.png"
    output_path_opencv = f"images/{file_name}_opencv.png"

    binary_image = read_image(input_path)

    eroded_image_native = erosion(binary_image)
    save_image(output_path_native, eroded_image_native)

    eroded_image_opencv = erosion_opencv(binary_image)
    save_image(output_path_opencv, eroded_image_opencv)

    print("Обработка завершена. Результат сохранён.")
