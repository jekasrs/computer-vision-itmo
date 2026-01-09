from Lab_1.src.utils import read_image, erosion, save_image, erosion_opencv
import os

if __name__ == '__main__':
    file_name = "composition"

    module_dir = os.path.dirname(__file__)
    input_path = os.path.join(module_dir, 'images', f'{file_name}.png')
    output_path_native = os.path.join(module_dir, 'images', f'{file_name}_native.png')
    output_path_opencv = os.path.join(module_dir, 'images', f'{file_name}_opencv.png')

    binary_image = read_image(input_path)

    eroded_image_native = erosion(binary_image)
    save_image(output_path_native, eroded_image_native)

    eroded_image_opencv = erosion_opencv(binary_image)
    save_image(output_path_opencv, eroded_image_opencv)

    print("Обработка завершена. Результат сохранён.")
