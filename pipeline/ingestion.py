
# ingestion.py
"""
Модуль ingestion: загрузка и валидация входных изображений для OCR-пайплайна.
"""
import cv2
import os
from typing import Optional


def load_image(path: str) -> Optional[cv2.Mat]:
    """
    Загружает изображение из файла.

    :param path: Путь к файлу изображения (PNG или JPG).
    :return: Изображение в формате BGR (cv2.Mat) или None, если загрузка не удалась.
    """
    if not os.path.isfile(path):
        print(f"Error: файл не найден: {path}")
        return None
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"Error: не удалось загрузить изображение: {path}")
    return img


def validate_image(img: cv2.Mat) -> bool:
    """
    Проверяет, что загруженное изображение соответствует требованиям:
    - Не пустое
    - Имеет допустимые каналы (3 канала BGR)

    :param img: Изображение для проверки
    :return: True, если изображение валидно, иначе False
    """
    if img is None:
        print("Error: изображение отсутствует (None)")
        return False
    # Проверяем, что это цветное изображение
    if len(img.shape) != 3 or img.shape[2] != 3:
        print(f"Error: ожидается BGR-изображение с 3 каналами, получено: {img.shape}")
        return False
    # Дополнительные проверки (размер, формат и т.д.) можно добавить здесь
    h, w, _ = img.shape
    if h < 50 or w < 50:
        print(f"Warning: слишком маленькое изображение: {w}x{h}")
    return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Инструмент загрузки и проверки изображений для OCR-пайплайна.")
    parser.add_argument("--input", required=True, help="Путь к входному изображению PNG/JPG")
    args = parser.parse_args()

    img = load_image(args.input)
    if not validate_image(img):
        exit(1)
    print(f"Изображение успешно загружено и валидировано: {args.input}")
