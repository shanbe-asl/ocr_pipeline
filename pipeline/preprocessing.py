# preprocessing.py
"""
Модуль preprocessing: предобработка изображений для улучшения качества OCR.
"""
import cv2
import numpy as np


def to_grayscale(img: cv2.Mat) -> cv2.Mat:
    """
    Преобразование BGR-изображения в оттенки серого.
    """
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def enhance_contrast(img_gray: cv2.Mat) -> cv2.Mat:
    """
    Улучшение контраста методом CLAHE (адаптивное выравнивание гистограммы).
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(img_gray)


def denoise(img_gray: cv2.Mat) -> cv2.Mat:
    """
    Удаление шума с помощью билинейной фильтрации.
    """
    return cv2.bilateralFilter(img_gray, d=9, sigmaColor=75, sigmaSpace=75)


def preprocess(img: cv2.Mat) -> cv2.Mat:
    """
    Полная цепочка предобработки:
    1. Перевод в градации серого
    2. Удаление шума
    3. Улучшение контраста

    :param img: Исходное BGR-изображение
    :return: Предобработанное grayscale-изображение
    """
    gray = to_grayscale(img)
    denoised = denoise(gray)
    enhanced = enhance_contrast(denoised)
    return enhanced


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Предобработка изображений для OCR-пайплайна.")
    parser.add_argument("--input", required=True, help="Путь к входному изображению PNG/JPG")
    parser.add_argument("--output", required=True, help="Путь для сохранения результата")
    args = parser.parse_args()

    img = cv2.imread(args.input)
    preproc = preprocess(img)
    cv2.imwrite(args.output, preproc)
    print(f"Предобработанное изображение сохранено: {args.output}")