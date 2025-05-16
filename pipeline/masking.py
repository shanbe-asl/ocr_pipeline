# masking.py
"""
Модуль masking: удаление интерфейсных элементов (кнопок, эмодзи, иконок) из области чата.
"""
import cv2
import numpy as np
from typing import List, Tuple


def mask_ui_elements(chat_region: np.ndarray, ui_hsv_ranges: List[Tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """
    Маскирование интерфейсных элементов по HSV-диапазонам.
    :param chat_region: BGR-изображение области чата
    :param ui_hsv_ranges: список кортежей (hsv_min, hsv_max) для элементов UI
    :return: модифицированное изображение
    """
    hsv = cv2.cvtColor(chat_region, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for hsv_min, hsv_max in ui_hsv_ranges:
        curr_mask = cv2.inRange(hsv, hsv_min, hsv_max)
        mask = cv2.bitwise_or(mask, curr_mask)
    # инвертируем маску: 1 там, где нужно сохранять текст
    inv_mask = cv2.bitwise_not(mask)
    # применяем маску к изображению: заливаем UI-области белым
    result = chat_region.copy()
    result[mask > 0] = (255,255,255)
    return result

if __name__ == "__main__":
    import argparse
    from ingestion import load_image
    from detection import detect_chat_area

    parser = argparse.ArgumentParser(description="Маскирование UI-элементов перед OCR.")
    parser.add_argument("--input", required=True, help="Путь к скриншоту")
    parser.add_argument("--output", required=True, help="Путь для сохранения результата")
    args = parser.parse_args()

    img = load_image(args.input)
    x1,y1,x2,y2 = detect_chat_area(img)
    chat = img[y1:y2, x1:x2]
    # пример диапазонов HSV для кнопок и эмодзи (настройте под UI)
    ranges = [
        (np.array([0,0,0]), np.array([180,255,30])),  # возможно, тёмные элементы
        (np.array([0, 20, 70]), np.array([20, 255, 255])),  # пример
    ]
    masked = mask_ui_elements(chat, ranges)
    cv2.imwrite(args.output, masked)
    print(f"Masked chat saved: {args.output}")