# ocr_engine.py
"""
Модуль ocr_engine: распознавание текста в сегментированных областях с помощью Tesseract.
"""
import cv2
import pytesseract
from pytesseract import Output
from typing import List, Tuple, Dict
import numpy as np

# Тип для текстового блока: Dict с полями text, coords, confidence
TextBlock = Dict[str, object]


def run_tesseract(img: cv2.Mat, lang: str = 'rus+eng') -> List[TextBlock]:
    """
    Запускает Tesseract на изображении и возвращает список текстовых блоков.

    :param img: Grayscale или BGR-изображение области текста
    :param lang: Языковой пакет для Tesseract (rus+eng)
    :return: Список слов/строк с координатами и уверенностью
    """
    # Конфигурация: уровень строки
    config = '--psm 6'
    data = pytesseract.image_to_data(img, lang=lang, config=config, output_type=Output.DICT)
    blocks: List[TextBlock] = []
    n = len(data['text'])
    for i in range(n):
        text = data['text'][i].strip()
        if not text:
            continue
        x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
        conf = float(data['conf'][i]) / 100.0
        block = {
            'text': text,
            'coordinates': [x, y, x + w, y + h],
            'confidence': conf
        }
        blocks.append(block)
    return blocks

if __name__ == "__main__":
    import argparse
    from ingestion import load_image
    from detection import detect_chat_area, segment_messages
    from masking import mask_ui_elements

    parser = argparse.ArgumentParser(description="Запуск OCR на изображениях чата.")
    parser.add_argument("--input", required=True, help="Путь к входному изображению PNG/JPG")
    parser.add_argument("--output", required=True, help="Путь для сохранения JSON результата")
    args = parser.parse_args()

    img = load_image(args.input)
    x1,y1,x2,y2 = detect_chat_area(img)
    chat = img[y1:y2, x1:x2]
    masked = mask_ui_elements(chat, ui_hsv_ranges=[(np.array([0,0,0]), np.array([180,255,30]))])
    # получаем коробочки сообщений
    boxes = segment_messages(masked)
    all_blocks: List[TextBlock] = []
    for bx, by, bx2, by2 in boxes:
        region = masked[by:by2, bx:bx2]
        blocks = run_tesseract(region)
        # корректируем координаты относительно полного чата
        for b in blocks:
            x0, y0, x1b, y1b = b['coordinates']
            b['coordinates'] = [x0+bx, y0+by, x1b+bx, y1b+by]
        all_blocks.extend(blocks)

    import json
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump({'text_blocks': all_blocks}, f, ensure_ascii=False, indent=2)
    print(f"OCR result saved to {args.output}")