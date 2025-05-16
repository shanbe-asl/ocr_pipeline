
# detection.py
"""
Модуль detection: локализация области чата и сегментация сообщений внутри.
"""
import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Tuple

# Тип для рамки: (x1, y1, x2, y2)
BoundingBox = Tuple[int, int, int, int]

# Загрузка вашей обученной модели YOLO
model = YOLO('models/yolo_chat.pt')


def detect_chat_area(img: np.ndarray, conf: float = 0.5) -> BoundingBox:
    """
    Обнаруживает область чата на полном скриншоте.

    :param img: Полное BGR-изображение
    :param conf: Порог уверенности для детектора
    :return: Координаты области чата (x1, y1, x2, y2)
    """
    results = model(img, imgsz=640, conf=conf, verbose=False)[0]
    if not len(results.boxes):
        raise RuntimeError('Chat area not detected')
    boxes = results.boxes.xyxy.cpu().numpy()
    areas = (boxes[:,2] - boxes[:,0]) * (boxes[:,3] - boxes[:,1])
    x1, y1, x2, y2 = boxes[areas.argmax()].astype(int)
    return x1, y1, x2, y2


def segment_messages(chat_region: np.ndarray) -> List[BoundingBox]:
    """
    Сегментирует сообщения внутри области чата методом детекции контуров.

    :param chat_region: BGR-изображение области чата
    :return: Список координат сообщений [(x1,y1,x2,y2), ...]
    """
    gray = cv2.cvtColor(chat_region, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 10))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # Поиск контуров
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bubbles = [cv2.boundingRect(cnt) for cnt in contours if cv2.contourArea(cnt) > 1000]
    boxes = [(x, y, x+w, y+h) for x, y, w, h in bubbles]
    boxes.sort(key=lambda b: b[1])

    return boxes

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Детекция области чата и сегментация сообщений.")
    parser.add_argument("--input", required=True, help="Путь к входному изображению PNG/JPG")
    parser.add_argument("--debug", action="store_true", help="Показать результат с рамками")
    args = parser.parse_args()

    img = cv2.imread(args.input)
    x1, y1, x2, y2 = detect_chat_area(img)
    chat = img[y1:y2, x1:x2]
    message_boxes = segment_messages(chat)

    if args.debug:
        # Визуализация результатов
        vis = chat.copy()
        for bx, by, bx2, by2 in message_boxes:
            cv2.rectangle(vis, (bx, by), (bx2, by2), (0,255,0), 2)
        cv2.imshow("Messages", vis)
        cv2.waitKey(0)