"""
################################################################################
# detection.py
# -----------------------------------------------------------------------------
# Цель файла
#   • Надёжно локализовать окно чата на скриншоте.
#   • Обнаружить и замаскировать эмодзи/реакции до OCR‑этапа.
#   • Сегментировать каждый текстовый пузырёк в отдельный bounding‑box.
# -----------------------------------------------------------------------------
# Ключевые идеи
#   1. Сначала детектируем **chat_area**, чтобы сузить обработку.
#   2. Затем YOLO‑моделью ищем **emoji**.  Любое подозрительное «круглое»
#      изображение вырезается и восстанавливается inpaint‑алгоритмом, чтобы
#      фон пузырька стал сплошным — это повышает качество последующей
#      контурной сегментации.
#   3. После очистки применяем морфологический CLOSING + contour‑анализ для
#      выделения прямоугольников сообщений.
# -----------------------------------------------------------------------------
# Автор: ChatGPT‑assistant (май 2025)
################################################################################
"""

from __future__ import annotations

# ───── Импорт стандартных и сторонних библиотек ───────────────────────────────
import argparse
from pathlib import Path
from typing import List, Tuple

import cv2                         # OpenCV   — базовые операции с изображениями
import numpy as np                 # NumPy    — работа с массивами
from ultralytics import YOLO       # Ultralytics YOLOv8/v12 детектор объектов

# ───── Типы данных ────────────────────────────────────────────────────────────
BoundingBox = Tuple[int, int, int, int]   # (x1, y1, x2, y2) — int‑координаты

# ───── Пути к весам моделей (можно менять по необходимости) ───────────────────
MODELS_DIR         = Path(__file__).resolve().parents[1] / 'models'
CHAT_MODEL_PATH    = MODELS_DIR / 'chat_window_detector.pt'
EMOJI_MODEL_PATH   = MODELS_DIR / 'emoji_detector_yolov12n.pt'

# ───── Лениво инициализируем YOLO‑модели, чтобы не грузить их дважды ──────────
_chat_model  : YOLO | None = None
_emoji_model : YOLO | None = None

###############################################################################
# 1. Локализация окна чата
###############################################################################

def detect_chat_area(img: np.ndarray, conf: float = 0.50) -> BoundingBox:
    """Определяет координаты прямоугольной области переписки.

    Параметры
    ----------
    img  : np.ndarray  — исходный скриншот (формат BGR, uint8).
    conf : float       — минимальный порог уверенности YOLO‑предсказаний.

    Возврат
    -------
    BoundingBox : (x1, y1, x2, y2) левой‑верхней и правой‑нижней точек окна.
    """
    global _chat_model
    if _chat_model is None:
        _chat_model = YOLO(str(CHAT_MODEL_PATH))      # загружаем один раз

    # ─── Запускаем инференс — imgsz=640 хватает для FullHD‑скринов ───────────
    results = _chat_model(img, imgsz=640, conf=conf, verbose=False)[0]
    if not len(results.boxes):
        raise RuntimeError('Не удалось обнаружить окно чата — проверьте модель.')

    # ─── Выбираем самый крупный прямоугольник (он и есть главный чат) ─────────
    boxes = results.boxes.xyxy.cpu().numpy()                    # (n,4)
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    x1, y1, x2, y2 = boxes[areas.argmax()].astype(int)
    return x1, y1, x2, y2

###############################################################################
# 2. Детекция эмодзи
###############################################################################

def detect_emoji(img: np.ndarray, conf: float = 0.25) -> List[BoundingBox]:
    """Находит все эмодзи/реакции на переданном изображении.

    Возвращает список прямоугольников; координаты глобальные относительно img.
    """
    global _emoji_model
    if _emoji_model is None:
        _emoji_model = YOLO(str(EMOJI_MODEL_PATH))

    preds = _emoji_model(img, imgsz=640, conf=conf, verbose=False)[0]
    if not len(preds.boxes):
        return []

    bxs = preds.boxes.xyxy.cpu().numpy().astype(int)
    return [(x1, y1, x2, y2) for x1, y1, x2, y2 in bxs]

###############################################################################
# 3. Маскирование эмодзи
###############################################################################

def mask_emojis(img: np.ndarray, bbox_list: List[BoundingBox]) -> np.ndarray:
    """Возвращает копию изображения, где все bbox‑области заинпейнчены.

    Алгоритм:
      • Создаём чёрно‑белую маску (255 — там, где эмодзи)
      • Запускаем OpenCV‑inpaint (Tele­gram/WhatsApp фон → подходит метод NS)
    """
    if not bbox_list:
        # нет эмодзи — просто вернём исходную картинку, чтобы избежать копирования
        return img.copy()

    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    for x1, y1, x2, y2 in bbox_list:
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, thickness=-1)

    # inpaintRadius=3 — компромисс между скоростью и качеством «затирания»
    cleaned = cv2.inpaint(img, mask, inpaintRadius=3, flags=cv2.INPAINT_NS)
    return cleaned

###############################################################################
# 4. Сегментация пузырьков сообщений
###############################################################################

def segment_messages(chat_region: np.ndarray) -> List[BoundingBox]:
    """Находит каждый пузырёк сообщения внутри обрезанной области чата.

    Метод основан на бинаризации + морфологии.
    Возврат: список bbox‑координат **относительно chat_region**.
    """
    # ── 1. Превращаем в grayscale, чтобы упростить порог. ────────────────────
    gray = cv2.cvtColor(chat_region, cv2.COLOR_BGR2GRAY)

    # ── 2. Адаптивная бинаризация: устойчива к любой яркости ─────────────
    block = max(11, (gray.shape[1] // 40) | 1)   # нечётное
    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block,
        10,
    )

       # ── 3. Closing с ядром, масштабируемым от ширины окна ───────────────
    k_w = max(3, int(gray.shape[1] * 0.03))
    k_h = max(3, int(gray.shape[0] * 0.02))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_w, k_h))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # ── 4. Контуры внешних объектов — будущие границы пузырьков. ──────────────
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # ── 5. Фильтруем «шум»: сохраняем только крупные области (>1000 px). ─────
    bubbles = [cv2.boundingRect(cnt) for cnt in contours if cv2.contourArea(cnt) > 1000]

    # ── 6. Приводим (x,y,w,h) → (x1,y1,x2,y2) и сортируем сверху вниз. ───────
    boxes = [(x, y, x + w, y + h) for x, y, w, h in bubbles]
    boxes.sort(key=lambda b: b[1])

    return boxes

###############################################################################
# 5. CLI‑вход для отладки модуля в одиночку
###############################################################################

def _parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Детекция области чата, эмодзи и сегментация сообщений")
    parser.add_argument("--input", required=True, help="Путь к PNG/JPG скриншоту")
    parser.add_argument("--chat-conf", type=float, default=0.50,
                        help="Порог уверенности YOLO‑модели чата (0–1)")
    parser.add_argument("--emoji-conf", type=float, default=0.25,
                        help="Порог YOLO‑модели эмодзи (0–1)")
    parser.add_argument("--debug", action="store_true",
                        help="Показать окна визуализации")
    return parser.parse_args()


def _debug_visualisation(img: np.ndarray, chat_box: BoundingBox,
                          emoji_boxes: List[BoundingBox], message_boxes: List[BoundingBox]) -> None:
    """Рисует все bbox‑ы разными цветами для визуальной проверки."""
    vis = img.copy()

    # Рисуем окно чата (синее)
    cv2.rectangle(vis, (chat_box[0], chat_box[1]), (chat_box[2], chat_box[3]), (255, 0, 0), 2)

    # Рисуем эмодзи (красное)
    for x1, y1, x2, y2 in emoji_boxes:
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Рисуем пузырьки (зелёное)
    for x1, y1, x2, y2 in message_boxes:
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow("Детекция", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    args = _parse_cli()

    # ─── 0. Читаем исходный скрин ─────────────────────────────────────────────
    screenshot = cv2.imread(args.input)
    if screenshot is None:
        raise FileNotFoundError(f"Не удалось открыть файл {args.input}")

    # ─── 1. Локализация окна чата ─────────────────────────────────────────────
    cx1, cy1, cx2, cy2 = detect_chat_area(screenshot, conf=args.chat_conf)
    chat_crop = screenshot[cy1:cy2, cx1:cx2]

    # ─── 2. Детекция + маскирование эмодзи ────────────────────────────────────
    emoji_boxes_global = detect_emoji(screenshot, conf=args.emoji_conf)
    screenshot_clean   = mask_emojis(screenshot, emoji_boxes_global)
    chat_crop_clean    = screenshot_clean[cy1:cy2, cx1:cx2]

    # ─── 3. Сегментация пузырьков ─────────────────────────────────────────────
    bubble_boxes_local = segment_messages(chat_crop_clean)

    # ─── 4. Визуализация (по желанию) ─────────────────────────────────────────
    if args.debug:
        _debug_visualisation(screenshot_clean,
                             (cx1, cy1, cx2, cy2),
                             emoji_boxes_global,
                             [(bx1+cx1, by1+cy1, bx2+cx1, by2+cy1) for bx1,by1,bx2,by2 in bubble_boxes_local])

    # ─── 5. Выводим статистику в консоль ──────────────────────────────────────
    print(f"Найдено эмодзи: {len(emoji_boxes_global)}, пузырьков: {len(bubble_boxes_local)}")
