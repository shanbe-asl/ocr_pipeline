import cv2
import numpy as np
from pipeline.ocr_engine import run_tesseract

def test_run_tesseract_simple():
    # создаём белое изображение и пишем текст
    img = np.ones((100,300,3), dtype=np.uint8)*255
    cv2.putText(img, 'Test', (5,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
    # конвертируем в серое
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blocks = run_tesseract(gray)
    texts = [b['text'] for b in blocks]
    assert any('Test' in t for t in texts)