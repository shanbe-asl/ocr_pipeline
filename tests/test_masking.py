import cv2
import numpy as np
from pipeline.masking import mask_ui_elements

def test_mask_ui_elements():
    # создаём цветной квадрат, который попадает в HSV-диапазон
    img = np.zeros((100,100,3), dtype=np.uint8)
    # закрашиваем середину красным (HSV около [0,255,255])
    img[:] = (0,0,255)
    # определяем диапазон HSV для красного
    hsv_min = np.array([0, 200, 200])
    hsv_max = np.array([10, 255, 255])
    masked = mask_ui_elements(img, [(hsv_min, hsv_max)])
    # все пиксели должны стать белыми
    assert np.all(masked == 255)