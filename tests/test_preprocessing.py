import cv2
import numpy as np
from pipeline.preprocessing import to_grayscale, denoise, enhance_contrast, preprocess

# создаём цветное изображение
color = np.random.randint(0,255,(100,100,3),dtype=np.uint8)

def test_to_grayscale():
    gray = to_grayscale(color)
    assert len(gray.shape) == 2
    assert gray.dtype == np.uint8

def test_denoise():
    gray = to_grayscale(color)
    den = denoise(gray)
    assert den.shape == gray.shape

def test_enhance_contrast():
    gray = to_grayscale(color)
    con = enhance_contrast(gray)
    assert con.shape == gray.shape

def test_preprocess_pipeline():
    pre = preprocess(color)
    assert len(pre.shape) == 2