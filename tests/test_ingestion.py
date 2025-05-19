# tests/test_ingestion.py
import pytest
import cv2
from pipeline.ingestion import load_image, validate_image

def test_load_image_success(tmp_path):
    # создаём небольшой чёрный png
    img_path = tmp_path / "test.png"
    blank = cv2.UMat(10, 10, cv2.CV_8UC3).get() * 0
    cv2.imwrite(str(img_path), blank)
    img = load_image(str(img_path))
    assert img is not None
    assert img.shape[0] == 10 and img.shape[1] == 10

def test_load_image_not_found():
    img = load_image("nonexistent.png")
    assert img is None

def test_validate_image_none():
    assert not validate_image(None)

def test_validate_image_wrong_shape():
    import numpy as np
    img = np.zeros((10,10), dtype=np.uint8)
    assert not validate_image(img)


env = pytest.importorskip("cv2")