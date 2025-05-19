# tests/test_detection.py
import numpy as np
import pytest
from pipeline.detection import detect_chat_area, segment_messages

# Dummy classes для имитации res.boxes.xyxy.cpu().numpy()
class DummyXY:
    def __init__(self, arr):
        self._arr = arr
    def cpu(self):
        return self
    def numpy(self):
        return self._arr

class DummyBoxes:
    def __init__(self, coords):
        # coords: numpy array of shape (N,4)
        self.xyxy = DummyXY(coords)
    def __len__(self):
        return self.xyxy._arr.shape[0]

class DummyResult:
    def __init__(self, coords):
        self.boxes = DummyBoxes(coords)
    def __getitem__(self, idx):
        return self

class DummyModel:
    def __call__(self, img, imgsz=640, conf=0.5, verbose=False):
        # возвращаем список из одного DummyResult с одним боксом
        return [DummyResult(np.array([[0,0,20,20]]))]

@pytest.fixture(autouse=True)
def patch_model(monkeypatch):
    # Патчим модель в модуле pipeline.detection
    import pipeline.detection as det_mod
    monkeypatch.setattr(det_mod, 'model', DummyModel(), raising=True)
    yield


def test_detect_chat_area():
    img = np.zeros((100,100,3), dtype=np.uint8)
    x1, y1, x2, y2 = detect_chat_area(img)
    assert (x1, y1, x2, y2) == (0, 0, 20, 20)


def test_segment_messages_empty():
    chat = np.ones((100,100,3), dtype=np.uint8) * 255
    boxes = segment_messages(chat)
    assert boxes == []
