# postprocessing.py
"""
Модуль postprocessing: объединение блоков, определение языка и очистка текста.
"""
from typing import List, Dict, Tuple
from langdetect import detect

TextBlock = Dict[str, object]


def merge_overlapping(blocks: List[TextBlock], iou_threshold: float = 0.5) -> List[TextBlock]:
    merged = []
    for block in sorted(blocks, key=lambda b: b['coordinates'][1]):
        if not merged:
            merged.append(block)
            continue
        prev = merged[-1]
        if _iou(prev['coordinates'], block['coordinates']) > iou_threshold:
            # объединяем тексты и рамки
            prev['text'] += ' ' + block['text']
            prev['coordinates'] = [
                min(prev['coordinates'][0], block['coordinates'][0]),
                min(prev['coordinates'][1], block['coordinates'][1]),
                max(prev['coordinates'][2], block['coordinates'][2]),
                max(prev['coordinates'][3], block['coordinates'][3]),
            ]
            prev['confidence'] = max(prev['confidence'], block['confidence'])
        else:
            merged.append(block)
    return merged


def _iou(a: List[int], b: List[int]) -> float:
    xA = max(a[0], b[0]); yA = max(a[1], b[1])
    xB = min(a[2], b[2]); yB = min(a[3], b[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = (a[2]-a[0])*(a[3]-a[1])
    areaB = (b[2]-b[0])*(b[3]-b[1])
    return inter / float(areaA + areaB - inter) if (areaA+areaB-inter)>0 else 0


def detect_language(block: TextBlock) -> str:
    try:
        lang = detect(block['text'])
    except:
        lang = 'unknown'
    return lang


def postprocess(blocks: List[TextBlock]) -> List[TextBlock]:
    merged = merge_overlapping(blocks)
    for block in merged:
        block['language'] = detect_language(block)
    return merged
