# exporter.py
"""
Модуль exporter: экспорт результатов OCR в JSON.
"""
import cv2
import json
from typing import List, Dict
import numpy as np

TextBlock = Dict[str, object]


def export_json(blocks: List[TextBlock], output_path: str) -> None:
    result = {'text_blocks': blocks}
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    import argparse
    from .ocr_engine import run_tesseract
    from .preprocessing import preprocess
    from .detection import detect_chat_area, segment_messages
    from .masking import mask_ui_elements
    from .postprocessing import postprocess

    parser = argparse.ArgumentParser(description='Полный запуск OCR-пайплайна')
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    img = cv2.imread(args.input)
    x1,y1,x2,y2 = detect_chat_area(img)
    chat = img[y1:y2, x1:x2]
    masked = mask_ui_elements(chat, ui_hsv_ranges=[(np.array([0,0,0]), np.array([180,255,30]))])
    pre = preprocess(masked)
    boxes = segment_messages(masked)
    all_blocks = []
    for bx,by,bx2,by2 in boxes:
        region = masked[by:by2, bx:bx2]
        blocks = run_tesseract(region)
        for b in blocks:
            b['coordinates'] = [b['coordinates'][0]+bx, b['coordinates'][1]+by, b['coordinates'][2]+bx, b['coordinates'][3]+by]
        all_blocks.extend(blocks)
    final = postprocess(all_blocks)
    export_json(final, args.output)
    print(f'Result saved to {args.output}')
    cv2.imwrite("debug_chat_region.jpg", chat)
    cv2.imwrite("debug_masked.jpg", masked)
    cv2.imwrite("debug_preprocessed.jpg", pre)
    for idx, (bx, by, bx2, by2) in enumerate(boxes):
        region = pre[by:by2, bx:bx2]
        cv2.imwrite(f"debug_segment_{idx}.jpg", region)
