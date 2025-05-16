from pipeline.postprocessing import merge_overlapping, detect_language

def test_merge_overlapping():
    blocks = [
        {'text': 'A', 'coordinates': [0, 0, 50, 20], 'confidence': 0.9},
        {'text': 'B', 'coordinates': [10, 5, 60, 25], 'confidence': 0.8},
        {'text': 'C', 'coordinates': [100, 100, 150, 120], 'confidence': 0.95}
    ]
    merged = merge_overlapping(blocks, iou_threshold=0.1)
    # первые два должны слиться в один
    assert len(merged) == 2
    assert 'A' in merged[0]['text'] and 'B' in merged[0]['text']

def test_detect_language():
    # Используем более длинное предложение для надёжной детекции
    block = {'text': 'This is a sample English sentence for language detection.'}
    lang = detect_language(block)
    assert lang == 'en', f"Expected 'en' for English text, got '{lang}'"