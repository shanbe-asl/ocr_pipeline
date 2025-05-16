import subprocess
import json
import pytest
from pathlib import Path

def normalize_text_blocks(blocks):
    return sorted(
        [
            {
                "text": block["text"].strip(),
                "language": block["language"],
                "coordinates": tuple(block["coordinates"]),
                "confidence": round(block["confidence"], 2)
            }
            for block in blocks
        ],
        key=lambda x: (x["coordinates"], x["text"])
    )

@pytest.mark.parametrize("fixture", ["chat1", "chat2"])
def test_pipeline_end_to_end(tmp_path, fixture):
    # Paths to fixtures
    img_path = Path("tests/fixtures") / f"{fixture}.jpg"
    expected_json_path = Path("tests/fixtures") / f"{fixture}.json"

    # Output JSON in temporary directory
    out_path = tmp_path / "result.json"

    # Execute CLI pipeline
    subprocess.run([
        "python", "-m", "pipeline.exporter",
        "--input", str(img_path),
        "--output", str(out_path)
    ], check=True)

    # Compare normalized results for robustness
    result = json.loads(out_path.read_text(encoding="utf-8"))
    expected = json.loads(expected_json_path.read_text(encoding="utf-8"))

    result_normalized = normalize_text_blocks(result["text_blocks"])
    expected_normalized = normalize_text_blocks(expected["text_blocks"])

    assert result_normalized == expected_normalized, (
        f"Mismatch in OCR result for fixture '{fixture}':\n"
        f"Expected: {expected_normalized}\n"
        f"Got: {result_normalized}"
    )
