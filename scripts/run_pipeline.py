# scripts/run_pipeline.py
import argparse
from pathlib import Path
from pipeline.exporter import export_json   # <— наш главный вход
import sys, pathlib

ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

def main() -> None:
    p = argparse.ArgumentParser(
        description="Run full OCR pipeline on one image.")
    p.add_argument("image", type=Path, help="Path to screenshot")
    p.add_argument("-o", "--out", type=Path, default=Path("result.json"),
                   help="Where to save JSON")
    p.add_argument("--chat-weights", type=Path, default=Path("models/yolo_chat.pt"))
    p.add_argument("--emoji-weights", type=Path, default=Path("models/emoji_ui.pt"))
    p.add_argument("--debug-dir", type=Path, default=None,
                   help="If set, saves intermediate images")
    args = p.parse_args()

    export_json(
        img_path=args.image,
        out_json=args.out,
        chat_weights=args.chat_weights,
        emoji_weights=args.emoji_weights,
        debug_dir=args.debug_dir,
    )
    print("✓  JSON saved to", args.out)

if __name__ == "__main__":
    main()
