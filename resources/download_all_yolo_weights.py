#!/usr/bin/env python3
import os
import sys
from pathlib import Path

MODELS = [
    # YOLOv8 (detect)
    "yolov8n.pt","yolov8s.pt","yolov8m.pt","yolov8l.pt","yolov8x.pt",
    # YOLOv9 (detect): official sizes are t/s/m/c/e
    "yolov9t.pt","yolov9s.pt","yolov9m.pt","yolov9c.pt","yolov9e.pt",
    # YOLOv10 (detect)
    "yolov10n.pt","yolov10s.pt","yolov10m.pt","yolov10l.pt","yolov10x.pt",
    # YOLO11 (detect) - note: Ultralytics uses "yolo11*.pt" (not "yolov11*.pt")
    "yolo11n.pt","yolo11s.pt","yolo11m.pt","yolo11l.pt","yolo11x.pt",
    # YOLO12 (detect)
    "yolo12n.pt","yolo12s.pt","yolo12m.pt","yolo12l.pt","yolo12x.pt",
    # YOLO26 (detect)
    "yolo26n.pt","yolo26s.pt","yolo26m.pt","yolo26l.pt","yolo26x.pt",
]

def main() -> int:
    try:
        from ultralytics import YOLO
    except Exception as e:
        print("ERROR: ultralytics is not installed. Run: pip install -U ultralytics")
        print(f"Details: {e}")
        return 2

    out_dir = Path.cwd() / "weights"
    out_dir.mkdir(parents=True, exist_ok=True)

    ok, fail = 0, 0

    for name in MODELS:
        print(f"\n=== Downloading: {name} ===")
        try:
            # Ultralytics will auto-download if weight is not found.
            model = YOLO(name)
            # Try to copy the resolved weights file into ./weights for convenience.
            w = getattr(model, "ckpt_path", None) or getattr(model, "model", None)
            # Newer Ultralytics: model.ckpt_path is a string path to the .pt
            if isinstance(w, (str, os.PathLike)) and os.path.exists(w):
                dst = out_dir / os.path.basename(str(w))
                if not dst.exists():
                    import shutil
                    shutil.copy2(str(w), str(dst))
            ok += 1
        except Exception as e:
            fail += 1
            print(f"FAILED: {name}\n  {e}")

    print("\n==============================")
    print(f"Done. Success: {ok} | Failed: {fail}")
    print(f"Local folder: {out_dir}")
    return 0 if fail == 0 else 1

if __name__ == "__main__":
    raise SystemExit(main())
