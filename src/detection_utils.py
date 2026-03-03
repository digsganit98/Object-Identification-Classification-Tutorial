from pathlib import Path
import cv2
import pandas as pd
from ultralytics import YOLO

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def predict_detection_on_image(
    weights_path: Path,
    image_path: Path,
    output_dir: Path,
    conf: float = 0.25,
    class_name: str = "solar-panel",
):
    output_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(weights_path))

    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    result = model.predict(source=str(image_path), conf=conf, device="cpu", verbose=False)[0]
    boxes = result.boxes

    rows = []
    annotated = image.copy()
    if boxes is not None:
        for i, b in enumerate(boxes):
            x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
            score = float(b.conf[0].item())
            x1, y1 = max(x1, 0), max(y1, 0)
            x2, y2 = max(x2, 0), max(y2, 0)

            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 200, 0), 2)
            cv2.putText(
                annotated,
                f"{class_name} {score:.2f}",
                (x1, max(15, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 200, 0),
                1,
                cv2.LINE_AA,
            )

            rows.append(
                {
                    "bbox_id": i,
                    "class_name": class_name,
                    "confidence": score,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                }
            )

    out_path = output_dir / f"{image_path.stem}_detected.jpg"
    cv2.imwrite(str(out_path), annotated)

    return pd.DataFrame(rows), out_path


def run_detection_and_generate_crops(weights_path: Path, image_dir: Path, output_crop_dir: Path, conf: float = 0.25) -> pd.DataFrame:
    output_crop_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(weights_path))

    rows = []
    image_paths = sorted([p for p in image_dir.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES])

    for img_path in image_paths:
        image = cv2.imread(str(img_path))
        if image is None:
            continue

        result = model.predict(source=str(img_path), conf=conf, device="cpu", verbose=False)[0]
        boxes = result.boxes
        if boxes is None:
            continue

        for i, b in enumerate(boxes):
            x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
            x1, y1 = max(x1, 0), max(y1, 0)
            x2, y2 = max(x2, 0), max(y2, 0)
            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            crop_id = f"{img_path.stem}_{i}"
            crop_path = output_crop_dir / f"{crop_id}.jpg"
            cv2.imwrite(str(crop_path), crop)

            rows.append(
                {
                    "crop_id": crop_id,
                    "parent_image_id": img_path.stem,
                    "crop_path": str(crop_path),
                    "bbox_confidence": float(b.conf[0].item()),
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                }
            )

    return pd.DataFrame(rows)
