from pathlib import Path
from datetime import datetime
import pandas as pd
import cv2
from ultralytics import YOLO

from classification_utils import predict_single_image


def run_end_to_end_inference(
    sample_image: Path,
    det_weights: Path,
    cls_checkpoint: Path,
    conf: float = 0.25,
    output_dir: Path | None = None,
    actual_class: str | None = None,
) -> pd.DataFrame:
    detector = YOLO(str(det_weights))
    image = cv2.imread(str(sample_image))
    if image is None:
        return pd.DataFrame([{"error": f"Could not read image: {sample_image}"}])

    result = detector.predict(source=str(sample_image), conf=conf, device="cpu", verbose=False)[0]
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return pd.DataFrame([{"image": str(sample_image), "message": "No solar-panel detected"}])

    if output_dir is None:
        output_dir = Path("artifacts/predictions/end_to_end")
    output_dir.mkdir(parents=True, exist_ok=True)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / f"{sample_image.stem}_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    annotated = image.copy()
    for i, b in enumerate(boxes):
        x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = max(x2, 0), max(y2, 0)
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        crop_path = run_dir / f"crop_{i}.jpg"
        cv2.imwrite(str(crop_path), crop)
        preds = predict_single_image(crop_path, cls_checkpoint, top_k=3)

        top1 = preds[0] if len(preds) > 0 else {"class": "unknown", "probability": 0.0}

        label = f"{top1['class']} {float(top1['probability']):.2f}"
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 200, 0), 2)
        cv2.putText(
            annotated,
            label,
            (x1, max(15, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 200, 0),
            1,
            cv2.LINE_AA,
        )

        rows.append(
            {
                "image": str(sample_image),
                "bbox_id": i,
                "det_confidence": float(b.conf[0].item()),
                "actual_class": actual_class,
                "predicted_condition": top1["class"],
                "condition_probability": float(top1["probability"]),
                "crop_path": str(crop_path),
            }
        )

    annotated_path = run_dir / f"{sample_image.stem}_annotated.jpg"
    cv2.imwrite(str(annotated_path), annotated)

    for row in rows:
        row["annotated_image_path"] = str(annotated_path)
        row["run_dir"] = str(run_dir)

    return pd.DataFrame(rows)
