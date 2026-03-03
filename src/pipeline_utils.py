from pathlib import Path
import tempfile
import pandas as pd
import cv2
from ultralytics import YOLO

from classification_utils import predict_single_image


def run_end_to_end_inference(sample_image: Path, det_weights: Path, cls_checkpoint: Path, conf: float = 0.25) -> pd.DataFrame:
    detector = YOLO(str(det_weights))
    image = cv2.imread(str(sample_image))
    if image is None:
        return pd.DataFrame([{"error": f"Could not read image: {sample_image}"}])

    result = detector.predict(source=str(sample_image), conf=conf, device="cpu", verbose=False)[0]
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return pd.DataFrame([{"image": str(sample_image), "message": "No solar-panel detected"}])

    rows = []
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir = Path(tmp_dir)
        for i, b in enumerate(boxes):
            x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
            x1, y1 = max(x1, 0), max(y1, 0)
            x2, y2 = max(x2, 0), max(y2, 0)
            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            crop_path = tmp_dir / f"crop_{i}.jpg"
            cv2.imwrite(str(crop_path), crop)
            preds = predict_single_image(crop_path, cls_checkpoint, top_k=3)

            top = preds[0] if preds else {"class": "unknown", "probability": 0.0}
            rows.append(
                {
                    "image": str(sample_image),
                    "bbox_id": i,
                    "det_confidence": float(b.conf[0].item()),
                    "predicted_condition": top["class"],
                    "condition_probability": top["probability"],
                    "top3": preds,
                }
            )

    return pd.DataFrame(rows)
