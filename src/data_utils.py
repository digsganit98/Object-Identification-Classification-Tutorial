from pathlib import Path
from shutil import copy2
import random
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images(directory: Path):
    if not directory.exists():
        return []
    return sorted([p for p in directory.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES])


def dataset_audit(detection_root: Path, classification_root: Path) -> dict:
    det_images = list_images(detection_root / "images")
    det_labels = list((detection_root / "labels").glob("*.txt")) if (detection_root / "labels").exists() else []

    class_counts = {}
    if classification_root.exists():
        for class_dir in sorted([p for p in classification_root.iterdir() if p.is_dir()]):
            class_counts[class_dir.name] = len(list_images(class_dir))

    return {
        "detection_images": len(det_images),
        "detection_labels": len(det_labels),
        "detection_class_file": str(detection_root / "classes.txt"),
        "classification_total_images": sum(class_counts.values()),
        "classification_class_counts": class_counts,
    }


def parse_yolo_annotations(images_dir: Path, labels_dir: Path) -> pd.DataFrame:
    rows = []
    for img_path in list_images(images_dir):
        label_path = labels_dir / f"{img_path.stem}.txt"
        if not label_path.exists():
            continue

        lines = label_path.read_text(encoding="utf-8").splitlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            class_id, xc, yc, w, h = parts
            w = float(w)
            h = float(h)
            rows.append(
                {
                    "image_id": img_path.stem,
                    "image_path": str(img_path),
                    "class_id": int(float(class_id)),
                    "x_center": float(xc),
                    "y_center": float(yc),
                    "width": w,
                    "height": h,
                    "bbox_area": w * h,
                    "aspect_ratio": w / max(h, 1e-8),
                }
            )

    return pd.DataFrame(rows)


def _copy_yolo_pair(stem: str, src_images: Path, src_labels: Path, dst_images: Path, dst_labels: Path):
    src_img = None
    for ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
        candidate = src_images / f"{stem}{ext}"
        if candidate.exists():
            src_img = candidate
            break

    src_lbl = src_labels / f"{stem}.txt"
    if src_img is None or not src_lbl.exists():
        return False

    copy2(src_img, dst_images / src_img.name)
    copy2(src_lbl, dst_labels / src_lbl.name)
    return True


def prepare_yolo_single_class_split(
    detection_root: Path,
    output_root: Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
):
    src_images = detection_root / "images"
    src_labels = detection_root / "labels"

    image_paths = list_images(src_images)
    stems = [p.stem for p in image_paths if (src_labels / f"{p.stem}.txt").exists()]
    if not stems:
        raise ValueError("No matched image-label pairs found in detection dataset")

    train_stems, test_stems = train_test_split(stems, train_size=train_ratio + val_ratio, random_state=seed, shuffle=True)
    val_fraction = val_ratio / (train_ratio + val_ratio)
    train_stems, val_stems = train_test_split(train_stems, test_size=val_fraction, random_state=seed, shuffle=True)

    split_map = {"train": train_stems, "val": val_stems, "test": test_stems}

    for split in ["train", "val", "test"]:
        (output_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    copied = {"train": 0, "val": 0, "test": 0}
    for split, split_stems in split_map.items():
        for stem in split_stems:
            ok = _copy_yolo_pair(
                stem,
                src_images,
                src_labels,
                output_root / "images" / split,
                output_root / "labels" / split,
            )
            if ok:
                copied[split] += 1

    data_yaml_path = output_root / "data.yaml"
    yaml_data = {
        "path": str(output_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": 1,
        "names": ["solar-panel"],
    }
    data_yaml_path.write_text(yaml.safe_dump(yaml_data, sort_keys=False), encoding="utf-8")

    summary = {
        "total_pairs": len(stems),
        "train": copied["train"],
        "val": copied["val"],
        "test": copied["test"],
        "data_yaml": str(data_yaml_path),
    }
    return summary


def build_classification_index(classification_root: Path) -> pd.DataFrame:
    rows = []
    if not classification_root.exists():
        return pd.DataFrame(columns=["image_id", "file_path", "label"])

    class_dirs = [d for d in sorted(classification_root.iterdir()) if d.is_dir()]
    for class_dir in class_dirs:
        label = class_dir.name
        for img_path in list_images(class_dir):
            rows.append(
                {
                    "image_id": img_path.stem,
                    "file_path": str(img_path),
                    "label": label,
                }
            )

    return pd.DataFrame(rows)


def add_stratified_splits(df: pd.DataFrame, seed: int = 42, train_ratio: float = 0.7, val_ratio: float = 0.15) -> pd.DataFrame:
    if df.empty:
        out = df.copy()
        out["split"] = []
        return out

    train_df, test_df = train_test_split(
        df,
        train_size=train_ratio + val_ratio,
        random_state=seed,
        stratify=df["label"],
    )

    val_fraction = val_ratio / (train_ratio + val_ratio)
    train_df, val_df = train_test_split(
        train_df,
        test_size=val_fraction,
        random_state=seed,
        stratify=train_df["label"],
    )

    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    out = pd.concat([train_df, val_df, test_df], ignore_index=True)
    return out
