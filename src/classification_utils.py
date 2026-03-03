from pathlib import Path
import json
import time
from dataclasses import dataclass

import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, classification_report

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class IndexedImageDataset(Dataset):
    def __init__(self, df: pd.DataFrame, class_to_idx: dict, image_size: int = 224, train: bool = False):
        self.df = df.reset_index(drop=True)
        self.class_to_idx = class_to_idx
        if train:
            self.transform = transforms.Compose(
                [
                    transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomRotation(degrees=8),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((image_size, image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row["file_path"]).convert("RGB")
        x = self.transform(image)
        y = self.class_to_idx[row["label"]]
        return x, y


@dataclass
class TrainArtifacts:
    model_path: Path
    metrics_path: Path
    report_path: Path


def _build_model(num_classes: int, pretrained: bool = False):
    weights = None
    if pretrained:
        try:
            weights = models.MobileNet_V3_Small_Weights.DEFAULT
        except Exception:
            weights = None
    model = models.mobilenet_v3_small(weights=weights)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    return model


def _run_epoch(model, loader, criterion, optimizer, device, train: bool):
    if train:
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    y_true = []
    y_pred = []

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        if train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train):
            logits = model(xb)
            loss = criterion(logits, yb)
            preds = torch.argmax(logits, dim=1)
            if train:
                loss.backward()
                optimizer.step()

        running_loss += loss.item() * xb.size(0)
        y_true.extend(yb.detach().cpu().tolist())
        y_pred.extend(preds.detach().cpu().tolist())

    epoch_loss = running_loss / max(len(loader.dataset), 1)
    acc = accuracy_score(y_true, y_pred) if y_true else 0.0
    f1m = f1_score(y_true, y_pred, average="macro", zero_division=0) if y_true else 0.0
    return epoch_loss, acc, f1m


def train_classification_cpu(project_root: Path, cfg: dict):
    index_path = project_root / cfg.get("index_csv", "data/processed/classification_index.csv")
    if not index_path.exists():
        return {"error": f"Missing index file: {index_path}"}

    df = pd.read_csv(index_path)
    required = {"file_path", "label", "split"}
    if not required.issubset(df.columns):
        return {"error": f"Index must contain columns: {sorted(required)}"}

    classes = sorted(df["label"].unique().tolist())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    idx_to_class = {i: c for c, i in class_to_idx.items()}

    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()
    test_df = df[df["split"] == "test"].copy()

    if train_df.empty or val_df.empty or test_df.empty:
        return {"error": "Train/val/test splits are required and cannot be empty"}

    image_size = int(cfg.get("image_size", 224))
    batch_size = int(cfg.get("batch_size", 16))
    num_epochs = int(cfg.get("num_epochs", 8))
    lr = float(cfg.get("learning_rate", 1e-3))
    pretrained = bool(cfg.get("pretrained", False))

    train_ds = IndexedImageDataset(train_df, class_to_idx, image_size=image_size, train=True)
    val_ds = IndexedImageDataset(val_df, class_to_idx, image_size=image_size, train=False)
    test_ds = IndexedImageDataset(test_df, class_to_idx, image_size=image_size, train=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    device = torch.device("cpu")
    model = _build_model(num_classes=len(classes), pretrained=pretrained).to(device)
    class_counts = train_df["label"].value_counts()
    count_tensor = torch.tensor([float(class_counts.get(c, 1.0)) for c in classes], dtype=torch.float32)
    class_weights = count_tensor.sum() / (len(classes) * count_tensor.clamp(min=1.0))
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3,
    )

    best_val_f1 = -1.0
    best_state = None
    history = []

    start = time.time()
    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc, train_f1 = _run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc, val_f1 = _run_epoch(model, val_loader, criterion, optimizer, device, train=False)

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "train_f1_macro": train_f1,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_f1_macro": val_f1,
            }
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = model.state_dict()
        scheduler.step(val_f1)

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for xb, yb in test_loader:
            logits = model(xb.to(device))
            preds = torch.argmax(logits, dim=1).cpu().tolist()
            y_pred.extend(preds)
            y_true.extend(yb.tolist())

    elapsed = time.time() - start
    metrics = {
        "model_name": "mobilenet_v3_small_cpu",
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "best_val_f1_macro": float(best_val_f1),
        "num_classes": len(classes),
        "num_train": int(len(train_df)),
        "num_val": int(len(val_df)),
        "num_test": int(len(test_df)),
        "epochs": num_epochs,
        "elapsed_seconds": round(elapsed, 2),
        "timestamp": int(time.time()),
    }

    report = classification_report(
        [idx_to_class[i] for i in y_true],
        [idx_to_class[i] for i in y_pred],
        labels=classes,
        target_names=classes,
        zero_division=0,
        output_dict=True,
    )

    artifacts_dir = project_root / "artifacts"
    models_dir = artifacts_dir / "models"
    metrics_dir = artifacts_dir / "metrics"
    models_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / "cls_mobilenetv3_small_cpu.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "class_to_idx": class_to_idx,
            "idx_to_class": idx_to_class,
            "image_size": image_size,
            "model_name": "mobilenet_v3_small",
        },
        model_path,
    )

    metrics_path = metrics_dir / "classification_mobilenetv3_small_cpu_metrics.json"
    report_path = metrics_dir / "classification_mobilenetv3_small_cpu_report.json"
    history_path = metrics_dir / "classification_mobilenetv3_small_cpu_history.json"

    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

    return metrics


def predict_single_image(image_path: Path, checkpoint_path: Path, top_k: int = 3):
    device = torch.device("cpu")
    ckpt = torch.load(checkpoint_path, map_location=device)

    idx_to_class = {int(k): v for k, v in ckpt["idx_to_class"].items()} if any(isinstance(k, str) for k in ckpt["idx_to_class"].keys()) else ckpt["idx_to_class"]
    image_size = int(ckpt.get("image_size", 224))

    model = _build_model(num_classes=len(idx_to_class), pretrained=False).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    image = Image.open(image_path).convert("RGB")
    x = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]

    top_probs, top_idxs = torch.topk(probs, k=min(top_k, len(idx_to_class)))
    rows = []
    for prob, idx in zip(top_probs.tolist(), top_idxs.tolist()):
        rows.append({"class": idx_to_class[int(idx)], "probability": float(prob)})
    return rows


def predict_directory(image_dir: Path, checkpoint_path: Path, top_k: int = 3) -> pd.DataFrame:
    rows = []
    image_paths = sorted([p for p in image_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES])
    for image_path in image_paths:
        preds = predict_single_image(image_path=image_path, checkpoint_path=checkpoint_path, top_k=top_k)
        top = preds[0] if preds else {"class": "unknown", "probability": 0.0}
        rows.append(
            {
                "image_path": str(image_path),
                "predicted_class": top["class"],
                "predicted_probability": float(top["probability"]),
                "top_k": preds,
            }
        )
    return pd.DataFrame(rows)
