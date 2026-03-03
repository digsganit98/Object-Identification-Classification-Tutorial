from pathlib import Path
import json
import pandas as pd


def load_metrics_table(metrics_dir: Path) -> pd.DataFrame:
    rows = []
    if not metrics_dir.exists():
        return pd.DataFrame()

    for p in metrics_dir.glob("*.json"):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(data, dict) and "model_name" in data:
                rows.append(data)
        except Exception:
            continue

    return pd.DataFrame(rows)


def summarize_metrics(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame([{"status": "No metrics found"}])
    cols = [c for c in ["accuracy", "f1_macro", "elapsed_seconds"] if c in df.columns]
    if not cols:
        return pd.DataFrame([{"status": "No standard metric columns available"}])
    return df[cols].describe().T.reset_index().rename(columns={"index": "metric"})
