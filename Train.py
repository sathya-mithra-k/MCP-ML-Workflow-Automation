from pathlib import Path
import json
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

def _load_data(path: str) -> pd.DataFrame:
    path = Path(path)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_parquet(path)

def _train_model(train_path: str, target: str, task: str, output_dir: str = "artifacts",random_state: int = 42,n_jobs: int = -1) -> str:
    try:
        train_data = _load_data(train_path)
        if target not in train_data.columns:
            raise ValueError(f"Target column '{target}' not found in data.")

        if task not in ["classification", "regression"]:
            raise ValueError(f"Invalid task: {task}")

        X = train_data.drop(columns=[target])
        y = train_data[target]
        if task == "classification":
            model = RandomForestClassifier(random_state=random_state, n_jobs=n_jobs)
        else:
            model = RandomForestRegressor(random_state=random_state, n_jobs=n_jobs)
        model.fit(X,y)
        return model
    except Exception as e:
        return f"Error training model: {e}"

    run_dir = Path(output_dir) / f"train_rf_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    model_path = run_dir / "model.joblib"
    joblib.dump(model, model_path)

    manifest = {
        "task": task,
        "target": target,
        "train_rows": len(train_df),
        "features": list(X.columns),
        "artifacts": {
            "model": str(model_path),
            "manifest": str(run_dir / "train_manifest.json"),
        },
    }

    (run_dir / "train_manifest.json").write_text(json.dumps(manifest, indent=2))

    return json.dumps(
        {
            "status": "ok",
            "message": "Model trained successfully",
            "manifest": manifest,
        },
        indent=2,
    )
