# preprocess_mcp.py
from pathlib import Path
from typing import Dict, Any, List, Union
import json
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.utils.multiclass import type_of_target
from datetime import datetime


def _normalize_colName(col_name: str) -> str:
    return (
        col_name.lower()
        .strip()
        .replace(" ", "_")
        .replace("-", "_")
    )

def _is_classification_target_safe(series: pd.Series) -> bool:
    try:
        return type_of_target(series) in {"binary", "multiclass"}
    except Exception:
        return False

def _to_dense_if_sparse(arr: Union[np.ndarray, "scipy.sparse.spmatrix"]):
    # lazy import to avoid hard dependency unless needed
    try:
        import scipy.sparse as sp
    except Exception:
        sp = None
    if sp is not None and sp.issparse(arr):
        return arr.toarray()
    return arr

def _get_ohe_feature_names(transformer, cols: List[str]) -> List[str]:
    # robustly obtain OHE output names with fallbacks
    encoder = None
    if hasattr(transformer, "named_steps") and "encoder" in transformer.named_steps:
        encoder = transformer.named_steps["encoder"]
    elif hasattr(transformer, "named_steps") and "onehot" in transformer.named_steps:
        encoder = transformer.named_steps["onehot"]

    if encoder is None:
        return cols

    try:
        return list(encoder.get_feature_names_out(cols))
    except Exception:
        # fallback: create names manually with category lengths if available
        try:
            cats = encoder.categories_
            out = []
            for col, cat in zip(cols, cats):
                out.extend([f"{col}__ohe_{i}" for i in range(len(cat))])
            return out
        except Exception:
            # final fallback
            return cols

def _make_onehot_encoder() -> OneHotEncoder:
    """
    sklearn compatibility helper:
    - newer versions use `sparse_output`
    - older versions use `sparse`
    """
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)

def preprocess_data_mcp(
    data: pd.DataFrame,
    target: str,
    output_dir: Union[str, Path] = "artifacts",
    drop_missing_threshold: float = 0.5,
    onehot_max_cardinality: int = 20,
    scale_numeric: bool = True,
    test_size: float = 0.2,
    random_state: int = 42,
) -> str:
    """
    Hardened minimal preprocess. Returns JSON string with manifest or error.
    """
    try:
        if data is None or data.empty:
            raise ValueError("Input dataframe is empty.")

        # normalise and copy
        df = data.copy()
        normalized_cols = {col: _normalize_colName(col) for col in df.columns}
        df = df.rename(columns=normalized_cols)
        target_norm = _normalize_colName(target)
        if target_norm not in df.columns:
            raise ValueError(f"Target column '{target}' not found after normalization.")

        # drop high-missing, preserve target
        missing_frac = df.isna().mean()
        dropped_columns = [c for c, frac in missing_frac.items() if frac > drop_missing_threshold and c != target_norm]
        if dropped_columns:
            df = df.drop(columns=dropped_columns)

        if set(df.columns) == {target_norm}:
            raise ValueError("No feature columns remain after dropping missing columns.")

        y = df[target_norm]
        X = df.drop(columns=[target_norm])

        numeric_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
        categorical_cols = [c for c in X.columns if c not in numeric_cols]

        low_card_cats = [c for c in categorical_cols if X[c].nunique(dropna=True) <= onehot_max_cardinality]
        # high_card_cats intentionally not used in transformers
        high_card_cats = [c for c in categorical_cols if c not in low_card_cats]

        # pipelines
        num_steps = [("imputer", SimpleImputer(strategy="median"))]
        if scale_numeric:
            num_steps.append(("scaler", StandardScaler()))
        numeric_pipeline = Pipeline(num_steps)

        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
            ("encoder", _make_onehot_encoder()),
        ])

        transformers = []
        if numeric_cols:
            transformers.append(("numeric", numeric_pipeline, numeric_cols))
        if low_card_cats:
            transformers.append(("categorical", categorical_pipeline, low_card_cats))

        if not transformers:
            raise ValueError("No usable feature columns after preprocessing selections.")

        preprocessor = ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=0)

        # stratify if classification-like target
        stratify = y if _is_classification_target_safe(y) else None

        # split first (so imputer/encoder are fit only on train)
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=stratify
        )

        preprocessor.fit(X_train_raw)
        X_train_proc = preprocessor.transform(X_train_raw)
        X_test_proc = preprocessor.transform(X_test_raw)

        # make dense if sparse
        X_train_proc = _to_dense_if_sparse(X_train_proc)
        X_test_proc = _to_dense_if_sparse(X_test_proc)

        # feature names
        feature_names: List[str] = []
        if numeric_cols:
            feature_names.extend(numeric_cols)
        if low_card_cats:
            try:
                cat_transformer = preprocessor.named_transformers_.get("categorical")
                feature_names.extend(_get_ohe_feature_names(cat_transformer, low_card_cats))
            except Exception:
                # fallback: generate positional names later
                pass

        # ensure feature_names length matches columns
        n_features = X_train_proc.shape[1] if hasattr(X_train_proc, "shape") else len(feature_names)
        if len(feature_names) != n_features:
            feature_names = [f"f_{i}" for i in range(n_features)]

        train_df = pd.DataFrame(X_train_proc, columns=feature_names)
        test_df = pd.DataFrame(X_test_proc, columns=feature_names)
        train_df[target_norm] = np.array(y_train).reshape(-1)
        test_df[target_norm] = np.array(y_test).reshape(-1)

        # create run-specific output dir
        base_output = Path(output_dir)
        run_dir = base_output / f"preprocess_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        run_dir.mkdir(parents=True, exist_ok=True)

        train_path = run_dir / "train.parquet"
        test_path = run_dir / "test.parquet"
        preproc_path = run_dir / "preprocessor.joblib"

        train_df.to_parquet(train_path, index=False)
        test_df.to_parquet(test_path, index=False)
        joblib.dump(preprocessor, preproc_path)

        manifest: Dict[str, Any] = {
            "target": target_norm,
            "rows": {"train": len(train_df), "test": len(test_df)},
            "features": feature_names,
            "dropped_columns": dropped_columns,
            "high_cardinality_categoricals": high_card_cats,
            "onehot_max_cardinality": onehot_max_cardinality,
            "drop_missing_threshold": drop_missing_threshold,
            "scale_numeric": scale_numeric,
            "classification": stratify is not None,
            "artifacts": {
                "train": str(train_path),
                "test": str(test_path),
                "preprocessor": str(preproc_path),
                "manifest": str(run_dir / "manifest.json"),
            },
        }

        manifest_path = run_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2))

        return json.dumps({"status": "ok", "manifest": manifest}, indent=2)

    except Exception as exc:
        return json.dumps({"status": "error", "message": str(exc)}, indent=2)


