from pathlib import Path
from typing import List
import pandas as pd


def read_csv(file_path: str) -> str:
    path = Path(file_path)
    if not path.exists():
        return f"File not found: {path}"
    if not path.is_file():
        return f"Path is not a file: {path}"

    try:
        df = pd.read_csv(path)
    except Exception as exc:  
        return f"Failed to read CSV: {exc}"

    parts: List[str] = []
    parts.append("Successfully read the CSV file.")
    parts.append("Preview (head):")
    parts.append(df.head().to_string())
    parts.append(f"Rows: {len(df)}")
    parts.append(f"Columns: {len(df.columns)}")
    parts.append(f"Dtypes:\n{df.dtypes.to_string()}")
    parts.append(f"Summary:\n{df.describe(include='all').to_string()}")
    parts.append(f"Column names: {list(df.columns)}")
    parts.append(f"Index: {df.index}")
    parts.append(f"Shape: {df.shape}")

    return "\n".join(parts)
