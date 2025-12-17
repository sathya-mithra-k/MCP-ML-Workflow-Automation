from mcp.server.fastmcp import FastMCP
from tools.read_csv import read_csv
from pathlib import Path
import json
import pandas as pd
from tools.preprocess import preprocess_data_mcp
from tools.train import train_model

mcp = FastMCP()


@mcp.tool()
def read_csv_tool(file_path: str) -> str:
    return read_csv(file_path)

@mcp.tool()
def preprocess_tool_from_csv(file_path: str, target: str, **kwargs) -> str:
    p = Path(file_path)
    if not p.exists() or not p.is_file():
        return json.dumps({"status": "error", "message": f"File not found: {p}"})
    try:
        df = pd.read_csv(p)
    except Exception as exc:
        return json.dumps({"status": "error", "message": f"Failed to read CSV: {exc}"})

    allowed_kwargs = {
        "output_dir",
        "drop_missing_threshold",
        "onehot_max_cardinality",
        "scale_numeric",
        "test_size",
        "random_state",
    }
    filtered = {k: v for k, v in kwargs.items() if k in allowed_kwargs}
    return preprocess_data_mcp(df, target, **filtered)

@mcp.tool()
def train_model_tool(train_path: str, target: str, task: str, **kwargs) -> str:
    return train_model(train_path, target, task, **kwargs)

if __name__ == "__main__":
    mcp.run(transport="stdio")