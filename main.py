from mcp.server.fastmcp import FastMCP
from read_csv import read_csv

mcp = FastMCP()


@mcp.tool()
def read_csv_tool(file_path: str) -> str:
    return read_csv(file_path)

@mcp.tool()
def preprocess_tool_from_csv(file_path: str, target: str, **kwargs) -> str:
    p = Path(file_path)
    if not p.exists() or not p.is_file():
        return json.dumps({"status": "error", "message": f"File not found: {p}"})
    df = pd.read_csv(p)
    return preprocess_data_mcp(df, target, **kwargs)


if __name__ == "__main__":
    mcp.run(transport="stdio")