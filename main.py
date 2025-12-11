from mcp.server.fastmcp import FastMCP
from read_csv import read_csv

mcp = FastMCP()


@mcp.tool()
def read_csv_tool(file_path: str) -> str:
    return read_csv(file_path)


if __name__ == "__main__":
    mcp.run(transport="stdio")