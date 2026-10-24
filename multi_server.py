from mcp.server.fastmcp import FastMCP
mcp_server = FastMCP("math")

@mcp_server.tool()
def add(a:int, b:int)->int:
    sum = a + b
    return sum
@mcp_server.tool()
def multiply(a:int,  b:int)->int:
    return a*b


if __name__ == "__main__":
    mcp_server.run(transport = "stdio")
    print("server initialized...")