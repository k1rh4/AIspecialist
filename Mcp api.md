# ==========================================
# RAG 10일차 - MCP 관련 API 정리
# ==========================================

# 1. MCP Client (LlamaIndex)
mcp_client = BasicMCPClient(external_mcp_server)
mcp_tool = McpToolSpec(client=mcp_client)
tools = await mcp_tool.to_tool_list_async()
resources = await mcp_tool.fetch_resources()

# 2. MCP Agent
agent = FunctionAgent(...)
ctx = Context(agent)
handler = agent.run(question, ctx=ctx)
handler.stream_events()
response = await handler

# 3. MCP Server (Text2SQL)
mcp = FastMCP("text2sql")

@mcp.tool()
def get_schema() -> str:
    ...

mcp.run(transport="stdio")

# 4. 실습에서 정의한 MCP 관련 함수
KGQueryEngineWithMCP.init_agent()
KGQueryEngineWithMCP.query()

RAGwithMCP.retrieve()
RAGwithMCP.generate_response()
RAGwithMCP.inference()
