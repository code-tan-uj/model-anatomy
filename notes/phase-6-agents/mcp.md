# Model Context Protocol (MCP)

> Based on: Anthropic MCP specification (2024)
> Official docs: https://modelcontextprotocol.io/
> Local copy: papers/5.mcp.pdf

---

## What is MCP?

MCP (Model Context Protocol) is an **open protocol** that standardizes how AI models connect to external tools, data sources, and services.

Before MCP, every AI application had to build its own custom integrations for each tool — a different API, a different format, a different authentication method. This created a fragmented ecosystem.

MCP solves this by defining a universal interface:
```
Any LLM  ←→  MCP  ←→  Any Tool/Data Source
```

Analogy: MCP is to AI agents what HTTP is to the web — a universal protocol that anyone can implement, enabling interoperability.

---

## The Problem MCP Solves

**Before MCP:**
```
LangChain app → custom Google Search integration
              → custom SQL integration
              → custom GitHub integration
              → ...

Each integration: different code, different auth, different format
```

**With MCP:**
```
Any MCP client → MCP Protocol → Google Search MCP Server
                              → SQL MCP Server
                              → GitHub MCP Server
                              → ...

One standard. Write once, use anywhere.
```

---

## Architecture

MCP has three components:

**MCP Host:** the application that uses an LLM (e.g., Claude Desktop, your custom agent)
**MCP Client:** runs inside the host, manages connections to servers
**MCP Server:** a separate process that exposes capabilities via the MCP protocol

```
[Your Application / Claude Desktop]
    [MCP Client]
         ↕ MCP Protocol (JSON-RPC over stdio/SSE)
    [MCP Server A] — e.g., filesystem access
    [MCP Server B] — e.g., database queries
    [MCP Server C] — e.g., GitHub API
```

Servers run as separate processes (or remote services). The protocol is transport-agnostic: currently supports stdio (local) and HTTP+SSE (remote).

---

## What MCP Servers Can Expose

### 1. Tools
Functions the LLM can call:
```json
{
  "name": "read_file",
  "description": "Read the contents of a file from the filesystem",
  "inputSchema": {
    "type": "object",
    "properties": {
      "path": {"type": "string", "description": "File path to read"}
    },
    "required": ["path"]
  }
}
```

### 2. Resources
Data sources the LLM can read (like files, database records, API responses):
```
resource://filesystem/home/user/document.txt
resource://database/customers/id/12345
```

### 3. Prompts
Pre-built prompt templates the host can offer to users:
```json
{
  "name": "summarize_document",
  "description": "Summarize a document with a specified style",
  "arguments": [
    {"name": "document", "description": "The document to summarize"},
    {"name": "style", "description": "Summary style: brief, detailed, bullets"}
  ]
}
```

---

## The Tool Call Flow

When an LLM is connected to an MCP server, tool use works like this:

```
1. User: "What files are in my /tmp directory?"

2. LLM thinks: I need to list files. I have a `list_directory` tool.
   → LLM outputs a tool call request (structured JSON)

3. MCP Client: receives tool call, routes to the right MCP Server

4. MCP Server: executes list_directory("/tmp"), returns result

5. MCP Client: returns result to LLM as a tool result message

6. LLM: reads tool result, incorporates into response
   → "Here are the files in /tmp: file1.txt, file2.py, ..."
```

The LLM never directly executes anything — it requests, the server acts, the result comes back.

---

## Building an MCP Server (Python)

```python
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types

server = Server("my-server")

@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="get_weather",
            description="Get current weather for a city",
            inputSchema={
                "type": "object",
                "properties": {
                    "city": {"type": "string"}
                },
                "required": ["city"]
            }
        )
    ]

@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    if name == "get_weather":
        city = arguments["city"]
        # call weather API
        result = fetch_weather(city)
        return [types.TextContent(type="text", text=result)]

async def main():
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())
```

---

## Why MCP Matters for AI Engineering

1. **Composability:** build once, use in any MCP-compatible client (Claude Desktop, custom agents, IDEs)
2. **Security:** servers run in separate processes with explicit permissions — the LLM cannot bypass the server's access controls
3. **Ecosystem:** a growing library of pre-built MCP servers (filesystem, GitHub, Postgres, Slack, etc.)
4. **Standard interface:** forces you to define clear tool contracts — good engineering practice

---

## Key Concepts

| Term | Meaning |
|------|---------|
| Host | The application embedding the LLM |
| Client | The MCP protocol handler inside the host |
| Server | The separate process exposing tools/resources |
| Tool | A callable function exposed by a server |
| Resource | A readable data source exposed by a server |
| Transport | How client and server communicate (stdio, SSE) |
