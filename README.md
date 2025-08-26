# Microsoft Learn MCP Wrapper using FASTMCP

A high-performance CLI client for Microsoft Learn documentation using the Model Context Protocol (MCP). Features optimized Unicode handling, flexible output formats, and comprehensive console formatting.

## 🚀 Features

- **🔍 Microsoft Learn Search**: Query the latest Microsoft documentation with semantic search
- **📄 Document Fetching**: Retrieve complete documentation pages by URL
- **🧹 Unicode Normalization**: Advanced Unicode escape sequence handling for clean text
- **📋 MCP Compliance**: Full Model Context Protocol specification compliance
- **🎯 Flexible Output**: Human-readable console formatting with machine-readable raw mode
- **⚡ Streaming Support**: NDJSON streaming for large result sets
- **🔧 Self-Testing**: Built-in validation and testing capabilities

## 📦 Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/FastMCP.git
   cd FastMCP
   ```

2. **Create and activate virtual environment**:
   ```bash
   python -m venv .venv
   
   # Windows
   .venv\Scripts\Activate.ps1
   
   # Linux/Mac
   source .venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 🎯 Usage

> **Note**: The command-line interface shown below is primarily for testing and verification purposes. In production, this tool is designed to be integrated into LLM frameworks (LangChain, Ollama, etc.) or used as an MCP server. See the [MCP Integration section](#-mcp-integration-with-llm-frameworks) for proper usage patterns.

### Basic Search
```bash
python mcp_client/microsoft_docs_mcp.py --q "azure container apps" --k 5
```

### Fetch Specific Document
```bash
python mcp_client/microsoft_docs_mcp.py --url "https://learn.microsoft.com/en-us/azure/container-apps/overview"
```

### List Available Tools
```bash
python mcp_client/microsoft_docs_mcp.py --list-tools
```

### Raw JSON Output (for machine processing)
```bash
python mcp_client/microsoft_docs_mcp.py --q "python" --raw
```

### Different Output Formats
```bash
# Default: NDJSON streaming (best for large results)
python mcp_client/microsoft_docs_mcp.py --q "azure"

# Envelope format (single JSON response)
python mcp_client/microsoft_docs_mcp.py --q "azure" --envelope

# Raw format (preserve escape sequences)
python mcp_client/microsoft_docs_mcp.py --q "azure" --raw
```

### Self-Test
```bash
python mcp_client/microsoft_docs_mcp.py --self-test
```

## 🔧 Command Line Options

| Option | Description | Example |
|--------|-------------|---------|
| `--q, --query` | Search query string | `--q "azure functions"` |
| `--url` | Fetch specific documentation URL | `--url "https://learn.microsoft.com/..."` |
| `--k` | Number of results to return (default: 10) | `--k 5` |
| `--list-tools` | List available MCP tools | `--list-tools` |
| `--self-test` | Run built-in validation tests | `--self-test` |
| `--raw` | Output raw JSON without console formatting | `--raw` |
| `--ndjson` | Force NDJSON streaming output | `--ndjson` |
| `--envelope` | Force envelope (single JSON) output | `--envelope` |

## 📊 Output Formats

### 1. Default (Human-Readable Console)
Clean, formatted text with proper line breaks and readable characters:
```
Visual Studio is a powerful Python IDE on Windows.
Python supports all manner of development...
```

### 2. Raw Mode (`--raw`)
Preserves all escape sequences for machine processing:
```json
{"content": "Visual Studio is a powerful Python IDE on Windows.\\nPython supports..."}
```

### 3. NDJSON Streaming (Default)
Best for large result sets. Each result is a separate JSON line:
```json
{"schema_version": "1.0", "tool": "microsoft_docs_search", "result": {...}}
{"schema_version": "1.0", "tool": "microsoft_docs_search", "result": {...}}
```

### 4. Envelope Format (`--envelope`)
Single JSON response containing all results:
```json
{
  "schema_version": "1.0",
  "tool": "microsoft_docs_search",
  "success": true,
  "result": {
    "matches": [...]
  }
}
```

## 🧪 Testing

Run the comprehensive self-test suite:
```bash
python mcp_client/microsoft_docs_mcp.py --self-test
```

This validates:
- Unicode normalization (0 escape sequences in output)
- MCP tool functionality
- Network connectivity
- Response format compliance

## 🏗️ Technical Architecture

### Core Components

1. **`mcp_client/microsoft_docs_mcp.py`** - Comprehensive CLI interface with full MCP client functionality
2. **Unicode Normalization** - Advanced escape sequence processing using `codecs.decode()`
3. **Console Formatting** - Human vs machine-readable output modes
4. **Flexible Output** - NDJSON streaming, envelope, and raw modes

### MCP Integration

- **Endpoint**: `https://learn.microsoft.com/api/mcp`
- **Protocol**: Model Context Protocol (MCP) JSON-RPC 2.0
- **Tools**: `microsoft_docs_search`, `microsoft_docs_fetch`

### Unicode Processing
- Uses `codecs.decode()` for optimal performance
- Handles double-escaped sequences (`\\u0027` → `'`)
- Recursive normalization of nested data structures
- Function-level processing ensures clean LLM consumption

### Output Formatting
- **Default**: Converts `\n`, `\t`, `\r` to actual formatting for readability
- **Raw**: Preserves literal escape sequences for machine processing
- **Streaming**: NDJSON for memory-efficient processing of large datasets
- **Envelope**: Single JSON response format for simple integrations

## 📝 Examples

### Search with Console Formatting
```bash
python mcp_client/microsoft_docs_mcp.py --q "azure functions" --k 3
```
Output: Clean, readable text with proper line breaks and formatting

### Machine Processing Pipeline
```bash
python mcp_client/microsoft_docs_mcp.py --q "azure functions" --k 3 --raw | jq '.result.value.snippet'
```
Output: Raw JSON with preserved escape sequences, perfect for parsing

### Fetch Complete Documentation
```bash
python mcp_client/microsoft_docs_mcp.py --url "https://learn.microsoft.com/en-us/azure/azure-functions/functions-overview"
```

### Integration Example
```bash
# Get structured data for processing
python mcp_client/microsoft_docs_mcp.py --q "python azure sdk" --envelope --raw > results.json

# Human-readable summary
python mcp_client/microsoft_docs_mcp.py --q "python azure sdk" --k 3
```

## 🏗️ Project Structure

```
FastMCP/
├── README.md               # This comprehensive guide
├── requirements.txt        # Python dependencies
├── .gitignore             # Git ignore patterns
├── .venv/                 # Virtual environment (gitignored)
├── .vscode/               # VS Code configuration
└── mcp_client/            # Main client implementation
    └── microsoft_docs_mcp.py    # Primary CLI interface
```

## 🔗 Dependencies

- **`fastmcp`** - MCP client library for protocol communication
- **Standard library** - `argparse`, `json`, `codecs`, `asyncio` for core functionality

## 🤝 Integration & Usage Patterns

### CLI Interface
Direct command-line usage for manual queries and automation scripts.

### Python Module Integration
```python
import asyncio
from mcp_client.microsoft_docs_mcp import microsoft_search_docs

# Programmatic usage
result = asyncio.run(microsoft_search_docs("azure functions", k=5))
```

### API Gateway Integration
JSON output suitable for API responses and microservice architectures.

### CI/CD Documentation Validation
Use in build pipelines to validate documentation references.

## 🔗 MCP Integration with LLM Frameworks

This FastMCP client can be integrated into various Python LLM frameworks to provide Microsoft Learn documentation access as a tool for AI agents.

### 🦙 Integration with Ollama + LangChain

#### 1. Install Dependencies
```bash
pip install langchain-community langchain-ollama fastmcp
```

#### 2. Create MCP Tool Wrapper
```python
import asyncio
from langchain.tools import BaseTool
from typing import Optional, Type
from pydantic import BaseModel, Field
from mcp_client.microsoft_docs_mcp import microsoft_search_docs, microsoft_fetch_url

class MicrosoftSearchInput(BaseModel):
    query: str = Field(description="Search query for Microsoft Learn documentation")
    k: int = Field(default=5, description="Number of results to return")

class MicrosoftFetchInput(BaseModel):
    url: str = Field(description="Microsoft Learn documentation URL to fetch")

class MicrosoftSearchTool(BaseTool):
    name = "microsoft_docs_search"
    description = "Search Microsoft Learn documentation for the latest information on Azure, .NET, Microsoft 365, and other Microsoft technologies"
    args_schema: Type[BaseModel] = MicrosoftSearchInput

    def _run(self, query: str, k: int = 5) -> str:
        """Execute the search and return results"""
        try:
            result = asyncio.run(microsoft_search_docs(query, k=k, raw_output=True))
            return str(result.get('structuredContent', result))
        except Exception as e:
            return f"Error searching Microsoft docs: {str(e)}"

class MicrosoftFetchTool(BaseTool):
    name = "microsoft_docs_fetch"
    description = "Fetch complete Microsoft Learn documentation page content"
    args_schema: Type[BaseModel] = MicrosoftFetchInput

    def _run(self, url: str) -> str:
        """Fetch the URL and return content"""
        try:
            result = asyncio.run(microsoft_fetch_url(url))
            return str(result.get('structuredContent', result))
        except Exception as e:
            return f"Error fetching Microsoft docs: {str(e)}"
```

#### 3. Set Up Ollama Agent with Tools
```python
from langchain_ollama import OllamaLLM
from langchain.agents import create_react_agent, AgentExecutor
from langchain.prompts import PromptTemplate

# Initialize Ollama model
llm = OllamaLLM(model="llama3.2")

# Create tools
tools = [
    MicrosoftSearchTool(),
    MicrosoftFetchTool()
]

# Create agent prompt
prompt = PromptTemplate.from_template("""
You are an AI assistant with access to Microsoft Learn documentation.
You can search for information and fetch specific documentation pages.

Tools available:
- microsoft_docs_search: Search Microsoft Learn for topics
- microsoft_docs_fetch: Get complete content from a specific URL

Question: {input}
{agent_scratchpad}
""")

# Create and run agent
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Example usage
response = agent_executor.invoke({
    "input": "How do I deploy an Azure Container App using CLI?"
})
```

### 🚀 Integration with FastMCP Server Pattern

#### 1. Create a Custom MCP Server
```python
from fastmcp import FastMCP
import asyncio
from mcp_client.microsoft_docs_mcp import microsoft_search_docs, microsoft_fetch_url

# Create FastMCP server
mcp = FastMCP("Microsoft Learn Assistant")

@mcp.tool()
def search_microsoft_docs(query: str, max_results: int = 5) -> dict:
    """Search Microsoft Learn documentation for the given query"""
    result = asyncio.run(microsoft_search_docs(query, k=max_results, raw_output=True))
    return result.get('structuredContent', result)

@mcp.tool()
def fetch_microsoft_doc(url: str) -> dict:
    """Fetch complete content from a Microsoft Learn documentation URL"""
    result = asyncio.run(microsoft_fetch_url(url))
    return result.get('structuredContent', result)

if __name__ == "__main__":
    mcp.run()
```

#### 2. Connect to Any MCP-Compatible LLM Client
```bash
# Run your custom MCP server
python microsoft_learn_server.py

# Connect from Claude Desktop, Continue.dev, or any MCP client
# Add to your MCP client configuration:
{
  "microsoft_learn": {
    "command": "python",
    "args": ["path/to/microsoft_learn_server.py"]
  }
}
```

### 🤖 Direct Integration with OpenAI/Anthropic APIs

#### Using with OpenAI Function Calling
```python
import openai
import asyncio
from mcp_client.microsoft_docs_mcp import microsoft_search_docs

client = openai.OpenAI()

def search_docs_function(query: str, k: int = 5):
    """Function for OpenAI to call for Microsoft docs search"""
    result = asyncio.run(microsoft_search_docs(query, k=k, raw_output=True))
    return result.get('structuredContent', result)

# Define function schema for OpenAI
functions = [{
    "name": "search_microsoft_docs",
    "description": "Search Microsoft Learn documentation",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
            "k": {"type": "integer", "description": "Number of results", "default": 5}
        },
        "required": ["query"]
    }
}]

# Chat with function calling
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "How do I use Azure Functions with Python?"}],
    functions=functions,
    function_call="auto"
)
```

### 🎯 Universal MCP Client Integration

For any MCP-compatible framework:

#### 1. Install the Universal MCP Python SDK
```bash
pip install mcp
```

#### 2. Use as Standard MCP Client
```python
from mcp import Client
import asyncio

async def use_microsoft_learn_mcp():
    # This client can be used with any MCP server implementation
    async with Client("path/to/mcp_client/search_fetch.py") as client:
        # List available tools
        tools = await client.list_tools()
        
        # Call Microsoft docs search
        result = await client.call_tool("microsoft_docs_search", {
            "query": "azure container apps",
            "k": 3
        })
        
        return result

# Usage
result = asyncio.run(use_microsoft_learn_mcp())
```

### 📋 Integration Checklist

- ✅ **Ollama + LangChain**: Local AI with Microsoft docs access
- ✅ **FastMCP Server**: Create reusable MCP server for any client
- ✅ **OpenAI Function Calling**: Cloud AI with documentation tools
- ✅ **Universal MCP**: Compatible with Claude Desktop, Continue.dev, etc.
- ✅ **Custom Frameworks**: Easy integration via Python imports

### 🔧 Configuration Tips

1. **Environment Variables**: Set API keys and endpoints in `.env` files
2. **Rate Limiting**: Implement rate limiting for production usage
3. **Caching**: Add response caching for frequently accessed docs
4. **Error Handling**: Implement robust error handling for network issues
5. **Logging**: Add comprehensive logging for debugging and monitoring

## 🔧 Troubleshooting

### Common Issues

**Unicode escape sequences in output**:
- Use default mode (not `--raw`) for human-readable output
- Ensure you're using the latest version with Unicode normalization

**Connection errors**:
- Check internet connectivity
- Verify MCP endpoint is accessible: `https://learn.microsoft.com/api/mcp`
- Run self-test: `python mcp_client/microsoft_docs_mcp.py --self-test`

**Large result sets**:
- Use `--k` to limit results for faster processing
- NDJSON streaming mode handles large datasets memory-efficiently
- Consider `--envelope` for smaller, complete responses

**Performance optimization**:
- Use `--raw` mode for machine processing to skip console formatting
- NDJSON streaming reduces memory usage for large result sets
- Unicode normalization is optimized with single-pass processing

### Debug Mode
For detailed debugging, redirect stderr to see internal processing:
```bash
python mcp_client/microsoft_docs_mcp.py --q "test" 2>&1
```

### Validation
Verify your installation and functionality:
```bash
# Quick functionality test
python mcp_client/microsoft_docs_mcp.py --self-test

# Test search with small result
python mcp_client/microsoft_docs_mcp.py --q "azure" --k 1

# Test different output modes
python mcp_client/microsoft_docs_mcp.py --q "test" --raw
python mcp_client/microsoft_docs_mcp.py --q "test" --envelope
```

## 📈 Performance Features

- **Unicode Processing**: Single-pass `codecs.decode()` for optimal performance
- **Streaming**: NDJSON mode for memory-efficient large result processing
- **Caching**: Efficient MCP client connection management
- **Output**: Console formatting applied only when needed (skipped in `--raw` mode)
- **Network**: Async MCP client for non-blocking requests
- **Streamlined Architecture**: Single comprehensive CLI tool for all operations

## 🧹 Project Cleanup

This project has been optimized for clarity and maintainability:
- **Single Tool**: Consolidated all functionality into one comprehensive CLI
- **Standard Structure**: Follows Python project conventions with root-level dependencies
- **Clean Codebase**: Removed redundant utilities and duplicate configuration files
- **Comprehensive Documentation**: All information centralized in this README

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes
4. Run tests: `python mcp_client/microsoft_docs_mcp.py --self-test`
5. Test functionality: `python mcp_client/microsoft_docs_mcp.py --q "test" --k 1`
6. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🏷️ Version

Current version features:
- ✅ Optimized Unicode handling with `codecs.decode()`
- ✅ MCP JSON-RPC 2.0 specification compliance
- ✅ Console formatting with `--raw` override
- ✅ NDJSON streaming for large datasets
- ✅ Comprehensive self-testing
- ✅ Zero Unicode escape sequences in default output

---

**Built with ❤️ for the Microsoft Learn community**

*For questions, issues, or contributions, please visit the [GitHub repository](https://github.com/yourusername/FastMCP).*
