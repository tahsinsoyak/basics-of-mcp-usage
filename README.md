# Basics of MCP Usage

This project demonstrates how to use the Model Context Protocol (MCP) to create and interact with AI-powered tools. It includes implementations in both JavaScript (Node.js) and Python, showcasing different use cases.

## Overview

The Model Context Protocol (MCP) is a standardized way for AI models to interact with tools and resources. This project provides examples of:

1. **JavaScript MCP Server**: A basic implementation with a simple arithmetic tool
2. **Python MCP Server**: A more advanced implementation with document retrieval capabilities

## Features

- **MCP Server Implementation**: Examples in both Node.js and Python
- **Simple Tool Example**: Basic arithmetic operations
- **Advanced RAG Tool**: Retrieval Augmented Generation for documentation queries
- **Document Processing**: Loading, parsing, and vectorizing documentation
- **Resource Provisioning**: Providing resources via MCP

## Technologies Used

- **Model Context Protocol (MCP)**
  - JavaScript: `@modelcontextprotocol/sdk`
  - Python: `mcp` library
- **JavaScript / Node.js**
  - `zod` for schema validation
- **Python**
  - `FastMCP`
  - `Langchain` for document processing
  - `BeautifulSoup4` for HTML parsing
  - `OpenAI` for embeddings

## Project Structure

```
├── [index.js](index.js)               # JavaScript MCP server
├── [langgraph-mpc.py](langgraph-mpc.py)       # Python MCP server
├── [load_documents.py](load_documents.py)      # Document processing script
├── [sklearn_vectorstore.parquet](sklearn_vectorstore.parquet) # Vector store for document retrieval
```
## Setup and Running

### Prerequisites

- Node.js and npm (for JavaScript example)
- Python 3.x and pip (for Python examples)
- OpenAI API Key (for Python example)

### JavaScript Example

1. **Install Dependencies**:
   ```bash
   npm install
   ```

2. **Run the Server**:
   ```bash
   node index.js
   ```

### Python Example

1. **Install Dependencies**:
   ```bash
   pip install mcp langchain-openai langchain-community scikit-learn beautifulsoup4 tiktoken
   ```

2. **Set up OpenAI API Key**:
   Ensure your `OPENAI_API_KEY` environment variable is set.

3. **Prepare Documentation & Vector Store**:
   ```bash
   python load_documents.py
   ```

4. **Run the MCP Server**:
   ```bash
   python langgraph-mpc.py
   ```

## Usage Examples

### JavaScript `add` Tool

```javascript
// Example client usage
const result = await client.runTool("add", { a: 5, b: 3 });
console.log(result); // { content: [{ type: "text", text: "8" }] }
```

### Python `langgraph_query_tool`

```python
# Example client usage
result = client.run_tool("langgraph_query_tool", {"query": "How do I create a graph in LangGraph?"})
print(result)  # Returns relevant documentation snippets
```

## Contributing

Contributions are welcome! Feel free to fork this repository, make improvements, and submit pull requests.

## License

[MIT License](LICENSE)
