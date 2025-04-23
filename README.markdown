# Deep Research AI Agentic System

This repository implements a **Deep Research AI Agentic System** for Kairon’s qualifying assignment. The system leverages Tavily for web-based information gathering, LangChain for LLM integration, and LangGraph for agent orchestration. It features a multi-agent architecture with a **Research Agent** for data collection and an **Answer Drafter Agent** for generating structured reports.

## Table of Contents

- Architecture Overview
- Prerequisites
- Setup Instructions
- Usage Examples
- Project Structure
- Testing
- Code Formatting
- Submission Details
- Potential Improvements
- Author

## Architecture Overview

The system is a modular, extensible agentic workflow:

- **Research Agent**: Queries the web using Tavily’s Search API, filters results for relevance, and stores data in a shared state.
- **Answer Drafter Agent**: Synthesizes collected data into a markdown report with Introduction, Findings, and Conclusion sections.
- **LangGraph Workflow**: Orchestrates agents via a stateful graph, supporting iterative refinement and persistence.
- **State Persistence**: Uses SQLite (`langgraph-checkpoint-sqlite`) for debugging and resuming workflows.
- **LLM Backend**: Employs OpenAI’s GPT-4o for reasoning and text generation (configurable for other models).

See `docs/explanation.pdf` for a detailed explanation with architecture diagrams.

## Prerequisites

- **Python**: 3.8+ (3.10+ recommended, compatible with Google Colab).
- **API Keys**:
  - OpenAI API Key
  - Tavily API Key
- **System**: ≥4GB RAM, internet access.

## Setup Instructions

Click to expand

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/your-username/deep-research-agent.git
   cd deep-research-agent
   ```

2. **Create a Virtual Environment**:

   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

   **Dependencies**:

   | Package | Version | Purpose |
   | --- | --- | --- |
   | `langchain` | 0.3.14 | LLM and tool integration |
   | `langchain-core` | 0.3.14 | Core LangChain utilities |
   | `langchain-community` | 0.3.14 | Community tools (e.g., Tavily) |
   | `langchain-openai` | 0.3.0 | OpenAI LLM integration |
   | `tavily-python` | 0.5.0 | Web search via Tavily API |
   | `langgraph` | 0.2.64 | Agent workflow orchestration |
   | `langgraph-checkpoint-sqlite` | 2.0.0 | SQLite state persistence |
   | `rich` | 13.9.2 | Console output formatting |
   | `openai` | 1.51.0 | OpenAI API client |
   | `python-dotenv` | 1.0.1 | Environment variable management |
   | `pytest` | 8.3.3 | Testing framework |
   | `black` | 24.8.0 | Code formatting |

4. **Set Up Environment Variables**: Create a `.env` file:

   ```bash
   echo "OPENAI_API_KEY=your_openai_api_key" > .env
   echo "TAVILY_API_KEY=your_tavily_api_key" >> .env
   ```

5. **Verify Setup**:

   ```bash
   pytest tests/test_agent.py
   ```

## Usage Examples

Click to expand

### Default Query

Run the script with the default query:

```bash
python deep_research_agent.py
```

Query: “Compare CPU and GPU architectures for AI workloads”.

### Custom Query

Edit `deep_research_agent.py`:

```python
if __name__ == "__main__":
    query = "Impact of quantum computing on cryptography"
    run_research(query)
```

Run:

```bash
python deep_research_agent.py
```

## Project Structure

```
deep-research-agent/
├── deep_research_agent.py  # Main script
├── requirements.txt        # Dependencies
├── .env                   # Environment variables (gitignored)
├── README.md              # This file
├── tests/                 # Test scripts
│   └── test_agent.py      # Unit tests
├── docs/                  # Documentation
│   └── explanation.pdf    # Detailed explanation
└── .gitignore             # Excludes .env, __pycache__, etc.
```

## Testing

Run tests:

```bash
pytest tests/test_agent.py
```

Tests verify that `run_research` produces a non-empty draft.

## Author

\[Your Name\]\
Licensed under MIT.
