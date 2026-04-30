from dotenv import load_dotenv
# import sys
# from pathlib import Path

# Allow `import schemas` when the runner's cwd is not the project root (e.g. IDE "Run file").
# _ROOT = Path(__file__).resolve().parent
# if str(_ROOT) not in sys.path:
#     sys.path.insert(0, str(_ROOT))

load_dotenv()

try:
    from langchain_tavily import TavilySearch
except ImportError as e:
    raise ImportError(
        "langchain_tavily is not installed in this Python environment. "
        "From the project folder run: poetry install — then use "
        "`poetry run python main.py`, or select the `.venv` interpreter in your IDE "
        "(see poetry.toml: in-project virtualenv)."
    ) from e
from langchain_core.tools import StructuredTool
from langgraph.prebuilt import ToolNode
from schemas import AnswerQuestion, ReviseAnswer

tavily_tool = TavilySearch(max_results=5)


def run_queries(search_queries: list[str], **kwargs):
    """Run web search for each query via Tavily and return combined results."""
    return tavily_tool.batch([{"query": query} for query in search_queries])

execute_tools = ToolNode(
        [
            StructuredTool.from_function(
                run_queries,
                name=AnswerQuestion.__name__,
            ),
            StructuredTool.from_function(
                run_queries,
                name=ReviseAnswer.__name__,
            ),
        ]
    )