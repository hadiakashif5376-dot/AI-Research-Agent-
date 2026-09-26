import litellm
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from ddgs import DDGS

# --- Workaround for a known CrewAI bug (GitHub issue #6789) ---
# CrewAI tags messages with an internal "cache_breakpoint" key for prompt
# caching. Native providers strip it automatically, but the LiteLLM
# fallback path (used for Groq) does not, and Groq's API rejects the
# unknown field. This patch removes it before the request is sent.
_original_completion = litellm.completion


def _patched_completion(*args, **kwargs):
    messages = kwargs.get("messages")
    if messages:
        for m in messages:
            if isinstance(m, dict):
                m.pop("cache_breakpoint", None)
    return _original_completion(*args, **kwargs)


litellm.completion = _patched_completion


class DuckDuckGoSearchInput(BaseModel):
    query: str = Field(..., description="The search query to look up on the web.")


class DuckDuckGoSearchTool(BaseTool):
    name: str = "DuckDuckGo Web Search"
    description: str = (
        "Searches the web using DuckDuckGo and returns the top results "
        "(title, link, and short snippet) for a given query. "
        "Use this whenever you need current, factual, or web-based information."
    )
    args_schema: type[BaseModel] = DuckDuckGoSearchInput

    def _run(self, query: str) -> str:
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=5))

            if not results:
                return f"No search results found for '{query}'."

            formatted = []
            for i, r in enumerate(results, start=1):
                formatted.append(
                    f"{i}. {r.get('title')}\n"
                    f"   Link: {r.get('href')}\n"
                    f"   Snippet: {r.get('body')}\n"
                )
            return "\n".join(formatted)

        except Exception as e:
            return f"Search failed with error: {e}"


def build_research_crew(topic: str, groq_api_key: str) -> Crew:
    """Builds a single-agent CrewAI research crew for the given topic."""

    llm = LLM(
        model="groq/openai/gpt-oss-120b",
        api_key=groq_api_key,
        temperature=0.5,
    )

    search_tool = DuckDuckGoSearchTool()

    researcher = Agent(
        role="Senior Research Analyst",
        goal=f"Research '{topic}' thoroughly and produce a clear, well-organized summary.",
        backstory=(
            "You are a meticulous research analyst who always verifies facts "
            "using web search before writing anything. You cite where "
            "information came from and avoid making things up."
        ),
        tools=[search_tool],
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )

    research_task = Task(
        description=(
            f"Research the topic: '{topic}'.\n"
            "Use the DuckDuckGo Web Search tool to gather current, accurate "
            "information. Search multiple angles if needed. Then write a "
            "well-structured report."
        ),
        expected_output=(
            "A markdown report with:\n"
            "1. A short introduction (2-3 sentences)\n"
            "2. 4-6 key bullet points with the most important findings\n"
            "3. A brief conclusion\n"
            "Keep it factual and cite source links where relevant."
        ),
        agent=researcher,
    )

    crew = Crew(
        agents=[researcher],
        tasks=[research_task],
        process=Process.sequential,
        verbose=True,
    )

    return crew
