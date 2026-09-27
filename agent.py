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
                results = list(ddgs.text(query, max_results=2))

            if not results:
                return f"No search results found for '{query}'."

            formatted = []
            for i, r in enumerate(results, start=1):
                snippet = (r.get("body") or "")[:150]
                formatted.append(
                    f"{i}. {r.get('title')}\n"
                    f"   Link: {r.get('href')}\n"
                    f"   Snippet: {snippet}\n"
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
        max_tokens=700,  # caps response length so each call uses less of the 8,000 TPM budget
    )

    search_tool = DuckDuckGoSearchTool()

    researcher = Agent(
        role="Senior Research Analyst",
        goal=f"Research '{topic}' concisely and produce a clear summary.",
        backstory=(
            "You are an efficient research analyst. You search the web ONCE "
            "or twice at most, then write your report immediately. You do "
            "not over-research."
        ),
        tools=[search_tool],
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=3,  # caps tool-call loops so token use per run stays predictable
    )

    research_task = Task(
        description=(
            f"Research the topic: '{topic}'.\n"
            "Do ONE focused DuckDuckGo search (two at most) to gather "
            "current facts, then write the report directly. Do not keep "
            "searching repeatedly."
        ),
        expected_output=(
            "A concise markdown report with:\n"
            "1. A short introduction (2-3 sentences)\n"
            "2. 4-6 key bullet points\n"
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
