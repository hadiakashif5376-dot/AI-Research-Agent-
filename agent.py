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
        "Use this whenever you need current, factual, or web-based information. "
        "This is the ONLY tool available — you cannot open URLs, fetch pages, "
        "or read files directly. Work only from the search snippets returned."
    )
    args_schema: type[BaseModel] = DuckDuckGoSearchInput

    def _run(self, query: str) -> str:
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=3))

            if not results:
                return f"No search results found for '{query}'."

            formatted = []
            for i, r in enumerate(results, start=1):
                snippet = (r.get("body") or "")[:280]
                formatted.append(
                    f"{i}. {r.get('title')}\n"
                    f"   Link: {r.get('href')}\n"
                    f"   Snippet: {snippet}\n"
                )
            return "\n".join(formatted)

        except Exception as e:
            return f"Search failed with error: {e}"


def build_research_crew(topic: str, groq_api_key: str, report_length: str = "detailed") -> Crew:
    """Builds a single-agent CrewAI research crew for the given topic.

    report_length: "quick" for a short summary, "detailed" for a fuller report.
    """

    is_quick = report_length == "quick"

    llm = LLM(
        model="groq/openai/gpt-oss-120b",
        api_key=groq_api_key,
        temperature=0.4,
        max_tokens=700 if is_quick else 1800,
    )

    search_tool = DuckDuckGoSearchTool()

    researcher = Agent(
        role="Senior Research Analyst",
        goal=f"Research '{topic}' and produce a polished, well-structured report.",
        backstory=(
            "You are a professional research analyst who writes reports for "
            "executives. You always ground claims in search results, use "
            "tables when comparing data points, and cite the source name "
            "next to any figure you mention. You never fabricate statistics "
            "or citation markers you can't back up.\n\n"
            "IMPORTANT: You have exactly ONE tool available: 'DuckDuckGo Web "
            "Search'. You cannot open URLs, browse pages, or open files — "
            "only run search queries and read the returned snippets. Never "
            "attempt to call any tool other than 'DuckDuckGo Web Search'."
        ),
        tools=[search_tool],
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=3,
    )

    if is_quick:
        search_instruction = "Use the DuckDuckGo Web Search tool (1-2 focused searches)."
        body_instructions = (
            "1. A '## Overview' section (1-2 sentences)\n"
            "2. A '## Key Findings' section with 3-4 bullet points. Bold "
            "the headline of each bullet.\n"
            "3. A '## Conclusion' section (1 sentence).\n"
        )
    else:
        search_instruction = (
            "Use the DuckDuckGo Web Search tool with 3-4 different, "
            "specific search queries covering different angles of the "
            "topic (e.g. current state, statistics/data, expert opinions, "
            "future outlook). Do not stop after one search — a thin, "
            "shallow report is not acceptable for a detailed report."
        )
        body_instructions = (
            "1. A '## Overview' section (3-4 sentences of real background "
            "and context on the topic).\n"
            "2. A '## Key Findings' section with 7-9 bullet points. Bold "
            "the headline of each bullet, then follow it with 2-3 full "
            "sentences of explanation and supporting detail (not just a "
            "one-line headline). Where a search result gave a specific "
            "figure or fact, name the source in parentheses, e.g. "
            "(Source: TechCrunch).\n"
            "3. A '## Data & Trends' section: if the search results include "
            "any comparable data points (market size, statistics, "
            "timelines, percentages), present them in a markdown table "
            "with columns like Metric | Detail | Source. If truly no "
            "numeric data was found, briefly say so instead of inventing "
            "numbers.\n"
            "4. A '## Implications' section (2-3 sentences on what this "
            "means going forward).\n"
            "5. A '## Conclusion' section (2-3 sentences).\n"
        )

    research_task = Task(
        description=(
            f"Research the topic: '{topic}'.\n"
            f"{search_instruction} You cannot open links — work only from "
            "the search snippets. Then write a polished report."
        ),
        expected_output=(
            body_instructions
            + "6. A final '## Sources' section listing each link you actually "
            "used, as a markdown bullet list (e.g. '- [Page title](https://...)'). "
            "Only include links that really appeared in your search results — "
            "never invent a URL, statistic, or citation marker."
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
