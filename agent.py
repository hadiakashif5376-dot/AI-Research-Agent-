    def _run(self, query: str) -> str:
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=3))

            if not results:
                return f"No search results found for '{query}'."

            formatted = []
            for i, r in enumerate(results, start=1):
                snippet = (r.get("body") or "")[:200]
                formatted.append(
                    f"{i}. {r.get('title')}\n"
                    f"   Link: {r.get('href')}\n"
                    f"   Snippet: {snippet}\n"
                )
            return "\n".join(formatted)

        except Exception as e:
            return f"Search failed with error: {e}"
