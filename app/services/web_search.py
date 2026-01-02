import httpx


async def web_search(query: str) -> str:
    """
    Minimal web search fallback.
    Replace with SerpAPI / Tavily / Bing later.
    """
    async with httpx.AsyncClient(timeout=10) as client:
        response = await client.get(
            "https://api.duckduckgo.com/",
            params={
                "q": query,
                "format": "json",
                "no_redirect": 1
            }
        )

    data = response.json()
    return data.get("AbstractText", "")
