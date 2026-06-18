import re
from dataclasses import dataclass
from typing import Literal

Intent = Literal["assist", "weather", "search", "filesystem", "executor"]

@dataclass
class RouteDecision:
    intent: Intent
    confidence: float
    reason: str

_weather_re = re.compile(
    r"\b(weather|temperature|forecast|rain|snow|wind|humidity|what to wear|what should i wear|clothes|jacket|coat)\b",
    re.I
)
_search_re = re.compile(
    r"\b(search( the web)?|web search|google|look up|lookup|find on the web|news about|latest news|latest on|what's new|current news|breaking)\b",
    re.I
)
_files_re = re.compile(
    r"\b(open file|read file|list files|directory|folder|navigate|go to|show me files|find files|search folder|tail|last lines|desktop|documents)\b",
    re.I
)
_exec_re = re.compile(r"\b(run|execute|powershell|cmd|terminal|script)\b", re.I)

def route(text: str) -> RouteDecision:
    t = text.strip()

    if _weather_re.search(t):
        return RouteDecision("weather", 0.85, "Matched weather keywords")
    if _search_re.search(t):
        return RouteDecision("search", 0.75, "Matched web search keywords")
    if _files_re.search(t):
        return RouteDecision("filesystem", 0.70, "Matched filesystem keywords")
    if _exec_re.search(t):
        return RouteDecision("executor", 0.70, "Matched execution keywords")

    return RouteDecision("assist", 0.60, "Default fallback")
