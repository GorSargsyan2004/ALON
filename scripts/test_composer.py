import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from services.orchestrator.response_composer import compose_weather_reply, compose_search_reply


def run_tests():
    weather_examples = [
        "Naah, the weather is not nice, cloudy, can you look at the weather today?",
        "chance to rain 88 percent, fuck…",
    ]
    tool_data = {
        "location": "Yerevan",
        "day": "today",
        "summary": "Today in Yerevan: low 2°C, high 7°C, precipitation chance up to 88%, wind up to 9 km/h.",
    }

    for text in weather_examples:
        out = compose_weather_reply(text, tool_data, "Alon", "", "Asia/Yerevan")
        assert "http" not in out["spoken_text"].lower()
        assert out["spoken_text"]

    search_text = "search internet facebook link of my sister Gohar Sargsyan"
    search_summary = "I found several possible matches."
    links = [
        {"title": "Gohar Sargsyan - Facebook", "url": "https://facebook.com/..."},
        {"title": "Gohar Sargsyan - Instagram", "url": "https://instagram.com/..."},
    ]
    out = compose_search_reply(search_text, search_summary, links, "Alon", "", max_links=5)
    assert "http" not in out["spoken_text"].lower()
    assert "(" in out["display_text"] and ")" in out["display_text"]

    print("All composer checks passed.")


if __name__ == "__main__":
    run_tests()
