from services.orchestrator.prompt_builder import build_brain_messages, enforce_budget


def main():
    system_style = "You are Alon."
    user_text = "Explain the formula."
    memory = "MEMORY\n" + ("A" * 12000)
    tool_results = "TOOL_RESULTS\n" + ("B" * 12000)

    messages = build_brain_messages(system_style, user_text, memory, tool_results)
    capped = enforce_budget(
        messages,
        max_chars_total=16000,
        max_chars_tool_results=2000,
        max_chars_memory=2000,
    )
    total = sum(len(m.get("content") or "") for m in capped)
    print("Total chars:", total)
    for m in capped:
        print("---", m["role"])
        print(m["content"][:200])


if __name__ == "__main__":
    main()
