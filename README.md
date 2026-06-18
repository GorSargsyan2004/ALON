# ALON

Local Windows AI assistant with LLM routing, tools, and TTS.

**Project Structure**
```text
ALON/
├─ apps/
│  ├─ alon-cli/
│  └─ alon-desktop/        (future)
├─ services/
│  ├─ orchestrator/        (main brain loop)
│  ├─ stt/                 (Whisper.cpp)
│  ├─ tts/                 (LuxTTS, Piper fallback)
│  ├─ wakeword/            (planned)
│  └─ tools/
│     ├─ weather/
│     ├─ websearch/
│     ├─ filesystem/       (planned)
│     └─ executor/         (planned, sandboxed)
├─ models/
│  ├─ llm/                 (Ollama-managed)
│  └─ tts/
│     └─ luxtts/voices/
├─ data/
│  ├─ memory/
│  └─ cache/
├─ config/
│  └─ default.yaml
├─ scripts/
└─ logs/

```

**Objectives**

Done ✅
- Core structure and folder architecture created.
- Ollama installed and running locally.
- `llama3.1:8b` pulled and responding via API.
- Orchestrator can route `assist` to LLM and return responses.
- JSONL logging working (`session_start`/`turn_start`/`turn_end`/`tool_call`/`tool_result`).
- Weather tool enabled (geocode + forecast + conditional “what to wear”).
- Web search tool working (DDGS fixed; summarizing works).
- Piper TTS worked initially.
- LuxTTS integrated and usable + fast.
- Reference voice cleaning improved.
- Text preprocessing plan (contractions + stage-direction mood switching) accepted.
- Silence trimming idea validated.

Remaining to do 🔜
1. STT (Speech-to-text) 🎤
2. Wake word (optional)
3. Filesystem navigation tool 📁
4. Executor tool (sandboxed) 🧨
5. Better routing / tool selection (LLM-driven)
6. Memory (short-term + long-term)

**Details**

1) STT (Speech-to-text) 🎤  
You have:
- whisper.cpp binaries downloaded.
- STT python script exists but failed earlier due to path/exe wiring and no mic.

Remaining:
- Fix `services/stt/stt_whisper.py` to call `whisper-cli.exe` (not `whisper.exe`).
- Confirm mic recording pipeline works once mic arrives.
- Add push-to-talk or auto-record logic (duration/VAD).
- Return transcribed text back into orchestrator loop.

Testing needed:
- mic capture → wav → whisper transcription → text printed.

2) Wake word (optional)  
Remaining:
- openWakeWord integration (always-on listener).
- Wake word triggers recording + STT + response + TTS.

Testing needed:
- wake word triggers reliably without false positives.

3) Filesystem navigation tool 📁  
Remaining:
- Implement safe read-only browsing:
- list directories
- read small text files
- search filenames / grep text
- block sensitive paths by allowlist rules
- Optional: safe write actions behind confirmation (“are you sure?”).

Testing needed:
- request → tool call → correct constrained output → logged.

4) Executor tool (sandboxed) 🧨  
Remaining:
- Implement locked-down executor (allowlist commands only).
- Timeouts, cwd restriction, no dangerous commands.
- Always log what was run.

Testing needed:
- safe command allowed, unsafe command refused.

5) Better routing / tool selection (LLM-driven)  
Remaining:
- Improve router intent detection.
- Add structured tool-argument formatting by LLM (started for weather/search).
- Add tool-result summarization by LLM (search close).

Testing needed:
- tricky prompts still choose correct tool 90%+ of time.

6) Memory  
Remaining:
- Add short-term memory (conversation context window).
- Add long-term memory (facts, preferences) with a local store.
- Retrieval: inject relevant memories into LLM prompt.

Testing needed:
- “My name is Gor” → later “what’s my name?” works.
