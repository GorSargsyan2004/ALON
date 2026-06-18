import sys
import subprocess
from pathlib import Path
import difflib
import os

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from services.tts import luxtts_tts


SENTENCES = [
    "Hey there! How's it going?",
    "Please read this line exactly as written.",
    "The quick brown fox jumps over the lazy dog.",
    "LuxTTS should follow the text without adding words.",
    "If the output diverges, increase guidance scale or change the reference clip.",
]


def play_wav_windows(wav_path: Path):
    ps = f"""
Add-Type -AssemblyName presentationCore
$player = New-Object system.media.soundplayer "{wav_path}"
$player.PlaySync()
"""
    subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=False)


def transcribe_with_whisper_cpp(wav_path: Path) -> str | None:
    exe = os.environ.get("WHISPER_CPP_BIN")
    if not exe:
        return None
    out_base = wav_path.with_suffix("")
    cmd = [exe, "-f", str(wav_path), "-otxt", "-of", str(out_base), "-nt"]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        txt_path = out_base.with_suffix(".txt")
        if txt_path.exists():
            return txt_path.read_text(encoding="utf-8").strip()
    except Exception:
        return None
    return None


def transcribe_with_transformers(wav_path: Path) -> str | None:
    try:
        from transformers import pipeline
        import torch
    except Exception:
        return None
    device = 0 if torch.cuda.is_available() else -1
    asr = pipeline("automatic-speech-recognition", model="openai/whisper-base", device=device)
    try:
        res = asr(str(wav_path))
        return (res or {}).get("text", "").strip()
    except Exception:
        return None


def similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()


def main():
    cfg_path = ROOT / "config" / "default.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    tts_cfg = cfg.get("tts", {})
    lux_cfg = dict(tts_cfg.get("luxtts", {}))
    out_dir = ROOT / "data" / "cache"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[LuxTTS] Fidelity test starting...")

    for i, text in enumerate(SENTENCES, 1):
        out_wav = out_dir / f"tts_lux_fidelity_{i}.wav"
        luxtts_tts.speak(
            text,
            model_id_or_dir=str(lux_cfg.get("model_id") or "YatharthS/LuxTTS"),
            reference_wav=ROOT / (lux_cfg.get("reference_voice_wav") or ""),
            out_wav=out_wav,
            device=lux_cfg.get("device", "cuda"),
            num_steps=int(lux_cfg.get("num_step", 6)),
            timeout_sec=int(lux_cfg.get("timeout_sec", 20)),
            tts_cfg=lux_cfg,
            verbose=True,
        )

        transcript = transcribe_with_whisper_cpp(out_wav)
        if transcript is None:
            transcript = transcribe_with_transformers(out_wav)

        if transcript is None:
            print(f"[ASR] Skipped for {out_wav.name} (no ASR available).")
            continue

        score = similarity(text, transcript)
        print(f"[ASR] {out_wav.name} -> {transcript}")
        print(f"[ASR] similarity={score:.2f}")
        if score < 0.6:
            print("[WARN] Low fidelity. Consider guidance_scale 3.5-5.0 and a neutral reference clip.")

        play_wav_windows(out_wav)


if __name__ == "__main__":
    main()
