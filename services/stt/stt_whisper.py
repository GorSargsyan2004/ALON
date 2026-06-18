import subprocess
from pathlib import Path

import sounddevice as sd
import scipy.io.wavfile as wav

ROOT = Path(__file__).resolve().parents[2]

WHISPER_CLI = ROOT / "services" / "stt" / "whispercpp" / "whisper-cli.exe"
MODEL = ROOT / "models" / "stt" / "whisper.cpp" / "ggml-medium.en-q5_0.bin"
TEMP_WAV = ROOT / "data" / "cache" / "mic.wav"

def record(seconds: int = 5, samplerate: int = 16000):
    print("🎤 Recording... speak now")
    audio = sd.rec(int(seconds * samplerate), samplerate=samplerate, channels=1, dtype="int16")
    sd.wait()
    TEMP_WAV.parent.mkdir(parents=True, exist_ok=True)
    wav.write(str(TEMP_WAV), samplerate, audio)
    print(f"Saved: {TEMP_WAV}")

def transcribe() -> str:
    # whisper-cli prints transcription to stdout
    cmd = [
        str(WHISPER_CLI),
        "-m", str(MODEL),
        "-f", str(TEMP_WAV),
        "--no-timestamps",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"whisper-cli failed:\n{r.stderr}")
    return r.stdout.strip()

if __name__ == "__main__":
    record(5)
    text = transcribe()
    print("\nYou said:")
    print(text)
