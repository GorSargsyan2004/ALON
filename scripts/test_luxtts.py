import sys
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from services.tts import luxtts_tts


def play_wav_windows(wav_path: Path):
    ps = f"""
Add-Type -AssemblyName presentationCore
$player = New-Object system.media.soundplayer "{wav_path}"
$player.PlaySync()
"""
    subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=False)


def main():
    cfg_path = ROOT / "config" / "default.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    tts_cfg = cfg.get("tts", {})
    out_wav = ROOT / tts_cfg.get("output_wav", "data/cache/tts_last.wav")
    lux_cfg = tts_cfg.get("luxtts", {})

    luxtts_tts.speak(
        "Hello from LuxTTS.",
        model_id_or_dir=str(lux_cfg.get("model_id") or "YatharthS/LuxTTS"),
        reference_wav=ROOT / (lux_cfg.get("reference_voice_wav") or ""),
        out_wav=out_wav,
        device=lux_cfg.get("device", "cuda"),
        num_steps=int(lux_cfg.get("num_steps", 4)),
        timeout_sec=int(lux_cfg.get("timeout_sec", 20)),
    )

    play_wav_windows(out_wav)


if __name__ == "__main__":
    main()
