import sys
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from services.tts import luxtts_tts


SENTENCES = [
    "Hey there! How's it going?",
    "Thanks for reaching out. I'm here to help.",
    "I'll keep this short and clear.",
]


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
    lux_cfg = dict(tts_cfg.get("luxtts", {}))
    voice_pack = lux_cfg.get("voice_pack") or {}
    voices = voice_pack.get("voices") or {}

    out_dir = ROOT / "data" / "cache"
    out_dir.mkdir(parents=True, exist_ok=True)

    for voice_key in voices.keys():
        for i, text in enumerate(SENTENCES, 1):
            out_wav = out_dir / f"tts_voice_{voice_key}_{i}.wav"
            luxtts_tts.speak(
                text,
                model_id_or_dir=str(lux_cfg.get("model_id") or "YatharthS/LuxTTS"),
                reference_wav=ROOT / (lux_cfg.get("reference_voice_wav") or ""),
                out_wav=out_wav,
                device=lux_cfg.get("device", "cuda"),
                num_steps=int(lux_cfg.get("num_step", 6)),
                timeout_sec=int(lux_cfg.get("timeout_sec", 20)),
                tts_cfg=lux_cfg,
                voice_key=voice_key,
                verbose=True,
            )
            print(f"[LuxTTS] wrote {out_wav}")
            play_wav_windows(out_wav)


if __name__ == "__main__":
    main()
