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


def run_variant(cfg, out_path: Path, steps: int, label: str):
    lux_cfg = dict(cfg.get("tts", {}).get("luxtts", {}))
    lux_cfg["num_steps"] = steps
    luxtts_tts.speak(
        "Hello from LuxTTS. This is a quick quality A/B test.",
        model_id_or_dir=str(lux_cfg.get("model_id") or "YatharthS/LuxTTS"),
        reference_wav=ROOT / (lux_cfg.get("reference_voice_wav") or ""),
        out_wav=out_path,
        device=lux_cfg.get("device", "cuda"),
        num_steps=int(lux_cfg.get("num_steps", steps)),
        timeout_sec=int(lux_cfg.get("timeout_sec", 20)),
        tts_cfg=lux_cfg,
        verbose=True,
    )
    print(f"[LuxTTS] wrote {label}: {out_path}")
    play_wav_windows(out_path)


def main():
    cfg_path = ROOT / "config" / "default.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    out_dir = ROOT / "data" / "cache"
    out_dir.mkdir(parents=True, exist_ok=True)

    run_variant(cfg, out_dir / "tts_lux_fast.wav", 4, "fast")
    run_variant(cfg, out_dir / "tts_lux_quality.wav", 6, "quality")
    run_variant(cfg, out_dir / "tts_lux_max.wav", 8, "max")


if __name__ == "__main__":
    main()
