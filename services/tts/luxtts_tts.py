from __future__ import annotations

import time
import io
from pathlib import Path
from contextlib import contextmanager, redirect_stdout, redirect_stderr
from typing import Any, Dict, Optional, Tuple

# LuxTTS setup (repo cloned at ALON/LuxTTS and installed editable):
#   python -m pip install -r LuxTTS/requirements.txt
#   python -m pip install -e LuxTTS
# Key deps: torch, transformers, librosa, torchaudio, soundfile, numpy

from zipvoice.luxvoice import (
    LuxTTS,
    load_models_gpu,
    load_models_cpu,
    process_audio,
    generate,
    generate_cpu,
)

_LUX_CACHE: Dict[Tuple[str, str], Dict[str, Any]] = {}
_VOICE_CACHE: Dict[Tuple[str, str], Dict[str, Any]] = {}

PROMPT_DURATION_SEC = 5
PROMPT_RMS = 0.001
PROMPT_FEAT_SCALE = 0.1
OUTPUT_SAMPLE_RATE = 48000
SMOOTH_SAMPLE_RATE = 24000
MAX_TEXT_CHARS = 200


def speak(
    text: str,
    model_id_or_dir: Optional[str],
    reference_wav: Path,
    out_wav: Path,
    device: str,
    num_steps: int,
    voice_key: Optional[str] = None,
    logger: Optional[Any] = None,
    session_id: Optional[str] = None,
    turn_id: Optional[str] = None,
    timeout_sec: Optional[int] = None,
    tts_cfg: Optional[dict] = None,
    verbose: bool = False,
) -> None:
    if not text:
        raise ValueError("Empty text")
    if not reference_wav:
        raise ValueError("Missing reference voice wav path")

    tts_text, _ = preprocess_text_for_tts(text, tts_cfg)
    cfg = tts_cfg or {}
    num_step_cfg = cfg.get("num_step", num_steps)
    num_steps = _clamp_int(num_step_cfg, 3, 8)
    guidance_scale = _select_guidance_scale(cfg, voice_key)
    speed = _clamp_float(cfg.get("speed", 1.0), 0.9, 1.1)
    t_shift = _clamp_float(cfg.get("t_shift", 0.5), 0.3, 0.7)
    return_smooth = bool(cfg.get("return_smooth", False))
    ref_duration = _clamp_float(cfg.get("ref_duration", PROMPT_DURATION_SEC), 2.0, 12.0)
    target_rms = _clamp_float(cfg.get("target_rms", 0.1), 0.05, 0.15)
    feat_scale = _clamp_float(cfg.get("feat_scale", PROMPT_FEAT_SCALE), 0.05, 0.2)
    voice_key, voice_path = _resolve_voice(cfg, voice_key, reference_wav)
    if verbose:
        print(
            "[LuxTTS] params:",
            {
                "num_steps": num_steps,
                "t_shift": t_shift,
                "speed": speed,
                "return_smooth": return_smooth,
                "ref_duration": ref_duration,
                "guidance_scale": guidance_scale,
                "target_rms": target_rms,
                "feat_scale": feat_scale,
                "voice_key": voice_key,
                "voice_path": str(voice_path) if voice_path else None,
            },
        )

    t0 = time.perf_counter()
    models = _get_models(model_id_or_dir, device, verbose=verbose)
    t_load = time.perf_counter()

    prompt, voice_ms = _get_voice_prompt(
        voice_path,
        models,
        voice_key=voice_key,
        ref_duration=ref_duration,
        target_rms=target_rms,
        feat_scale=feat_scale,
        verbose=verbose,
    )
    t_voice = time.perf_counter()

    audio, gen_ms, sample_rate = _generate_audio(
        tts_text,
        prompt,
        models,
        num_steps,
        timeout_sec,
        target_rms=target_rms,
        guidance_scale=guidance_scale,
        t_shift=t_shift,
        speed=speed,
        return_smooth=return_smooth,
        verbose=verbose,
    )
    t_gen = time.perf_counter()

    audio = postprocess_audio(audio, tts_cfg, sample_rate, verbose=verbose)
    _write_wav(audio, out_wav, sample_rate)
    t_write = time.perf_counter()

    audio_sec = _audio_seconds(audio, sample_rate)
    if verbose:
        _log_timing(t0, t_load, t_voice, t_gen, t_write, voice_ms, gen_ms, audio_sec)


def _get_models(model_id_or_dir: Optional[str], device: str, verbose: bool = False) -> Dict[str, Any]:
    import torch

    requested = (device or "cuda").lower()
    actual = _select_device(requested)
    model_id = (model_id_or_dir or "YatharthS/LuxTTS").strip()
    key = (actual, model_id)
    if key in _LUX_CACHE:
        return _LUX_CACHE[key]

    model_path = None
    if model_id and model_id != "YatharthS/LuxTTS":
        model_path = model_id

    with _suppress_output(not verbose):
        if actual == "cpu":
            model, feature_extractor, vocos, tokenizer, transcriber = load_models_cpu(model_path)
        else:
            model, feature_extractor, vocos, tokenizer, transcriber = load_models_gpu(
                model_path, device=actual
            )

    if hasattr(vocos, "freq_range"):
        vocos.freq_range = 12000
    if hasattr(vocos, "return_48k"):
        vocos.return_48k = True

    payload = {
        "device": actual,
        "model": model,
        "feature_extractor": feature_extractor,
        "vocos": vocos,
        "tokenizer": tokenizer,
        "transcriber": transcriber,
    }
    _LUX_CACHE[key] = payload
    return payload


def _select_device(requested: str) -> str:
    import torch

    if requested == "cpu":
        return "cpu"
    if requested == "mps":
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if requested == "cuda":
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _get_voice_prompt(
    reference_wav: Path,
    models: Dict[str, Any],
    voice_key: Optional[str],
    ref_duration: float,
    target_rms: float,
    feat_scale: float,
    verbose: bool = False,
):
    ref_path = Path(reference_wav).resolve()
    if not ref_path.exists() or not ref_path.is_file():
        raise FileNotFoundError(f"Reference wav not found: {ref_path}")

    device = models["device"]
    key = (voice_key or "default", str(ref_path), str(ref_duration), str(target_rms), str(feat_scale), device)
    if key in _VOICE_CACHE:
        return _VOICE_CACHE[key], 0

    t0 = time.perf_counter()
    transcriber = _TranscriberWrapper(models["transcriber"])
    with _suppress_output(not verbose):
        prompt_tokens, prompt_features_lens, prompt_features, prompt_rms = process_audio(
            str(ref_path),
            transcriber,
            models["tokenizer"],
            models["feature_extractor"],
            device,
            target_rms=target_rms,
            duration=ref_duration,
            feat_scale=feat_scale,
        )
    if verbose:
        _maybe_warn_reference_hygiene(ref_path, transcriber.last_text)
    encode_dict = {
        "prompt_tokens": prompt_tokens,
        "prompt_features_lens": prompt_features_lens,
        "prompt_features": prompt_features,
        "prompt_rms": prompt_rms,
    }
    voice_ms = int((time.perf_counter() - t0) * 1000)
    _VOICE_CACHE[key] = encode_dict
    return encode_dict, voice_ms


def _generate_audio(
    text: str,
    encode_dict: Dict[str, Any],
    models: Dict[str, Any],
    num_steps: int,
    timeout_sec: Optional[int],
    target_rms: float,
    guidance_scale: float,
    t_shift: float,
    speed: float,
    return_smooth: bool,
    verbose: bool = False,
):
    t0 = time.perf_counter()
    prompt_tokens = encode_dict["prompt_tokens"]
    prompt_features_lens = encode_dict["prompt_features_lens"]
    prompt_features = encode_dict["prompt_features"]
    prompt_rms = encode_dict["prompt_rms"]

    device = models["device"]
    model = models["model"]
    vocos = models["vocos"]
    tokenizer = models["tokenizer"]

    if hasattr(vocos, "return_48k"):
        vocos.return_48k = not return_smooth

    with _suppress_output(not verbose):
        if device == "cpu":
            audio = generate_cpu(
                prompt_tokens,
                prompt_features_lens,
                prompt_features,
                prompt_rms,
                text,
                model,
                vocos,
                tokenizer,
                num_step=int(num_steps),
                guidance_scale=float(guidance_scale),
                t_shift=float(t_shift),
                speed=float(speed),
                target_rms=float(target_rms),
            )
        else:
            audio = generate(
                prompt_tokens,
                prompt_features_lens,
                prompt_features,
                prompt_rms,
                text,
                model,
                vocos,
                tokenizer,
                num_step=int(num_steps),
                guidance_scale=float(guidance_scale),
                t_shift=float(t_shift),
                speed=float(speed),
                target_rms=float(target_rms),
            )

    gen_ms = int((time.perf_counter() - t0) * 1000)
    if timeout_sec is not None and gen_ms > int(timeout_sec) * 1000:
        raise TimeoutError("LuxTTS generation timed out")

    audio_np = _to_numpy(audio)
    sample_rate = OUTPUT_SAMPLE_RATE if not return_smooth else SMOOTH_SAMPLE_RATE
    return audio_np, gen_ms, sample_rate


def _to_numpy(audio):
    import numpy as np

    if hasattr(audio, "detach"):
        audio = audio.detach().cpu().float().numpy()
    elif isinstance(audio, list):
        audio = np.array(audio, dtype=np.float32)
    elif not isinstance(audio, np.ndarray):
        audio = np.array(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.squeeze()
    return audio.astype("float32")


def _write_wav(audio, out_wav: Path, sample_rate: int) -> None:
    import soundfile as sf

    out_wav.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_wav), audio, sample_rate)


def _abs_peak(audio) -> float:
    import numpy as np

    if audio is None or not getattr(audio, "size", 0):
        return 0.0
    return float(np.max(np.abs(audio)))


def _audio_seconds(audio, sample_rate: int) -> float:
    try:
        return float(len(audio)) / float(sample_rate)
    except Exception:
        return 0.0


def _log_timing(
    t0: float,
    t_load: float,
    t_voice: float,
    t_gen: float,
    t_write: float,
    voice_ms: int,
    gen_ms: int,
    audio_sec: float,
) -> None:
    def ms(a, b):
        return int((b - a) * 1000)

    print(
        "[LuxTTS] timing_ms:",
        {
            "load": ms(t0, t_load),
            "voice": voice_ms,
            "generate": gen_ms,
            "write": ms(t_gen, t_write),
            "audio_sec": round(audio_sec, 2),
        },
    )


class _TranscriberWrapper:
    def __init__(self, transcriber):
        self._transcriber = transcriber
        self.last_text = None

    def __call__(self, *args, **kwargs):
        res = self._transcriber(*args, **kwargs)
        try:
            self.last_text = (res or {}).get("text")
        except Exception:
            self.last_text = None
        return res


def _clamp_float(val, lo: float, hi: float) -> float:
    try:
        v = float(val)
    except Exception:
        v = float(lo)
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v


def _clamp_int(val, lo: int, hi: int) -> int:
    try:
        v = int(val)
    except Exception:
        v = int(lo)
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v


def preprocess_text_for_tts(text: str, cfg: Optional[dict]) -> tuple[str, dict]:
    cfg = cfg or {}
    max_tts_chars = _normalize_limit(cfg.get("max_tts_chars", MAX_TEXT_CHARS), MAX_TEXT_CHARS)
    max_sentences = _normalize_limit(cfg.get("max_sentences", 2), 2)
    max_sentence_chars = _normalize_limit(cfg.get("max_sentence_chars", 130), 130)
    add_energy = bool(cfg.get("add_energy_exclamation", False))

    original = text or ""
    t = original.strip()

    t = t.replace("...", ".")
    t = _collapse_whitespace(t)
    t = _strip_outer_quotes(t)
    t = _remove_unmatched_quotes(t)
    excl_before = t.count("!")
    q_before = t.count("?")
    t = _reduce_repeated_punct(t)
    t = _limit_exclamation_per_sentence(t)
    excl_after = t.count("!")
    q_after = t.count("?")
    t = _strip_edge_noise(t)

    sentences = _split_sentences(t)
    sentences_before = len(sentences)
    if max_sentence_chars is not None:
        sentences = _split_long_sentences(sentences, max_sentence_chars)

    selected = []
    total_len = 0
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        if max_sentences is not None and len(selected) >= max_sentences:
            break
        if max_tts_chars is not None and (total_len + len(s) > max_tts_chars):
            remaining = max_tts_chars - total_len
            if remaining <= 0:
                break
            s = s[:remaining].rstrip()
            if s:
                selected.append(s)
            break
        selected.append(s)
        total_len += len(s)

    tts_text = " ".join(selected).strip()
    if not tts_text:
        base = (original.strip() or "")
        tts_text = base if max_tts_chars is None else base[:max_tts_chars]

    if add_energy and _should_add_exclamation(tts_text):
        tts_text = tts_text + "!"

    meta = {
        "original_len": len(original),
        "final_len": len(tts_text),
        "sentences_used": len(selected),
        "sentences_before": sentences_before,
        "removed_exclamations": max(0, excl_before - excl_after),
        "removed_questions": max(0, q_before - q_after),
    }
    return tts_text, meta


def _normalize_limit(value, default: int) -> Optional[int]:
    try:
        v = int(value)
    except Exception:
        v = int(default)
    if v <= 0:
        return None
    return v


def postprocess_audio(audio, cfg: Optional[dict], sample_rate: int, verbose: bool = False):
    cfg = cfg or {}
    normalize_peak = float(cfg.get("normalize_peak", 0.85))
    silence_peak_threshold = float(cfg.get("silence_peak_threshold", 0.01))
    fade_ms = float(cfg.get("fade_ms", 8.0))

    audio_np = _to_numpy(audio)
    if audio_np.size == 0:
        raise RuntimeError("LuxTTS produced empty audio")
    before_sec = _audio_seconds(audio_np, sample_rate)
    peak = _abs_peak(audio_np)
    if peak < silence_peak_threshold:
        raise RuntimeError("LuxTTS produced near-silent audio")
    if peak > normalize_peak:
        audio_np = audio_np * (normalize_peak / peak)

    trim_cfg = cfg.get("silence_trim") or {}
    if bool(trim_cfg.get("enabled", False)):
        audio_np, stats = compress_long_silences(
            audio_np,
            sample_rate,
            silence_threshold=float(trim_cfg.get("threshold", 0.015)),
            min_silence_sec=float(trim_cfg.get("min_silence_sec", 0.60)),
            keep_silence_sec=float(trim_cfg.get("keep_silence_sec", 0.12)),
            min_silence_start_ms=float(trim_cfg.get("min_silence_start_ms", 50)),
            min_non_silence_ms=float(trim_cfg.get("min_non_silence_ms", 30)),
        )
        after_sec = _audio_seconds(audio_np, sample_rate)
        if verbose:
            print(
                "[LuxTTS] silence_trim:",
                {
                    "segments_trimmed": stats.get("segments_trimmed", 0),
                    "removed_sec": round(stats.get("removed_sec", 0.0), 3),
                    "before_sec": round(before_sec, 2),
                    "after_sec": round(after_sec, 2),
                },
            )
        if after_sec < 0.3:
            raise RuntimeError("LuxTTS audio too short after silence trim")
        if _abs_peak(audio_np) < silence_peak_threshold:
            raise RuntimeError("LuxTTS produced near-silent audio after trim")

    audio_np = _apply_fade(audio_np, sample_rate, fade_ms)
    if _abs_peak(audio_np) < silence_peak_threshold:
        raise RuntimeError("LuxTTS produced near-silent audio after fade")
    return audio_np


def _apply_fade(audio, sample_rate: int, fade_ms: float):
    import numpy as np

    if audio is None or audio.size == 0:
        return audio
    fade_len = int(sample_rate * (fade_ms / 1000.0))
    if fade_len <= 1 or fade_len * 2 > audio.size:
        return audio
    fade_in = np.linspace(0.0, 1.0, fade_len, dtype=np.float32)
    fade_out = np.linspace(1.0, 0.0, fade_len, dtype=np.float32)
    audio[:fade_len] *= fade_in
    audio[-fade_len:] *= fade_out
    return audio


def compress_long_silences(
    audio,
    sr: int,
    silence_threshold: float = 0.015,
    min_silence_sec: float = 0.60,
    keep_silence_sec: float = 0.12,
    min_silence_start_ms: float = 50.0,
    min_non_silence_ms: float = 30.0,
):
    import numpy as np

    audio = _to_numpy(audio)
    n = audio.size
    if n == 0:
        return audio, {"segments_trimmed": 0, "removed_sec": 0.0}

    head_keep = int(sr * 0.08)
    tail_keep = int(sr * 0.12)
    head_keep = min(head_keep, n)
    tail_keep = min(tail_keep, n - head_keep) if n > head_keep else 0

    env = _moving_rms(audio, max(1, int(sr * 0.02)))
    silent = env < silence_threshold

    min_silence_samples = int(sr * (min_silence_start_ms / 1000.0))
    min_non_silence_samples = int(sr * (min_non_silence_ms / 1000.0))
    if min_silence_samples < 1:
        min_silence_samples = 1

    silent = _enforce_runs(silent, min_silence_samples, True)
    if min_non_silence_samples > 1:
        silent = _enforce_runs(silent, min_non_silence_samples, False)

    min_silence_samples = int(sr * min_silence_sec)
    keep_silence_samples = max(0, int(sr * keep_silence_sec))

    if n <= head_keep + tail_keep + 1:
        return audio, {"segments_trimmed": 0, "removed_sec": 0.0}

    start = head_keep
    end = n - tail_keep

    segments = []
    i = start
    while i < end:
        if silent[i]:
            j = i + 1
            while j < end and silent[j]:
                j += 1
            segments.append((i, j))
            i = j
        else:
            i += 1

    out = []
    out.append(audio[:start])
    removed = 0
    trimmed = 0

    cursor = start
    for seg_start, seg_end in segments:
        if seg_start > cursor:
            out.append(audio[cursor:seg_start])
        seg_len = seg_end - seg_start
        if seg_len >= min_silence_samples:
            if keep_silence_samples > 0:
                out.append(np.zeros(keep_silence_samples, dtype=audio.dtype))
            removed += max(0, seg_len - keep_silence_samples)
            trimmed += 1
        else:
            out.append(audio[seg_start:seg_end])
        cursor = seg_end

    if cursor < end:
        out.append(audio[cursor:end])
    if tail_keep > 0:
        out.append(audio[end:])

    merged = np.concatenate(out) if len(out) > 1 else out[0]
    return merged, {
        "segments_trimmed": trimmed,
        "removed_sec": float(removed) / float(sr),
    }


def _enforce_runs(mask, min_len: int, value: bool):
    import numpy as np

    if min_len <= 1:
        return mask
    out = mask.copy()
    n = len(mask)
    i = 0
    while i < n:
        if mask[i] == value:
            j = i + 1
            while j < n and mask[j] == value:
                j += 1
            if (j - i) < min_len:
                out[i:j] = ~value
            i = j
        else:
            i += 1
    return out


def _moving_rms(audio, win_samples: int):
    import numpy as np

    if win_samples <= 1:
        return np.abs(audio)
    kernel = np.ones(win_samples, dtype=np.float32) / float(win_samples)
    power = np.convolve(audio.astype("float32") ** 2, kernel, mode="same")
    return np.sqrt(power)


def _collapse_whitespace(text: str) -> str:
    import re

    return re.sub(r"\s+", " ", text).strip()


def _strip_outer_quotes(text: str) -> str:
    pairs = {
        '"': '"',
        "'": "'",
        "“": "”",
        "‘": "’",
    }
    t = text.strip()
    if len(t) >= 2:
        start = t[0]
        end = t[-1]
        if start in pairs and pairs[start] == end:
            return t[1:-1].strip()
    return t


def _remove_unmatched_quotes(text: str) -> str:
    for ch in ['"', "'", "`"]:
        if text.count(ch) % 2 == 1:
            text = text.replace(ch, "")
    return text


def _reduce_repeated_punct(text: str) -> str:
    import re

    text = re.sub(r"!{2,}", "!", text)
    text = re.sub(r"\?{2,}", "?", text)
    return text


def _limit_exclamation_per_sentence(text: str) -> str:
    sentences = _split_sentences(text)
    cleaned = []
    for s in sentences:
        if "!" in s:
            first = s.find("!")
            s = s[: first + 1] + s[first + 1 :].replace("!", ".")
        cleaned.append(s)
    return " ".join(cleaned).strip()


def _strip_edge_noise(text: str) -> str:
    import re

    return re.sub(r"^\W+|\W+$", "", text).strip()


def _split_sentences(text: str) -> list[str]:
    import re

    parts = re.findall(r"[^.!?]+[.!?]?", text)
    return [p.strip() for p in parts if p.strip()]


def _split_long_sentences(sentences: list[str], max_len: int) -> list[str]:
    if max_len <= 0:
        return sentences
    result = []
    for s in sentences:
        if len(s) <= max_len:
            result.append(s)
            continue
        parts = [p.strip() for p in s.replace(";", ",").split(",") if p.strip()]
        for p in parts:
            if len(p) <= max_len:
                result.append(p)
            else:
                # hard split
                while len(p) > max_len:
                    result.append(p[:max_len].rstrip())
                    p = p[max_len:].lstrip()
                if p:
                    result.append(p)
    return result


def _should_add_exclamation(text: str) -> bool:
    t = text.strip()
    if not t:
        return False
    if "!" in t:
        return False
    if "?" in t:
        return False
    return len(t) <= 60


def _maybe_warn_reference_hygiene(ref_path: Path, transcript: Optional[str]):
    import re

    name = ref_path.name.lower()
    bad_name = any(k in name for k in ["hello", "hey", "how are you", "hi"])
    bad_transcript = False
    if transcript:
        t = transcript.lower()
        bad_transcript = bool(re.search(r"\\b(hello|hey|hi|how are you|what's up)\\b", t))
    if bad_name or bad_transcript:
        print(
            "[LuxTTS] ref_hygiene:",
            "Reference clip seems greeting-like; consider a neutral sample to reduce prompt leakage."
        )


@contextmanager
def _suppress_output(enabled: bool):
    if not enabled:
        yield
        return
    buf_out = io.StringIO()
    buf_err = io.StringIO()
    with redirect_stdout(buf_out), redirect_stderr(buf_err):
        yield


def _select_guidance_scale(cfg: dict, voice_key: Optional[str]) -> float:
    base = cfg.get("guidance_scale", 3.0)
    by_mood = cfg.get("guidance_scale_by_mood") or {}
    if voice_key and voice_key in by_mood:
        base = by_mood.get(voice_key, base)
    return _clamp_float(base, 1.0, 5.0)


def _resolve_voice(cfg: dict, voice_key: Optional[str], fallback_path: Path) -> tuple[str, Path]:
    pack = cfg.get("voice_pack") or {}
    default_key = pack.get("default") or "neutral"
    voices = pack.get("voices") or {}
    key = (voice_key or default_key).strip().lower()
    path_str = voices.get(key)
    if path_str:
        p = Path(path_str)
        if not p.is_absolute():
            p = (Path.cwd() / p).resolve()
        if p.exists() and p.is_file():
            return key, p
    return key, fallback_path
