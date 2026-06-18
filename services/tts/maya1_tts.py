from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# Dependencies:
# python -m pip install torch transformers snac soundfile numpy


_MODEL_CACHE: Dict[Tuple[str, str, str], Tuple[Any, Any, Any]] = {}

# Official Maya1 constants
CODE_START_TOKEN_ID = 128257
CODE_END_TOKEN_ID = 128258
CODE_TOKEN_OFFSET = 128266
SNAC_MIN_ID = 128266
SNAC_MAX_ID = 156937
SNAC_TOKENS_PER_FRAME = 7
DEFAULT_VOICE_DESCRIPTION = "female voice, warm, friendly, clear"
DEFAULT_MAX_FRAMES = 55


def speak(text: str, model_dir: Path, out_wav: Path, device: str, dtype: str) -> None:
    if not text:
        raise ValueError("Empty text")

    t0 = time.perf_counter()
    tokenizer, model, snac_model = _get_models(model_dir, device, dtype)
    if device == "cuda" and _is_offloaded(model):
        raise RuntimeError("maya1_offload_slow")
    t_load = time.perf_counter()

    import torch

    text = text.strip()
    if len(text) > 120:
        text = text[:120]
    text = _apply_description_prefix(text)

    input_ids = _build_prompt_input_ids(tokenizer, text).to(model.device)
    t_prompt = time.perf_counter()

    gen_args = {
        "do_sample": True,
        "temperature": 0.4,
        "top_p": 0.9,
        "repetition_penalty": 1.1,
        "no_repeat_ngram_size": 3,
        "max_new_tokens": 160,
        "min_new_tokens": 28,
        "eos_token_id": CODE_END_TOKEN_ID,
    }

    with torch.no_grad():
        generated = model.generate(input_ids=input_ids, **gen_args)
    t_gen = time.perf_counter()
    if (t_gen - t_prompt) > 25:
        raise TimeoutError("Maya1 generation timed out")

    new_ids = generated[0][input_ids.shape[1]:].tolist()
    if CODE_END_TOKEN_ID in new_ids:
        end_idx = new_ids.index(CODE_END_TOKEN_ID)
        new_ids = new_ids[:end_idx]
    had_code_end = CODE_END_TOKEN_ID in generated[0].tolist()
    if not new_ids:
        raise RuntimeError("No audio tokens generated")

    codes, stats = _extract_snac_codes(new_ids, max_frames=DEFAULT_MAX_FRAMES)
    if codes is None:
        gen_args["max_new_tokens"] = 240
        with torch.no_grad():
            generated = model.generate(input_ids=input_ids, **gen_args)
        t_gen = time.perf_counter()
        if (t_gen - t_prompt) > 25:
            raise TimeoutError("Maya1 generation timed out")
        new_ids = generated[0][input_ids.shape[1]:].tolist()
        if CODE_END_TOKEN_ID in new_ids:
            end_idx = new_ids.index(CODE_END_TOKEN_ID)
            new_ids = new_ids[:end_idx]
        had_code_end = CODE_END_TOKEN_ID in generated[0].tolist()
        codes, stats = _extract_snac_codes(new_ids, max_frames=DEFAULT_MAX_FRAMES)
        if codes is None:
            raise RuntimeError(
                "Not enough SNAC tokens "
                f"(num_generated_ids={stats['num_generated_ids']}, "
                f"num_snac_ids={stats['num_snac_ids']}, frames={stats['frames']}, "
                f"min_snac_id={stats['min_snac_id']}, max_snac_id={stats['max_snac_id']}, "
                f"first_30={stats['first_30']})"
            )

    try:
        audio = _decode_with_snac(codes, snac_model)
    except Exception as e:
        raise RuntimeError(
            "SNAC decode failed "
            f"(num_generated_ids={stats['num_generated_ids']}, "
            f"num_snac_ids={stats['num_snac_ids']}, frames={stats['frames']}, "
            f"min_snac_id={stats['min_snac_id']}, max_snac_id={stats['max_snac_id']}, "
            f"first_30={stats['first_30']})"
        ) from e
    t_decode = time.perf_counter()
    peak, rms = _audio_stats(audio)
    if peak < 1e-3 or rms < 1e-4:
        raise RuntimeError("Maya1 produced silent audio")
    duration = _write_wav(audio, out_wav, snac_model)
    t_write = time.perf_counter()

    _log_timing(
        t0, t_load, t_prompt, t_gen, t_decode, t_write,
        stats, duration, len(new_ids), had_code_end, peak, rms
    )


def _get_models(model_dir: Path, device: str, dtype: str):
    key = (str(model_dir.resolve()), device, dtype)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    if not model_dir.exists():
        raise FileNotFoundError(f"Maya1 model_dir not found: {model_dir}")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

    torch_dtype = _parse_dtype(dtype)
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = None
    if device == "cuda":
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_dir,
                dtype=torch_dtype,
                device_map=None,
                trust_remote_code=True,
            )
            model.to("cuda")
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                model = AutoModelForCausalLM.from_pretrained(
                    model_dir,
                    dtype=torch_dtype,
                    device_map="auto",
                    trust_remote_code=True,
                )
            else:
                raise
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            dtype=torch_dtype,
            device_map=None,
            trust_remote_code=True,
        )
        model.to("cpu")

    model.eval()

    try:
        generation_config = GenerationConfig.from_pretrained(model_dir)
        model.generation_config = generation_config
    except Exception:
        pass

    snac_model = _load_snac_model(model_dir, device, torch_dtype)
    _MODEL_CACHE[key] = (tokenizer, model, snac_model)
    return tokenizer, model, snac_model


def _is_offloaded(model) -> bool:
    try:
        device_map = getattr(model, "hf_device_map", None)
        if isinstance(device_map, dict):
            for dev in device_map.values():
                dev_str = str(dev)
                if dev_str in {"cpu", "disk", "meta"}:
                    return True
        for p in model.parameters():
            if p.device.type in {"cpu", "meta"}:
                return True
    except Exception:
        return True
    return False


def _parse_dtype(dtype: str):
    import torch

    d = (dtype or "float16").lower()
    if d in {"float16", "fp16"}:
        return torch.float16
    if d in {"bfloat16", "bf16"}:
        return torch.bfloat16
    return torch.float32


def _load_snac_model(model_dir: Path, device: str, torch_dtype):
    try:
        from snac import SNAC
    except Exception as e:
        raise RuntimeError("snac package not available") from e

    snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz")

    snac_model = snac_model.to(device=device, dtype=torch_dtype)
    snac_model.eval()
    return snac_model


def _extract_snac_codes(
    token_ids: list[int],
    max_frames: int = 200,
) -> tuple[Optional[tuple[list[int], list[int], list[int]]], dict]:
    snac_tokens: list[int] = []
    for tid in token_ids:
        if tid == CODE_END_TOKEN_ID:
            break
        if SNAC_MIN_ID <= tid <= SNAC_MAX_ID:
            snac_tokens.append(tid)

    num_generated = len(token_ids)
    num_snac = len(snac_tokens)
    if num_snac < 7:
        return None, {
            "num_generated_ids": num_generated,
            "num_snac_ids": num_snac,
            "frames": 0,
            "min_snac_id": None,
            "max_snac_id": None,
            "first_30": snac_tokens[:30],
        }

    frames = min(num_snac // SNAC_TOKENS_PER_FRAME, max_frames)
    if frames == 0:
        return None, {
            "num_generated_ids": num_generated,
            "num_snac_ids": num_snac,
            "frames": 0,
            "min_snac_id": None,
            "max_snac_id": None,
            "first_30": snac_tokens[:30],
        }

    snac_tokens = snac_tokens[:frames * SNAC_TOKENS_PER_FRAME]

    level0: list[int] = []
    level1: list[int] = []
    level2: list[int] = []

    for i in range(frames):
        slots = snac_tokens[i * SNAC_TOKENS_PER_FRAME:(i + 1) * SNAC_TOKENS_PER_FRAME]
        slots = [t - CODE_TOKEN_OFFSET for t in slots]
        level0.append(slots[0] % 4096)
        level1.extend([
            slots[1] % 4096,
            slots[4] % 4096,
        ])
        level2.extend([
            slots[2] % 4096,
            slots[3] % 4096,
            slots[5] % 4096,
            slots[6] % 4096,
        ])

    return (level0, level1, level2), {
        "num_generated_ids": num_generated,
        "num_snac_ids": num_snac,
        "frames": frames,
        "min_snac_id": min(snac_tokens) if snac_tokens else None,
        "max_snac_id": max(snac_tokens) if snac_tokens else None,
        "first_30": snac_tokens[:30],
    }


def _decode_with_snac(codes: Any, snac_model):
    import torch

    if not (isinstance(codes, tuple) and len(codes) == 3):
        raise RuntimeError("Invalid SNAC codes")

    level0, level1, level2 = codes
    device = next(snac_model.parameters()).device
    c0 = torch.tensor(level0, dtype=torch.long, device=device).unsqueeze(0)
    c1 = torch.tensor(level1, dtype=torch.long, device=device).unsqueeze(0)
    c2 = torch.tensor(level2, dtype=torch.long, device=device).unsqueeze(0)

    audio = snac_model.decode([c0, c1, c2])

    if isinstance(audio, (list, tuple)):
        audio = audio[0]
    return audio


def _apply_description_prefix(text: str) -> str:
    if text.startswith("<description="):
        return text
    desc = os.getenv("MAYA1_VOICE_DESCRIPTION", DEFAULT_VOICE_DESCRIPTION)
    return f'<description="{desc}"> {text}'


def _build_prompt_input_ids(tokenizer, text: str):
    import torch

    tokens = _resolve_prompt_tokens(tokenizer)
    text_ids = tokenizer.encode(text, add_special_tokens=False)
    input_ids = [tokens["soh"]] + text_ids + [tokens["eoh"], tokens["soa"], tokens["code_start"]]
    return torch.tensor([input_ids], dtype=torch.long)


def _resolve_prompt_tokens(tokenizer) -> dict:
    candidates = {
        "soh": ["<|SOH|>", "<SOH>", "<custom_token_0>"],
        "eoh": ["<|EOH|>", "<EOH>", "<custom_token_1>"],
        "soa": ["<|SOA|>", "<SOA>", "<custom_token_2>"],
    }
    token_ids = {}
    for key, opts in candidates.items():
        tid = _first_valid_token_id(tokenizer, opts)
        if tid is None:
            raise RuntimeError(f"Missing prompt token for {key}")
        token_ids[key] = tid
    token_ids["code_start"] = CODE_START_TOKEN_ID
    return token_ids


def _first_valid_token_id(tokenizer, tokens: list[str]):
    unk = getattr(tokenizer, "unk_token_id", None)
    for tok in tokens:
        tid = tokenizer.convert_tokens_to_ids(tok)
        if tid is None:
            continue
        if unk is not None and tid == unk:
            continue
        if tid >= 0:
            return tid
    return None


def _write_wav(audio, out_wav: Path, snac_model) -> None:
    import numpy as np
    import soundfile as sf

    if hasattr(audio, "detach"):
        audio = audio.detach().cpu().float().numpy()
    elif isinstance(audio, list):
        audio = np.array(audio, dtype=np.float32)

    if audio.ndim > 1:
        audio = audio.squeeze()

    sample_rate = getattr(snac_model, "sample_rate", 24000)
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_wav), audio, sample_rate)
    try:
        return float(len(audio)) / float(sample_rate)
    except Exception:
        return 0.0


def _audio_stats(audio) -> tuple[float, float]:
    import numpy as np
    if hasattr(audio, "detach"):
        audio = audio.detach().cpu().float().numpy()
    elif isinstance(audio, list):
        audio = np.array(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.squeeze()
    audio = audio.astype("float32")
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    rms = float(np.sqrt(np.mean(audio ** 2))) if audio.size else 0.0
    return peak, rms


def _log_timing(t0, t_load, t_prompt, t_gen, t_decode, t_write, stats, duration_sec: float,
                generated_len: int, had_code_end: bool, peak: float, rms: float):
    def ms(a, b):
        return int((b - a) * 1000)
    print(
        "[Maya1] timing_ms:",
        {
            "load": ms(t0, t_load),
            "prompt": ms(t_load, t_prompt),
            "generate": ms(t_prompt, t_gen),
            "decode": ms(t_gen, t_decode),
            "write": ms(t_decode, t_write),
            "frames": stats.get("frames"),
            "audio_sec": round(duration_sec, 2),
            "generated_len": generated_len,
            "had_code_end": bool(had_code_end),
            "peak": round(peak, 6),
            "rms": round(rms, 6),
        }
    )
