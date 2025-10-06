#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
recordings_report.py
--------------------
Transcribe un conjunto de audios, los une y genera un resumen.

Novedades:
- ASR con dos motores: --asr_engine faster | hf  (por defecto: faster en CPU, hf en CUDA)
- TMP seguro en outdir/.tmp_wav
- Conversión a WAV 16k mono robusta con ffmpeg (fallback)
- Resumen con fallback (preferido -> Long-T5 -> BART)
- Orden configurable: name | mtime | manual (--order_file)

Ejemplo rápido (CPU, motor faster):
python recordings_report.py \
  --input "./audios" --outdir "./out_docs" \
  --asr_engine faster --language es --order name --clean_tmp
"""

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional

import torch
import soundfile as sf
from tqdm import tqdm

# ===== Transformers (para resumen y, si se elige, ASR HF) =====
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoProcessor,
    GenerationConfig,
)
from transformers.models.whisper import WhisperForConditionalGeneration

# ===== faster-whisper (ASR rápido) =====
try:
    from faster_whisper import WhisperModel
    _HAS_FASTER = True
except Exception:
    _HAS_FASTER = False

# ---------- Ajustes de hilos (mejor CPU/M2) ----------
try:
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
except Exception:
    pass

AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac", ".webm"}
TEMP_WAV_SUFFIX = ".16k.wav"


# ===========================
# Utilidades de archivo/orden
# ===========================
def list_audio_files(folder: Path) -> List[Path]:
    files = []
    for p in folder.rglob("*"):
        if not p.is_file():
            continue
        if p.name.startswith("."):
            continue
        if p.suffix.lower() in AUDIO_EXTS and not p.name.endswith(TEMP_WAV_SUFFIX):
            files.append(p)
    return files


def apply_order(files: List[Path], mode: str, order_file: Optional[Path] = None) -> List[Path]:
    mode = (mode or "name").lower()
    if mode == "name":
        return sorted(files, key=lambda p: p.name.lower())
    if mode == "mtime":
        return sorted(files, key=lambda p: p.stat().st_mtime)
    if mode == "manual":
        if not order_file or not order_file.exists():
            raise ValueError("--order manual requiere --order_file con rutas/nombres por línea.")
        with open(order_file, "r", encoding="utf-8") as f:
            wanted = [ln.strip() for ln in f if ln.strip()]
        index_by_name = {p.name: p for p in files}
        ordered, used = [], set()
        for item in wanted:
            base = Path(item).name
            cand = index_by_name.get(base)
            if cand and cand not in used:
                ordered.append(cand); used.add(cand); continue
            try:
                cand2 = next(p for p in files if str(p) == item or p.name == base)
                if cand2 not in used:
                    ordered.append(cand2); used.add(cand2); continue
            except StopIteration:
                pass
            print(f"[WARN] '{item}' no coincide con ningún archivo en la carpeta de entrada.")
        for p in files:
            if p not in used:
                ordered.append(p)
        return ordered
    raise ValueError(f"--order inválido: {mode}. Usa name|mtime|manual.")


# ===========================
# Conversión y normalización
# ===========================
def _ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def _convert_with_ffmpeg(input_path: Path, out_path: Path, target_sr: int = 16000) -> Path:
    if not _ffmpeg_available():
        raise RuntimeError(
            "ffmpeg no está en PATH. Instálalo (macOS: 'brew install ffmpeg', Debian/Ubuntu: 'sudo apt-get install -y ffmpeg')."
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_path),
        "-ac", "1",
        "-ar", str(target_sr),
        "-vn",
        "-c:a", "pcm_s16le",
        str(out_path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return out_path


def load_or_resample_to_wav(input_path: Path, tmp_dir: Path, target_sr: int = 16000) -> Path:
    """
    Devuelve un WAV mono 16 kHz en tmp_dir.
    Intenta con soundfile; si falla (p.ej. .m4a), usa ffmpeg.
    """
    out_wav = tmp_dir / (input_path.stem + TEMP_WAV_SUFFIX)
    if out_wav.exists():
        return out_wav

    # 1) soundfile directo (cuando puede)
    try:
        data, sr = sf.read(str(input_path), always_2d=False)
        import numpy as np
        if data.ndim == 2:
            data = data.mean(axis=1)
        if sr != target_sr:
            import resampy
            data = resampy.resample(data, sr, target_sr)
        sf.write(out_wav, data, target_sr)
        return out_wav
    except Exception:
        # 2) Fallback robusto: ffmpeg (maneja .m4a y más)
        return _convert_with_ffmpeg(input_path, out_wav, target_sr=target_sr)


# ===============
# ASR – motores
# ===============
def _split_audio_to_chunks(wav_path: Path, chunk_len_s: int = 60) -> List[tuple]:
    """
    Lee un WAV 16k mono y lo parte en trozos de chunk_len_s (segundos).
    Devuelve lista de tuplas (np.ndarray float32, sample_rate).
    """
    import numpy as np
    import soundfile as sf
    audio, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
    if sr != 16000:
        raise ValueError(f"Se esperaba 16kHz, llegó {sr} Hz en {wav_path}")
    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    chunk_samples = int(chunk_len_s * sr)
    chunks = []
    for i in range(0, len(audio), chunk_samples):
        ch = audio[i:i+chunk_samples]
        if ch.size > 0:
            chunks.append((ch, sr))
    return chunks


def transcribe_with_faster_whisper(
    wav_path: Path,
    device_str: str = "cpu",
    language: str = "es",
    model_size: str = "base",
    beam_size: int = 1,
) -> str:
    """
    Transcribe con faster-whisper dividiendo el audio en ventanas cortas.
    Usa greedy (temperature=0.0, beam_size=1, best_of=1), sin VAD, para máxima estabilidad en CPU.
    """
    if not _HAS_FASTER:
        raise RuntimeError("faster-whisper no está instalado. Ejecuta: pip install faster-whisper")

    # Configura el backend CTranslate2
    device = "cuda" if device_str == "cuda" else "cpu"
    compute_type = "int8" if device == "cpu" else "float16"

    # Limitar threads/workers ayuda a evitar “cuelgues” en Mac M2
    model = WhisperModel(
        model_size if model_size in {"tiny", "base", "small", "medium", "large"} else "base",
        device=device,
        compute_type=compute_type,
        cpu_threads=4 if device == "cpu" else 0,
        num_workers=1,
    )

    # Trocea en ventanas de 60 s para que cada decode sea corto
    chunks = _split_audio_to_chunks(wav_path, chunk_len_s=60)

    texts: List[str] = []
    for audio_np, sr in chunks:
        # Greedy puro: sin beam search, sin sampling, sin VAD
        segments, _ = model.transcribe(
            audio_np,
            language=language,
            temperature=0.0,        # greedy
            beam_size=1,            # sin beam search
            best_of=1,              # sin re-muestreo
            patience=1,
            condition_on_previous_text=False,
            vad_filter=False,       # evitamos VAD para no agregar complejidad/costos
            no_speech_threshold=0.6,
            log_prob_threshold=-1.0,
            compression_ratio_threshold=2.4,
            word_timestamps=False,
        )
        piece = " ".join(s.text.strip() for s in segments if s.text)
        if piece:
            texts.append(piece)

    return " ".join(texts)


def chunk_array(arr, chunk_size):
    for i in range(0, len(arr), chunk_size):
        yield arr[i:i + chunk_size]


def transcribe_with_hf_whisper(
    wav_path: Path,
    processor: AutoProcessor,
    asr_model: WhisperForConditionalGeneration,
    gen_cfg: GenerationConfig,
    device_str: str = "cpu",
    language: str = "es",
    chunk_length_s: int = 8,   # corto para CPU
) -> str:
    """
    Transcribe un WAV 16k mono haciendo chunking y greedy decoding (sin sampling) con HF.
    """
    audio, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
    if sr != 16000:
        raise ValueError(f"Se esperaba 16kHz, llegó {sr} Hz en {wav_path}")
    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    forced_ids = processor.get_decoder_prompt_ids(language=language, task="transcribe")
    device = torch.device("cuda") if device_str == "cuda" else torch.device("cpu")

    chunk_samples = int(chunk_length_s * sr)
    texts: List[str] = []

    asr_model.eval()
    with torch.inference_mode():
        for chunk in chunk_array(audio, chunk_samples):
            if len(chunk) == 0:
                continue
            inputs = processor(
                audio=chunk,
                sampling_rate=sr,
                return_attention_mask=True,
                return_tensors="pt"
            )
            input_features = inputs["input_features"].to(device)
            attention_mask = inputs.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

            generated_ids = asr_model.generate(
                input_features=input_features,
                attention_mask=attention_mask,
                forced_decoder_ids=forced_ids,
                generation_config=gen_cfg,  # 🔒 greedy
            )
            text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            texts.append(text.strip())

    return " ".join(t for t in texts if t)


# ====================
# Resumen con fallback
# ====================
def _chunk_text(text: str, max_words: int) -> List[str]:
    paras = [p.strip() for p in text.split("\n") if p.strip()]
    chunks, buf, count = [], [], 0
    for p in paras:
        words = p.split()
        if count + len(words) > max_words and buf:
            chunks.append(" ".join(buf)); buf, count = [], 0
        buf.append(p); count += len(words)
    if buf:
        chunks.append(" ".join(buf))
    return chunks


def summarize_text(text: str, preferred_model: Optional[str], device_str: str = "cpu") -> str:
    from transformers import pipeline
    candidates: List[str] = []
    if preferred_model:
        candidates.append(preferred_model)
    candidates += [
        "pszemraj/long-t5-tglobal-base-16384-book-summary",
        "facebook/bart-large-cnn",
    ]

    last_err = None
    for m in candidates:
        try:
            dtype = torch.float16 if device_str == "cuda" else torch.float32
            tok = AutoTokenizer.from_pretrained(m)
            mod = AutoModelForSeq2SeqLM.from_pretrained(m, torch_dtype=dtype)
            if device_str == "cuda":
                mod = mod.to("cuda")
            summ = pipeline(
                "summarization",
                model=mod,
                tokenizer=tok,
                device=0 if device_str == "cuda" else -1,
            )

            max_words_first = 2400 if "long-t5" in m else 1200
            parts = _chunk_text(text, max_words=max_words_first)

            partials: List[str] = []
            for ch in tqdm(parts, desc=f"Resumiendo ({m})"):
                params = {
                    "max_length": 360 if "bart" in m else 300,
                    "min_length": 120 if "bart" in m else 100,
                    "do_sample": False,
                }
                out = summ(ch, **params)
                partials.append(out[0]["summary_text"].strip())

            merged = " ".join(partials)
            if len(partials) > 1:
                params2 = {
                    "max_length": 300 if "bart" in m else 220,
                    "min_length": 100 if "bart" in m else 80,
                    "do_sample": False,
                }
                out2 = summ(merged, **params2)
                return out2[0]["summary_text"].strip()
            return merged

        except Exception as e:
            last_err = e
            print(f"[WARN] No se pudo usar '{m}': {e}. Probando siguiente...")

    raise RuntimeError(f"No se pudo inicializar ningún modelo de resumen. Último error: {last_err}")


# ================
# Programa principal
# ================
def main():
    ap = argparse.ArgumentParser(
        description="Transcribir audios, unir y resumir (TMP seguro, ffmpeg fallback, CPU optimizado, orden y fallback de resumen)."
    )
    ap.add_argument("--input", required=True, help="Carpeta de entrada con audios")
    ap.add_argument("--outdir", required=True, help="Carpeta de salida")
    ap.add_argument("--asr_engine", default="auto", choices=["auto", "faster", "hf"], help="Motor ASR a usar")
    ap.add_argument("--asr_model", default="openai/whisper-base", help="Modelo ASR (HF: repo id | faster: tamaño: tiny/base/small/medium/large)")
    ap.add_argument("--sum_model", default=None, help="Modelo de resumen preferido (opcional)")
    ap.add_argument("--language", default="es", help="Idioma para Whisper (ej. es, en)")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Dispositivo para inferencia")
    ap.add_argument("--order", default="name", choices=["name", "mtime", "manual"], help="Orden de los audios a procesar")
    ap.add_argument("--order_file", default=None, help="Ruta a archivo de texto con orden manual (uno por línea)")
    ap.add_argument("--keep_timestamps", action="store_true", help="(Sólo relevante con faster: sí devuelve segmentos)")
    ap.add_argument("--clean_tmp", action="store_true", help="Borrar outdir/.tmp_wav al finalizar")
    args = ap.parse_args()

    in_dir = Path(args.input)
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # TMP seguro
    tmp_wav_dir = out_dir / ".tmp_wav"
    tmp_wav_dir.mkdir(exist_ok=True)

    # Listado
    files = list_audio_files(in_dir)
    if not files:
        print("No se encontraron audios.")
        return

    # Orden
    order_file = Path(args.order_file) if args.order_file else None
    files = apply_order(files, args.order, order_file)

    # Dispositivo
    if args.device == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_str = args.device
    print(f"Usando dispositivo: {device_str}")

    # Motor ASR a usar
    engine = args.asr_engine
    if engine == "auto":
        engine = "hf" if device_str == "cuda" else "faster"
    print(f"Motor ASR: {engine}")

    transcripts: List[Dict[str, Any]] = []

    # Transcribir
    if engine == "faster":
        # args.asr_model para faster-whisper: usa tamaños (tiny/base/small/medium/large)
        fw_size = args.asr_model
        for f in tqdm(files, desc="Transcribiendo (faster-whisper)"):
            try:
                wav = load_or_resample_to_wav(f, tmp_dir=tmp_wav_dir, target_sr=16000)
                text = transcribe_with_faster_whisper(
                    wav_path=wav,
                    device_str=device_str,
                    language=args.language,
                    model_size=fw_size if fw_size in {"tiny", "base", "small", "medium", "large"} else "base",
                    beam_size=1,
                )
                rec = {"file": str(f), "text": text}
                transcripts.append(rec)
            except Exception as e:
                print(f"[WARN] Falló {f.name}: {e}")
                continue

    else:  # engine == "hf"
        # HF Whisper (más lento en CPU). Usa repo_id completo en --asr_model.
        asr_model_id = args.asr_model
        processor = AutoProcessor.from_pretrained(asr_model_id)
        asr_model = WhisperForConditionalGeneration.from_pretrained(asr_model_id)
        if device_str == "cuda":
            asr_model = asr_model.to("cuda")

        # GenerationConfig explicitamente greedy (nunca _sample)
        gen_cfg = GenerationConfig.from_model_config(asr_model.config)
        gen_cfg.do_sample = False
        gen_cfg.num_beams = 1
        gen_cfg.max_new_tokens = 225
        gen_cfg.top_k = None
        gen_cfg.top_p = None
        gen_cfg.temperature = None

        for f in tqdm(files, desc="Transcribiendo (HF Whisper)"):
            try:
                wav = load_or_resample_to_wav(f, tmp_dir=tmp_wav_dir, target_sr=16000)
                text = transcribe_with_hf_whisper(
                    wav_path=wav,
                    processor=processor,
                    asr_model=asr_model,
                    gen_cfg=gen_cfg,
                    device_str=device_str,
                    language=args.language,
                    chunk_length_s=8 if device_str == "cpu" else 30,
                )
                rec = {"file": str(f), "text": text}
                transcripts.append(rec)
            except Exception as e:
                print(f"[WARN] Falló {f.name}: {e}")
                continue

    if not transcripts:
        raise RuntimeError("No se obtuvo ninguna transcripción válida.")

    # Guardar transcripciones por archivo
    per_file_dir = out_dir / "per_file"
    per_file_dir.mkdir(exist_ok=True)
    for t in transcripts:
        base = Path(t["file"]).stem
        (per_file_dir / f"{base}.txt").write_text(t["text"], encoding="utf-8")

    # Unir transcripciones (con encabezado por archivo)
    combined = "\n\n".join([f"### {Path(t['file']).name}\n{t['text']}" for t in transcripts])

    # Resumen con fallback
    summary = summarize_text(
        "\n\n".join([t["text"] for t in transcripts]),
        preferred_model=args.sum_model,
        device_str=device_str
    )

    # Markdown final
    md_parts = []
    md_parts.append("# Informe de Transcripción y Resumen\n")
    md_parts.append("## Archivos Procesados\n")
    for t in transcripts:
        md_parts.append(f"- {Path(t['file']).name}")
    md_parts.append("\n## Resumen Ejecutivo\n")
    md_parts.append(summary.strip())
    md_parts.append("\n## Transcripción Consolidada\n")
    md_parts.append(combined.strip())
    report_md = "\n".join(md_parts)

    (out_dir / "informe_transcripcion_resumen.md").write_text(report_md, encoding="utf-8")
    (out_dir / "transcripcion_consolidada.txt").write_text(
        "\n\n".join([t["text"] for t in transcripts]), encoding="utf-8"
    )

    if args.clean_tmp:
        shutil.rmtree(tmp_wav_dir, ignore_errors=True)

    print("\nListo ✅")
    print(f"- {out_dir / 'informe_transcripcion_resumen.md'}")
    print(f"- {out_dir / 'transcripcion_consolidada.txt'}")
    print(f"- Transcripciones por archivo: {per_file_dir}")
    if not args.clean_tmp:
        print(f"- TMP WAVs (para depurar): {tmp_wav_dir} (borra con --clean_tmp si quieres)")


if __name__ == "__main__":
    main()