#!/usr/bin/env python3
import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path


def parse_timecode(value: str) -> float | None:
    try:
        hours, minutes, rest = value.split(":")
        seconds = float(rest)
        return int(hours) * 3600 + int(minutes) * 60 + seconds
    except ValueError:
        return None


def get_duration_seconds(input_path: Path) -> float | None:
    if shutil.which("ffprobe") is None:
        return None
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=nw=1:nk=1",
            str(input_path),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    try:
        return float(result.stdout.strip())
    except ValueError:
        return None


def render_progress(done_seconds: float, total_seconds: float) -> None:
    if total_seconds <= 0:
        return
    percent = min(done_seconds / total_seconds, 1.0)
    bar_width = 30
    filled = int(percent * bar_width)
    bar = "=" * filled + "-" * (bar_width - filled)
    print(f"\r[{bar}] {percent * 100:5.1f}%", end="", file=sys.stderr, flush=True)


def run_ffmpeg(args: list[str], duration_seconds: float | None) -> None:
    if duration_seconds is None:
        result = subprocess.run(args)
        if result.returncode != 0:
            raise RuntimeError("ffmpeg failed")
        return

    progress_args = list(args)
    insert_idx = max(len(progress_args) - 1, 1)
    progress_args[insert_idx:insert_idx] = [
        "-progress",
        "pipe:1",
        "-nostats",
        "-loglevel",
        "error",
    ]

    process = subprocess.Popen(
        progress_args,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    current_seconds = 0.0
    last_render = 0.0
    if process.stdout is not None:
        for line in process.stdout:
            line = line.strip()
            if line.startswith("out_time_ms="):
                try:
                    current_seconds = int(line.split("=", 1)[1]) / 1_000_000.0
                except ValueError:
                    continue
            elif line.startswith("out_time="):
                parsed = parse_timecode(line.split("=", 1)[1])
                if parsed is not None:
                    current_seconds = parsed
            elif line.startswith("progress=end"):
                render_progress(duration_seconds, duration_seconds)
                break

            now = time.time()
            if now - last_render >= 0.25:
                render_progress(current_seconds, duration_seconds)
                last_render = now

    return_code = process.wait()
    print("", file=sys.stderr)
    if return_code != 0:
        raise RuntimeError("ffmpeg failed")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert a WebM file to MP4 using ffmpeg without quality loss.",
    )
    parser.add_argument("input", type=Path, help="Path to the input .webm file")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Path to the output .mp4 file (default: same name with .mp4)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output if it exists",
    )
    parser.add_argument(
        "--audio-codec",
        default="alac",
        help="Audio codec for transcode fallback (default: alac)",
    )
    parser.add_argument(
        "--crf",
        type=int,
        default=0,
        help="CRF value for x264 fallback (0 is lossless)",
    )
    args = parser.parse_args()

    if shutil.which("ffmpeg") is None:
        print("ffmpeg not found in PATH.", file=sys.stderr)
        return 1

    input_path = args.input
    if not input_path.exists():
        print(f"Input not found: {input_path}", file=sys.stderr)
        return 1

    output_path = args.output or input_path.with_suffix(".mp4")
    if output_path.exists() and not args.overwrite:
        print(f"Output exists: {output_path}", file=sys.stderr)
        return 1

    overwrite_flag = "-y" if args.overwrite else "-n"

    duration_seconds = get_duration_seconds(input_path)

    # Fast path: try stream copy (lossless and fastest) when codecs are compatible.
    try:
        run_ffmpeg(
            [
                "ffmpeg",
                overwrite_flag,
                "-i",
                str(input_path),
                "-map",
                "0",
                "-c",
                "copy",
                "-movflags",
                "+faststart",
                str(output_path),
            ],
            duration_seconds,
        )
        return 0
    except RuntimeError:
        if output_path.exists():
            output_path.unlink()

    # Fallback: lossless transcode with a fast preset.
    try:
        run_ffmpeg(
            [
                "ffmpeg",
                overwrite_flag,
                "-fflags",
                "+genpts+discardcorrupt",
                "-err_detect",
                "ignore_err",
                "-i",
                str(input_path),
                "-map",
                "0:v:0",
                "-map",
                "0:a:0?",
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",
                "-crf",
                str(args.crf),
                "-c:a",
                args.audio_codec,
                "-movflags",
                "+faststart",
                str(output_path),
            ],
            duration_seconds,
        )
    except RuntimeError:
        print("ffmpeg failed to convert the file.", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
