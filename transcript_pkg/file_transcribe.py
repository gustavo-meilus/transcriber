import argparse
import os
import time
from pathlib import Path

from faster_whisper import WhisperModel
from tqdm import tqdm


def format_timestamp(seconds):
    """Convert seconds to SRT/VTT timestamp format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"


def save_transcription(
    segments, output_file, format_type, detected_language=None, multilingual=False
):
    """Save transcription in the specified format."""
    if format_type == "txt":
        # Plain text format
        with open(output_file, "w", encoding="utf-8") as f:
            if detected_language and not multilingual:
                f.write(f"[Detected language: {detected_language}]\n\n")
            elif multilingual:
                f.write("[Multilingual transcription - language shown in brackets]\n\n")

            for segment in segments:
                if multilingual and hasattr(segment, "language"):
                    f.write(f"[{segment.language}] {segment.text.strip()}\n")
                else:
                    f.write(f"{segment.text.strip()}\n")

    elif format_type == "srt":
        # SRT subtitle format
        with open(output_file, "w", encoding="utf-8") as f:
            for i, segment in enumerate(segments, 1):
                f.write(f"{i}\n")
                f.write(
                    f"{format_timestamp(segment.start).replace('.', ',')} --> {format_timestamp(segment.end).replace('.', ',')}\n"
                )
                if multilingual and hasattr(segment, "language"):
                    f.write(f"[{segment.language}] {segment.text.strip()}\n\n")
                else:
                    f.write(f"{segment.text.strip()}\n\n")

    elif format_type == "vtt":
        # WebVTT format
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("WEBVTT\n\n")

            for segment in segments:
                f.write(
                    f"{format_timestamp(segment.start)} --> {format_timestamp(segment.end)}\n"
                )
                if multilingual and hasattr(segment, "language"):
                    f.write(f"[{segment.language}] {segment.text.strip()}\n\n")
                else:
                    f.write(f"{segment.text.strip()}\n\n")


def run_file_transcription(args):
    """Run file transcription with the given arguments."""
    # Language mapping
    LANGUAGE_MAP = {
        "en": "en",
        "pt": "pt",
        "auto": None,  # None means auto-detect
    }

    selected_language = LANGUAGE_MAP[args.language]
    language_display = {"en": "English", "pt": "Portuguese", "auto": "Auto-detect"}[
        args.language
    ]

    # Supported audio formats
    AUDIO_EXTENSIONS = {
        ".mp3",
        ".wav",
        ".flac",
        ".ogg",
        ".m4a",
        ".mp4",
        ".aac",
        ".wma",
        ".opus",
        ".webm",
        ".mkv",
        ".avi",
        ".mov",
        ".m4v",
    }

    # Create output directory if it doesn't exist
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create input directory if it doesn't exist
    input_path = Path(args.input)
    if not input_path.exists():
        input_path.mkdir(parents=True, exist_ok=True)
        print(f"Created input directory: {input_path}")
        print(
            "Please place your audio files in this directory and run the script again."
        )
        return

    # Find all audio files in input directory
    audio_files = []
    for file in input_path.iterdir():
        if file.is_file() and file.suffix.lower() in AUDIO_EXTENSIONS:
            audio_files.append(file)

    if not audio_files:
        print(f"No audio files found in {input_path}")
        print(f"Supported formats: {', '.join(sorted(AUDIO_EXTENSIONS))}")
        return

    print(f"Found {len(audio_files)} audio file(s) to transcribe")
    print(f"Language mode: {language_display}")
    print(f"Output format: {args.format}")
    if args.multilingual:
        print("Multilingual mode: ENABLED (will detect language changes within files)")

    # Load the Whisper model
    print(f"\nLoading Whisper model ({args.model})...")
    model = WhisperModel(args.model, device="cpu", compute_type="int8")

    # Process each audio file
    print("\nStarting transcription...\n")

    # Statistics tracking
    total_duration = 0
    total_process_time = 0
    successful_files = 0
    failed_files = 0
    overall_start_time = time.time()

    for audio_file in tqdm(audio_files, desc="Transcribing files"):
        try:
            start_time = time.time()

            # Transcribe the audio file
            segments, info = model.transcribe(
                str(audio_file),
                language=selected_language,
                beam_size=5,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500),
                multilingual=args.multilingual,
            )

            # Convert generator to list to reuse segments
            segments_list = list(segments)

            # Get detected language
            detected_lang = info.language if selected_language is None else None

            # Save transcription based on format
            if args.format == "all":
                # Save in all formats
                for fmt in ["txt", "srt", "vtt"]:
                    output_file = output_path / f"{audio_file.stem}.{fmt}"
                    save_transcription(
                        segments_list,
                        output_file,
                        fmt,
                        detected_lang,
                        args.multilingual,
                    )
            else:
                # Save in specified format
                ext = "txt" if args.format == "txt" else args.format
                output_file = output_path / f"{audio_file.stem}.{ext}"
                save_transcription(
                    segments_list,
                    output_file,
                    args.format,
                    detected_lang,
                    args.multilingual,
                )

            # Calculate processing time
            process_time = time.time() - start_time

            # Update statistics
            total_duration += info.duration
            total_process_time += process_time
            successful_files += 1

            # Show summary for this file
            tqdm.write(f"✓ {audio_file.name}")
            if detected_lang:
                tqdm.write(f"  Language: {detected_lang}")
            tqdm.write(f"  Duration: {info.duration:.1f}s")
            tqdm.write(f"  Process time: {process_time:.1f}s")
            tqdm.write(f"  Speed: {info.duration / process_time:.1f}x realtime")
            tqdm.write("")

        except Exception as e:
            tqdm.write(f"✗ Error processing {audio_file.name}: {e}")
            failed_files += 1
            continue

    # Calculate overall statistics
    overall_time = time.time() - overall_start_time

    print(f"\nTranscription complete! Files saved to: {output_path}")
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Files processed: {successful_files} successful, {failed_files} failed")
    print(f"Output format: {args.format}")
    print(f"Language mode: {language_display}")
    if args.multilingual:
        print("Multilingual: Enabled")

    print("\n" + "-" * 50)
    print("PERFORMANCE STATISTICS")
    print("-" * 50)
    print(
        f"Total audio duration: {total_duration:.1f} seconds ({total_duration / 60:.1f} minutes)"
    )
    print(
        f"Total processing time: {total_process_time:.1f} seconds ({total_process_time / 60:.1f} minutes)"
    )
    print(
        f"Overall time (including loading): {overall_time:.1f} seconds ({overall_time / 60:.1f} minutes)"
    )

    if successful_files > 0 and total_duration > 0:
        avg_speed = total_duration / total_process_time
        print(f"\nAverage processing speed: {avg_speed:.1f}x realtime")
        print(
            f"Time per minute of audio: {(total_process_time / total_duration) * 60:.1f} seconds"
        )

        # Estimate for longer content
        print(
            f"\nEstimated time for 1 hour of audio: {(total_process_time / total_duration) * 3600 / 60:.1f} minutes"
        )

    print("=" * 50)


# For backward compatibility when running directly
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Transcribe audio files from input folder to output folder"
    )
    parser.add_argument(
        "--language",
        "-l",
        choices=["en", "pt", "auto"],
        default="auto",
        help="Language for transcription: 'en' for English, 'pt' for Portuguese, 'auto' for automatic detection",
    )
    parser.add_argument(
        "--model",
        "-m",
        choices=["tiny", "base", "small", "medium", "large"],
        default="base",
        help="Whisper model size (default: base)",
    )
    parser.add_argument(
        "--input",
        "-i",
        default="./input",
        help="Input folder containing audio files (default: ./input)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="./output",
        help="Output folder for transcription files (default: ./output)",
    )
    parser.add_argument(
        "--format",
        "-f",
        choices=["txt", "srt", "vtt", "all"],
        default="txt",
        help="Output format: txt (plain text), srt (subtitles), vtt (WebVTT), or all",
    )
    parser.add_argument(
        "--multilingual",
        action="store_true",
        help="Enable language detection for each segment (useful for audio with multiple languages)",
    )
    args = parser.parse_args()
    run_file_transcription(args)
