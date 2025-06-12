#!/usr/bin/env python3
"""
Transcript CLI - Unified interface for audio transcription tools.

This CLI provides access to both live audio transcription and file-based
transcription with support for English, Portuguese, and multilingual content.
"""

import argparse
import sys
from pathlib import Path


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="transcript",
        description="Audio transcription tools for live and file-based transcription",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Live transcription in English
  transcript live --language en
  
  # Live multilingual transcription
  transcript live --multilingual
  
  # Transcribe files with auto-detection
  transcript file
  
  # Transcribe files in Portuguese with SRT output
  transcript file --language pt --format srt
  
  # Use larger model for better accuracy
  transcript live --model large
  transcript file --model large --format all
        """,
    )

    parser.add_argument("--version", action="version", version="%(prog)s 0.1.0")

    # Create subparsers for different modes
    subparsers = parser.add_subparsers(
        dest="mode", required=True, help="Transcription mode"
    )

    # Live transcription subcommand
    live_parser = subparsers.add_parser(
        "live",
        help="Transcribe live audio from system audio device",
        description="Capture and transcribe audio from your TAE2146 audio device in real-time",
    )

    live_parser.add_argument(
        "--language",
        "-l",
        choices=["en", "pt", "auto"],
        default="en",
        help="Language for transcription (default: en)",
    )

    live_parser.add_argument(
        "--model",
        "-m",
        choices=["tiny", "base", "small", "medium", "large"],
        default="base",
        help="Whisper model size (default: base)",
    )

    live_parser.add_argument(
        "--multilingual",
        action="store_true",
        help="Enable language detection for each segment",
    )

    # File transcription subcommand
    file_parser = subparsers.add_parser(
        "file",
        help="Transcribe audio files from input folder",
        description="Batch transcribe audio files with support for multiple formats",
    )

    file_parser.add_argument(
        "--language",
        "-l",
        choices=["en", "pt", "auto"],
        default="auto",
        help="Language for transcription (default: auto)",
    )

    file_parser.add_argument(
        "--model",
        "-m",
        choices=["tiny", "base", "small", "medium", "large"],
        default="base",
        help="Whisper model size (default: base)",
    )

    file_parser.add_argument(
        "--input",
        "-i",
        default="./input",
        help="Input folder containing audio files (default: ./input)",
    )

    file_parser.add_argument(
        "--output",
        "-o",
        default="./output",
        help="Output folder for transcription files (default: ./output)",
    )

    file_parser.add_argument(
        "--format",
        "-f",
        choices=["txt", "srt", "vtt", "all"],
        default="txt",
        help="Output format (default: txt)",
    )

    file_parser.add_argument(
        "--multilingual",
        action="store_true",
        help="Enable language detection for each segment",
    )

    # Parse arguments
    args = parser.parse_args()

    # Execute the appropriate mode
    if args.mode == "live":
        # Import and run live transcription
        from transcript_pkg.live_transcribe import run_live_transcription

        run_live_transcription(args)
    elif args.mode == "file":
        # Import and run file transcription
        from transcript_pkg.file_transcribe import run_file_transcription

        run_file_transcription(args)


if __name__ == "__main__":
    main()
