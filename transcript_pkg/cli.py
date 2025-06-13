#!/usr/bin/env python3
"""
Transcript CLI - Unified interface for audio transcription tools.

This CLI provides access to both live audio transcription and file-based
transcription with support for English, Portuguese, and multilingual content.
"""

import argparse
import sys
from typing import List

from rich.console import Console

console = Console()

# Constants
LANGUAGES = ["en", "pt", "auto"]
MODELS = ["tiny", "base", "small", "medium", "large"]
OUTPUT_FORMATS = ["txt", "srt", "vtt", "all"]


def create_main_parser() -> argparse.ArgumentParser:
    """Create the main argument parser."""
    parser = argparse.ArgumentParser(
        prog="transcript",
        description="Audio transcription tools for live and file-based transcription",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="",
        add_help=False,
    )

    parser.add_argument("--version", action="version", version="%(prog)s 0.1.0")
    parser.add_argument(
        "-h", "--help", action="store_true", help="show this help message and exit"
    )

    return parser


def add_live_subparser(subparsers) -> argparse.ArgumentParser:
    """Add live transcription subcommand."""
    live_parser = subparsers.add_parser(
        "live",
        help="Transcribe live audio from system audio device",
        description="Capture and transcribe audio from your TAE2146 audio device in real-time",
        add_help=False,
    )

    live_parser.add_argument(
        "-h", "--help", action="store_true", help="show this help message and exit"
    )

    live_parser.add_argument(
        "--language",
        "-l",
        choices=LANGUAGES,
        default="en",
        help="Language for transcription (default: en)",
    )

    live_parser.add_argument(
        "--model",
        "-m",
        choices=MODELS,
        default="base",
        help="Whisper model size (default: base)",
    )

    live_parser.add_argument(
        "--multilingual",
        action="store_true",
        help="Enable language detection for each segment",
    )

    live_parser.add_argument(
        "--output",
        "-o",
        help="Output file for live transcription",
    )

    live_parser.add_argument(
        "--no-timestamps",
        action="store_true",
        help="Do not include timestamps in transcription output",
    )

    live_parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU usage even if GPU is available",
    )

    return live_parser


def add_file_subparser(subparsers) -> argparse.ArgumentParser:
    """Add file transcription subcommand."""
    file_parser = subparsers.add_parser(
        "file",
        help="Transcribe audio file(s) from input path",
        description="Batch transcribe audio files or a single file with support for multiple formats",
        add_help=False,
    )

    file_parser.add_argument(
        "-h", "--help", action="store_true", help="show this help message and exit"
    )

    file_parser.add_argument(
        "--language",
        "-l",
        choices=LANGUAGES,
        default="auto",
        help="Language for transcription (default: auto)",
    )

    file_parser.add_argument(
        "--model",
        "-m",
        choices=MODELS,
        default="base",
        help="Whisper model size (default: base)",
    )

    file_parser.add_argument(
        "--input",
        "-i",
        default="./input",
        help="Input file or folder containing audio files (default: ./input)",
    )

    file_parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output folder for transcription files (default: same as input)",
    )

    file_parser.add_argument(
        "--format",
        "-f",
        choices=OUTPUT_FORMATS,
        default="txt",
        help="Output format (default: txt)",
    )

    file_parser.add_argument(
        "--multilingual",
        action="store_true",
        help="Enable language detection for each segment",
    )

    file_parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with detailed statistics (for developers)",
    )

    file_parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU usage even if GPU is available",
    )

    return file_parser


def show_examples():
    """Display examples in rich format."""
    console.print("\n[bright_cyan]Examples:[/bright_cyan]")
    console.print("  [green]# Live transcription in English[/green]")
    console.print("  transcript live --language en")
    console.print("")
    console.print("  [green]# Live transcription with output file[/green]")
    console.print("  transcript live --output session.txt")
    console.print("")
    console.print("  [green]# Live multilingual transcription[/green]")
    console.print("  transcript live --multilingual")
    console.print("")
    console.print("  [green]# Transcribe files with auto-detection[/green]")
    console.print("  transcript file")
    console.print("")
    console.print("  [green]# Transcribe a single audio file[/green]")
    console.print("  transcript file --input audio.mp3")
    console.print("")
    console.print("  [green]# Transcribe files in Portuguese with SRT output[/green]")
    console.print("  transcript file --language pt --format srt")
    console.print("")
    console.print("  [green]# Use larger model for better accuracy[/green]")
    console.print("  transcript live --model large")
    console.print("  transcript file --model large --format all\n")


def handle_help(
    argv: List[str],
    parser: argparse.ArgumentParser,
    live_parser: argparse.ArgumentParser,
    file_parser: argparse.ArgumentParser,
) -> bool:
    """Handle help display. Returns True if help was shown."""
    # Handle main help
    if len(argv) == 1 or (len(argv) == 2 and argv[1] in ["--help", "-h"]):
        parser.print_help()
        show_examples()
        return True

    # Handle subcommand help
    if len(argv) >= 3 and argv[2] in ["--help", "-h"]:
        if argv[1] == "live":
            console.clear()
            live_parser.print_help()
            return True
        elif argv[1] == "file":
            console.clear()
            file_parser.print_help()
            return True

    return False


def run_mode(args):
    """Execute the appropriate mode based on arguments."""
    if args.mode == "live":
        from transcript_pkg.live_transcribe import run_live_transcription

        run_live_transcription(args)
    elif args.mode == "file":
        from transcript_pkg.file_transcribe import run_file_transcription

        run_file_transcription(args)
    else:
        return False

    return True


def main():
    """Main CLI entry point."""
    # Create parsers
    parser = create_main_parser()

    # Create subparsers
    subparsers = parser.add_subparsers(dest="mode", help="Transcription mode")
    live_parser = add_live_subparser(subparsers)
    file_parser = add_file_subparser(subparsers)

    # Handle help display
    if handle_help(sys.argv, parser, live_parser, file_parser):
        sys.exit(0)

    # Parse arguments
    try:
        args = parser.parse_args()
    except SystemExit:
        # If parsing fails, show examples
        show_examples()
        raise

    # Clear console when running commands
    if args.mode:
        console.clear()

    # Execute mode
    if not run_mode(args):
        parser.print_help()
        show_examples()


if __name__ == "__main__":
    main()
