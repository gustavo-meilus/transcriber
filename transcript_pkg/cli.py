#!/usr/bin/env python3
"""
Transcript CLI - Unified interface for audio transcription tools.

This CLI provides access to both live audio transcription and file-based
transcription with support for English, Portuguese, and multilingual content.
"""

import argparse
import sys

from rich.console import Console

console = Console()


def show_banner():
    """Display ASCII art banner."""
    banner = """
╔═══════════════════════════════════════════════════════════════╗
║  ████████ ██████   █████  ███    ██ ███████  ██████ ██████   ║
║     ██    ██   ██ ██   ██ ████   ██ ██      ██      ██   ██  ║
║     ██    ██████  ███████ ██ ██  ██ ███████ ██      ██████   ║
║     ██    ██   ██ ██   ██ ██  ██ ██      ██ ██      ██   ██  ║
║     ██    ██   ██ ██   ██ ██   ████ ███████  ██████ ██   ██  ║
╚═══════════════════════════════════════════════════════════════╝
                  Audio Transcription Tools v0.1.0                
    """
    console.print(banner, style="bright_cyan", justify="center")


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


def main():
    """Main CLI entry point."""
    # Show banner for main help
    if len(sys.argv) == 1 or (len(sys.argv) == 2 and sys.argv[1] in ["--help", "-h"]):
        show_banner()

    parser = argparse.ArgumentParser(
        prog="transcript",
        description="Audio transcription tools for live and file-based transcription",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="",  # Remove epilog, we'll show examples separately
        add_help=False,  # We'll handle help manually
    )

    parser.add_argument("--version", action="version", version="%(prog)s 0.1.0")
    parser.add_argument(
        "-h", "--help", action="store_true", help="show this help message and exit"
    )

    # Create subparsers for different modes
    subparsers = parser.add_subparsers(dest="mode", help="Transcription mode")

    # Live transcription subcommand
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

    # File transcription subcommand
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
        choices=["txt", "srt", "vtt", "all"],
        default="txt",
        help="Output format (default: txt)",
    )

    file_parser.add_argument(
        "--multilingual",
        action="store_true",
        help="Enable language detection for each segment",
    )

    # Handle help manually
    if len(sys.argv) == 1 or (len(sys.argv) == 2 and sys.argv[1] in ["--help", "-h"]):
        parser.print_help()
        show_examples()
        sys.exit(0)

    # Check for subcommand help
    if len(sys.argv) >= 3 and sys.argv[2] in ["--help", "-h"]:
        if sys.argv[1] == "live":
            console.clear()
            show_banner()
            live_parser.print_help()
        elif sys.argv[1] == "file":
            console.clear()
            show_banner()
            file_parser.print_help()
        sys.exit(0)

    # Parse arguments
    try:
        args = parser.parse_args()
    except SystemExit:
        # If parsing fails, show help
        show_examples()
        raise

    # Show appropriate banner when running commands
    if args.mode:
        console.clear()
        show_banner()

    # Execute the appropriate mode
    if args.mode == "live":
        # Import and run live transcription
        from transcript_pkg.live_transcribe import run_live_transcription

        run_live_transcription(args)
    elif args.mode == "file":
        # Import and run file transcription
        from transcript_pkg.file_transcribe import run_file_transcription

        run_file_transcription(args)
    else:
        parser.print_help()
        show_examples()


if __name__ == "__main__":
    main()
