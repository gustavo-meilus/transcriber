import argparse
import json
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from faster_whisper import WhisperModel
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.prompt import Confirm
from rich.table import Table

console = Console()


class TranscriptionCheckpoint:
    """Manages checkpoint data for resumable transcriptions."""

    def __init__(self, checkpoint_path: Path):
        self.checkpoint_file = checkpoint_path / ".transcription_checkpoint.json"
        self.data = {
            "files_completed": [],
            "current_file": None,
            "current_file_segments": [],
            "timestamp": None,
            "settings": {},
        }

    def load(self) -> bool:
        """Load checkpoint data if it exists."""
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, "r") as f:
                    self.data = json.load(f)
                return True
            except Exception as e:
                console.print(
                    f"[yellow]Warning: Could not load checkpoint: {e}[/yellow]"
                )
        return False

    def save(self):
        """Save current checkpoint data."""
        self.data["timestamp"] = datetime.now().isoformat()
        try:
            # Ensure directory exists
            self.checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.checkpoint_file, "w") as f:
                json.dump(self.data, f, indent=2)
        except Exception as e:
            console.print(f"[yellow]Warning: Could not save checkpoint: {e}[/yellow]")

    def clear(self):
        """Remove checkpoint file."""
        if self.checkpoint_file.exists():
            try:
                self.checkpoint_file.unlink()
            except Exception:
                pass

    def mark_file_complete(self, filename: str):
        """Mark a file as completed."""
        if filename not in self.data["files_completed"]:
            self.data["files_completed"].append(filename)
        if self.data["current_file"] == filename:
            self.data["current_file"] = None
            self.data["current_file_segments"] = []
        self.save()

    def set_current_file(self, filename: str):
        """Set the currently processing file."""
        self.data["current_file"] = filename
        self.data["current_file_segments"] = []
        self.save()

    def add_segment(self, segment_data: dict):
        """Add a processed segment to the current file."""
        self.data["current_file_segments"].append(segment_data)
        # Save every 10 segments to reduce I/O
        if len(self.data["current_file_segments"]) % 10 == 0:
            self.save()


class StreamingTranscriptionWriter:
    """Handles streaming output of transcription data."""

    def __init__(
        self,
        output_file: Path,
        format_type: str,
        detected_language: Optional[str] = None,
        multilingual: bool = False,
    ):
        self.output_file = output_file
        self.format_type = format_type
        self.detected_language = detected_language
        self.multilingual = multilingual
        self.segment_count = 0
        self.file_handle = None
        self._initialize_file()

    def _initialize_file(self):
        """Initialize the output file with headers if needed."""
        try:
            # Ensure parent directory exists
            self.output_file.parent.mkdir(parents=True, exist_ok=True)

            if self.format_type == "txt":
                self.file_handle = open(
                    self.output_file, "w", encoding="utf-8", buffering=1
                )
                if self.detected_language and not self.multilingual:
                    self.file_handle.write(
                        f"[Detected language: {self.detected_language}]\n\n"
                    )
                    self.file_handle.flush()
                elif self.multilingual:
                    self.file_handle.write(
                        "[Multilingual transcription - language shown in brackets]\n\n"
                    )
                    self.file_handle.flush()

            elif self.format_type == "vtt":
                self.file_handle = open(
                    self.output_file, "w", encoding="utf-8", buffering=1
                )
                self.file_handle.write("WEBVTT\n\n")
                self.file_handle.flush()

            elif self.format_type == "srt":
                self.file_handle = open(
                    self.output_file, "w", encoding="utf-8", buffering=1
                )

        except Exception as e:
            console.print(
                f"[red]Error creating output file {self.output_file}: {e}[/red]"
            )
            raise

    def write_segment(self, segment):
        """Write a single segment to the output file."""
        if not self.file_handle:
            return

        try:
            self.segment_count += 1

            if self.format_type == "txt":
                if self.multilingual and hasattr(segment, "language"):
                    self.file_handle.write(
                        f"[{segment.language}] {segment.text.strip()}\n"
                    )
                else:
                    self.file_handle.write(f"{segment.text.strip()}\n")
                self.file_handle.flush()

            elif self.format_type == "srt":
                self.file_handle.write(f"{self.segment_count}\n")
                start = format_timestamp(segment.start).replace(".", ",")
                end = format_timestamp(segment.end).replace(".", ",")
                self.file_handle.write(f"{start} --> {end}\n")
                if self.multilingual and hasattr(segment, "language"):
                    self.file_handle.write(
                        f"[{segment.language}] {segment.text.strip()}\n\n"
                    )
                else:
                    self.file_handle.write(f"{segment.text.strip()}\n\n")
                self.file_handle.flush()

            elif self.format_type == "vtt":
                start = format_timestamp(segment.start)
                end = format_timestamp(segment.end)
                self.file_handle.write(f"{start} --> {end}\n")
                if self.multilingual and hasattr(segment, "language"):
                    self.file_handle.write(
                        f"[{segment.language}] {segment.text.strip()}\n\n"
                    )
                else:
                    self.file_handle.write(f"{segment.text.strip()}\n\n")
                self.file_handle.flush()

        except Exception as e:
            console.print(f"[yellow]Warning: Error writing segment: {e}[/yellow]")

    def close(self):
        """Close the output file."""
        if self.file_handle:
            try:
                self.file_handle.close()
            except Exception:
                pass


def format_timestamp(seconds):
    """Convert seconds to SRT/VTT timestamp format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"


def create_progress_bar():
    """Create a rich progress bar with custom columns."""
    return Progress(
        SpinnerColumn(spinner_name="dots12", style="cyan"),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=40, style="cyan", complete_style="green"),
        TaskProgressColumn(),
        "•",
        TimeElapsedColumn(),
        "•",
        TimeRemainingColumn(),
        console=console,
        refresh_per_second=10,
    )


class GracefulInterruptHandler:
    """Handle graceful shutdown on interrupt."""

    def __init__(
        self, checkpoint: TranscriptionCheckpoint, writers: Optional[Dict] = None
    ):
        self.checkpoint = checkpoint
        self.writers = writers or {}
        self.interrupted = False
        self.original_sigint = None

    def __enter__(self):
        self.original_sigint = signal.signal(signal.SIGINT, self._handle_interrupt)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        signal.signal(signal.SIGINT, self.original_sigint)

    def _handle_interrupt(self, signum, frame):
        """Handle interrupt signal."""
        self.interrupted = True
        console.print("\n[yellow]Interrupt received! Saving progress...[/yellow]")

        # Close all writers to ensure data is flushed
        if self.writers:
            for fmt, writer in self.writers.items():
                try:
                    writer.close()
                    console.print(f"[green]✓ Saved {fmt} output[/green]")
                except Exception as e:
                    console.print(
                        f"[yellow]Warning: Error closing {fmt} writer: {e}[/yellow]"
                    )

        # Save checkpoint
        self.checkpoint.save()
        console.print("[green]Progress saved. You can resume later.[/green]")
        sys.exit(0)


def check_file_exists_with_retry(file_path: Path, max_retries: int = 3) -> bool:
    """Check if file exists with retries for resilience."""
    for i in range(max_retries):
        try:
            return file_path.exists()
        except Exception as e:
            if i < max_retries - 1:
                time.sleep(0.5)
            else:
                console.print(f"[red]Error accessing file {file_path}: {e}[/red]")
                return False
    return False


def is_audio_file(file_path: Path, audio_extensions: set) -> bool:
    """Check if a file is an audio file based on its extension."""
    return file_path.is_file() and file_path.suffix.lower() in audio_extensions


def run_file_transcription(args):
    """Run file transcription with streaming output and resume capability."""
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

    # Handle output directory
    use_input_dir_as_output = args.output is None

    # Parse input path
    input_path = Path(args.input)

    # Determine if input is a file or directory
    is_single_file = input_path.is_file()

    # Find audio files based on input type
    audio_files = []
    input_type_display = ""

    if is_single_file:
        # Single file mode
        if is_audio_file(input_path, AUDIO_EXTENSIONS):
            audio_files.append(input_path)
            input_type_display = "Single file"
        else:
            console.print(
                Panel(
                    f"[yellow]File is not a supported audio format: {input_path}[/yellow]\n"
                    f"[white]Supported formats: {', '.join(sorted(AUDIO_EXTENSIONS))}[/white]",
                    title="[bold red]Invalid Audio File[/bold red]",
                    border_style="red",
                )
            )
            return
    else:
        # Directory mode
        if not input_path.exists():
            input_path.mkdir(parents=True, exist_ok=True)
            console.print(
                Panel(
                    f"[yellow]Created input directory: {input_path}[/yellow]\n"
                    "[white]Please place your audio files in this directory and run the script again.[/white]",
                    title="[bold red]No Input Directory[/bold red]",
                    border_style="red",
                )
            )
            return

        # Find all audio files in the directory
        for file in input_path.iterdir():
            if is_audio_file(file, AUDIO_EXTENSIONS):
                audio_files.append(file)

        input_type_display = "Directory"

    if not audio_files:
        console.print(
            Panel(
                f"[yellow]No audio files found in {input_path}[/yellow]\n"
                f"[white]Supported formats: {', '.join(sorted(AUDIO_EXTENSIONS))}[/white]",
                title="[bold red]No Audio Files[/bold red]",
                border_style="red",
            )
        )
        return

    # Determine checkpoint location
    if is_single_file:
        # For single file, use the parent directory for checkpoint
        checkpoint_path = (
            input_path.parent if use_input_dir_as_output else Path(args.output)
        )
    else:
        # For directory, use the directory itself or output directory
        checkpoint_path = input_path if use_input_dir_as_output else Path(args.output)

    # Create output directory if specified and doesn't exist
    if not use_input_dir_as_output:
        output_path = Path(args.output)
        output_path.mkdir(parents=True, exist_ok=True)

    # Initialize checkpoint system
    checkpoint = TranscriptionCheckpoint(checkpoint_path)

    # Check for existing checkpoint
    resume_from_checkpoint = False
    if checkpoint.load():
        # Show checkpoint info
        checkpoint_time = checkpoint.data.get("timestamp", "Unknown")
        completed_files = len(checkpoint.data.get("files_completed", []))
        current_file = checkpoint.data.get("current_file", "None")

        console.print(
            Panel(
                f"[yellow]Found previous transcription session[/yellow]\n"
                f"[white]Time: {checkpoint_time}[/white]\n"
                f"[white]Files completed: {completed_files}[/white]\n"
                f"[white]Current file: {current_file}[/white]",
                title="[bold cyan]Resume Session?[/bold cyan]",
                border_style="cyan",
            )
        )

        resume_from_checkpoint = Confirm.ask(
            "Do you want to resume from where you left off?", default=True
        )

        if not resume_from_checkpoint:
            checkpoint.clear()
            checkpoint = TranscriptionCheckpoint(checkpoint_path)

    # Filter out completed files if resuming
    if resume_from_checkpoint:
        completed_files = checkpoint.data.get("files_completed", [])
        audio_files = [f for f in audio_files if f.name not in completed_files]

        if completed_files:
            console.print(
                f"[green]Skipping {len(completed_files)} already completed files[/green]"
            )

    # Display configuration
    config_table = Table(title="Transcription Configuration", show_header=False)
    config_table.add_column("Setting", style="cyan", width=20)
    config_table.add_column("Value", style="green")

    config_table.add_row("Input Type", input_type_display)
    config_table.add_row("Input Path", str(input_path))
    config_table.add_row("Files to Process", str(len(audio_files)))
    config_table.add_row("Language Mode", language_display)
    config_table.add_row("Output Format", args.format.upper())
    config_table.add_row("Model Size", args.model.upper())
    if use_input_dir_as_output:
        config_table.add_row("Output Location", "Same as input files")
    else:
        config_table.add_row("Output Location", str(args.output))
    if args.multilingual:
        config_table.add_row("Multilingual", "ENABLED")
    if resume_from_checkpoint:
        config_table.add_row("Resume Mode", "ENABLED")

    console.print("\n")
    console.print(config_table)
    console.print("\n")

    # Save settings to checkpoint
    checkpoint.data["settings"] = {
        "language": args.language,
        "model": args.model,
        "format": args.format,
        "multilingual": args.multilingual,
        "output_same_as_input": use_input_dir_as_output,
        "is_single_file": is_single_file,
    }

    # Load the Whisper model with progress
    with console.status(
        "[bold cyan]Loading Whisper model...[/bold cyan]", spinner="dots12"
    ):
        model = WhisperModel(args.model, device="cpu", compute_type="int8")
        console.print("✅ [green]Model loaded successfully![/green]")

    # Process each audio file
    console.print("\n[bold cyan]Starting transcription...[/bold cyan]\n")

    # Statistics tracking
    total_duration = 0
    total_process_time = 0
    successful_files = 0
    failed_files = 0
    overall_start_time = time.time()

    # Track current writers globally for interrupt handling
    current_writers = {}

    # Set up graceful interrupt handler
    with GracefulInterruptHandler(checkpoint, current_writers):
        # Create progress bar
        with create_progress_bar() as progress:
            # Main task for overall progress
            main_task = progress.add_task(
                "[cyan]Transcribing files", total=len(audio_files)
            )

            # Secondary task for current file segments
            segment_task = progress.add_task(
                "[yellow]Current file segments", visible=False
            )

            for audio_file in audio_files:
                # Clear current writers for new file
                current_writers.clear()

                try:
                    # Check if input file still exists
                    if not check_file_exists_with_retry(audio_file):
                        console.print(
                            f"[red]✗[/red] Input file not found: {audio_file.name}"
                        )
                        failed_files += 1
                        progress.update(main_task, advance=1)
                        continue

                    # Set current file in checkpoint
                    checkpoint.set_current_file(audio_file.name)

                    # Update progress description
                    progress.update(
                        segment_task,
                        description=f"[yellow]Processing: {audio_file.name}",
                        visible=True,
                        completed=0,
                        total=100,  # Will be updated with actual segment count
                    )

                    start_time = time.time()

                    # Get file info first
                    segments_preview, info = model.transcribe(
                        str(audio_file),
                        language=selected_language,
                        beam_size=5,
                        vad_filter=True,
                        vad_parameters=dict(min_silence_duration_ms=500),
                        multilingual=args.multilingual,
                    )

                    # Get detected language
                    detected_lang = info.language if selected_language is None else None

                    # Process each format with streaming output
                    formats_to_process = (
                        ["txt", "srt", "vtt"] if args.format == "all" else [args.format]
                    )

                    # Create writers for all formats
                    writers = {}
                    for fmt in formats_to_process:
                        ext = "txt" if fmt == "txt" else fmt

                        # Determine output file location
                        if use_input_dir_as_output:
                            # Save in the same directory as the input file
                            output_file = audio_file.parent / f"{audio_file.stem}.{ext}"
                        else:
                            # Save in the specified output directory
                            output_file = Path(args.output) / f"{audio_file.stem}.{ext}"

                        try:
                            writers[fmt] = StreamingTranscriptionWriter(
                                output_file, fmt, detected_lang, args.multilingual
                            )
                        except Exception as e:
                            console.print(
                                f"[red]Error creating {fmt} writer: {e}[/red]"
                            )

                    # Update current writers for interrupt handler
                    current_writers.update(writers)

                    # Process segments as they are generated (true streaming)
                    segment_count = 0
                    total_segments = 0  # We'll count as we go

                    # Update progress with estimated segments based on duration
                    # Rough estimate: 1 segment per 10-15 seconds of audio
                    estimated_segments = max(1, int(info.duration / 12))
                    progress.update(segment_task, total=estimated_segments)

                    try:
                        for segment in segments_preview:
                            segment_count += 1
                            total_segments = segment_count

                            # Write to all format writers
                            for fmt, writer in writers.items():
                                try:
                                    writer.write_segment(segment)
                                except Exception as e:
                                    console.print(
                                        f"[yellow]Warning: Error writing {fmt} segment: {e}[/yellow]"
                                    )

                            # Update progress
                            if segment_count > estimated_segments:
                                # Adjust estimate if we have more segments than expected
                                progress.update(segment_task, total=segment_count + 5)
                            progress.update(segment_task, completed=segment_count)

                            # Save segment to checkpoint
                            checkpoint.add_segment(
                                {
                                    "text": segment.text,
                                    "start": segment.start,
                                    "end": segment.end,
                                    "language": getattr(segment, "language", None),
                                }
                            )

                    finally:
                        # Close all writers
                        for writer in writers.values():
                            writer.close()

                    # Calculate processing time
                    process_time = time.time() - start_time

                    # Update statistics
                    total_duration += info.duration
                    total_process_time += process_time
                    successful_files += 1

                    # Mark file as complete in checkpoint
                    checkpoint.mark_file_complete(audio_file.name)

                    # Show summary for this file
                    speed = info.duration / process_time if process_time > 0 else 0
                    console.print(
                        f"[green]✓[/green] {audio_file.name} "
                        f"[dim]({info.duration:.1f}s @ {speed:.1f}x speed, {total_segments} segments)[/dim]"
                    )
                    if detected_lang:
                        console.print(f"  [dim]Language: {detected_lang}[/dim]")

                    # Update main progress
                    progress.update(main_task, advance=1)
                    progress.update(segment_task, visible=False)

                except Exception as e:
                    console.print(
                        f"[red]✗[/red] Error processing {audio_file.name}: {e}"
                    )
                    failed_files += 1
                    progress.update(main_task, advance=1)
                    continue

    # Clear checkpoint on successful completion
    if failed_files == 0:
        checkpoint.clear()

    # Calculate overall statistics
    overall_time = time.time() - overall_start_time

    # Display results
    console.print("\n")

    if successful_files > 0:
        output_location = (
            "same folder as input files"
            if use_input_dir_as_output
            else str(args.output)
        )
        console.print(
            Panel(
                f"[green]Transcription complete![/green]\n"
                f"Files saved to: [cyan]{output_location}[/cyan]",
                title="[bold green]Success[/bold green]",
                border_style="green",
            )
        )

    # Create summary table
    summary_table = Table(
        title="📊 Session Summary", show_header=True, header_style="bold cyan"
    )
    summary_table.add_column("Metric", style="white", width=30)
    summary_table.add_column("Value", style="yellow", justify="right")

    summary_table.add_row("Files Processed", f"{successful_files} successful")
    if failed_files > 0:
        summary_table.add_row("Failed Files", f"[red]{failed_files}[/red]")
    summary_table.add_row("Output Format", args.format.upper())
    summary_table.add_row("Language Mode", language_display)
    if args.multilingual:
        summary_table.add_row("Multilingual", "Enabled")

    console.print("\n")
    console.print(summary_table)

    # Performance statistics
    if successful_files > 0 and total_duration > 0:
        perf_table = Table(
            title="⚡ Performance Statistics",
            show_header=True,
            header_style="bold cyan",
        )
        perf_table.add_column("Metric", style="white", width=30)
        perf_table.add_column("Value", style="green", justify="right")

        perf_table.add_row(
            "Total Audio Duration",
            f"{total_duration:.1f}s ({total_duration / 60:.1f} min)",
        )
        perf_table.add_row(
            "Total Processing Time",
            f"{total_process_time:.1f}s ({total_process_time / 60:.1f} min)",
        )
        perf_table.add_row(
            "Overall Time", f"{overall_time:.1f}s ({overall_time / 60:.1f} min)"
        )

        avg_speed = total_duration / total_process_time
        perf_table.add_row("Average Speed", f"{avg_speed:.1f}x realtime")
        perf_table.add_row(
            "Time per Audio Minute",
            f"{(total_process_time / total_duration) * 60:.1f}s",
        )

        # Estimate for longer content
        hour_estimate = (total_process_time / total_duration) * 3600 / 60
        perf_table.add_row("Est. for 1 Hour Audio", f"{hour_estimate:.1f} minutes")

        console.print("\n")
        console.print(perf_table)

    console.print("\n")


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
        help="Input file or folder containing audio files (default: ./input)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output folder for transcription files (default: same as input)",
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
