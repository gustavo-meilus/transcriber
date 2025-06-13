"""File transcription with streaming output, checkpoints, and debug mode."""

import argparse
import json
import signal
import sys
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from faster_whisper import WhisperModel
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    ProgressColumn,
    SpinnerColumn,
    Task,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.prompt import Confirm
from rich.table import Table
from rich.text import Text

console = Console()

# Constants
LANGUAGE_MAP = {
    "en": "en",
    "pt": "pt",
    "auto": None,  # None means auto-detect
}

LANGUAGE_DISPLAY = {"en": "English", "pt": "Portuguese", "auto": "Auto-detect"}

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

SEGMENT_ESTIMATE_SECONDS = 12  # Rough estimate: 1 segment per 12 seconds


class TranscriptionCheckpoint:
    """Manage transcription progress checkpoints for individual files."""

    def __init__(self, audio_file_path: Path):
        """Initialize checkpoint for a specific audio file."""
        # Create checkpoint filename based on audio file
        self.audio_file_path = audio_file_path
        checkpoint_name = f".{audio_file_path.stem}_transcription_checkpoint.json"
        self.checkpoint_file = audio_file_path.parent / checkpoint_name
        self.data = {
            "audio_file": str(audio_file_path),
            "timestamp": None,
            "segments_completed": 0,
            "last_position": 0,
            "segments": [],
            "elapsed_time": 0.0,  # Total elapsed time from previous sessions
        }

    def load(self) -> bool:
        """Load checkpoint data if it exists."""
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, "r") as f:
                    self.data = json.load(f)
                return True
            except Exception:
                return False
        return False

    def save(self):
        """Save current checkpoint data."""
        self.data["timestamp"] = datetime.now().isoformat()
        try:
            with open(self.checkpoint_file, "w") as f:
                json.dump(self.data, f, indent=2)
        except Exception as e:
            console.print(f"[yellow]Warning: Could not save checkpoint: {e}[/yellow]")

    def clear(self):
        """Remove checkpoint file."""
        try:
            if self.checkpoint_file.exists():
                self.checkpoint_file.unlink()
        except Exception as e:
            console.print(
                f"[yellow]Warning: Could not remove checkpoint file: {e}[/yellow]"
            )

    def update_position(self, position: float):
        """Update the last processed position in the audio file."""
        self.data["last_position"] = position

    def add_segment(self, segment_data: dict):
        """Add a completed segment to checkpoint."""
        self.data["segments"].append(segment_data)
        self.data["segments_completed"] += 1
        if self.data["segments_completed"] % 10 == 0:
            self.save()  # Auto-save every 10 segments

    def update_elapsed_time(self, elapsed_time: float):
        """Update the total elapsed time."""
        self.data["elapsed_time"] = elapsed_time

    def get_elapsed_time(self) -> float:
        """Get the total elapsed time from previous sessions."""
        return self.data.get("elapsed_time", 0.0)


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
    """Convert seconds to timestamp format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"


def format_duration_hhmmss(seconds: float) -> str:
    """Format duration in seconds as HH:MM:SS."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


class RemainingAudioDurationColumn(ProgressColumn):
    """Custom column showing remaining audio duration that decreases second by second."""

    def __init__(self):
        super().__init__()
        self.remaining_duration = {}  # task_id -> remaining_duration
        self.last_update_time = {}  # task_id -> last_update_time
        self.lock = threading.Lock()

    def set_remaining_duration(self, task_id: int, duration: float):
        """Set the remaining audio duration for a task."""
        with self.lock:
            self.remaining_duration[task_id] = duration
            self.last_update_time[task_id] = time.time()

    def update_remaining_duration(self, task_id: int, actual_remaining: float):
        """Update remaining duration with actual value."""
        with self.lock:
            self.remaining_duration[task_id] = actual_remaining
            self.last_update_time[task_id] = time.time()

    def _get_current_remaining(self, task_id: int) -> float:
        """Get current remaining duration accounting for elapsed time."""
        if task_id not in self.remaining_duration:
            return 0.0

        # Calculate how much time has passed since last update
        elapsed_since_update = time.time() - self.last_update_time.get(
            task_id, time.time()
        )

        # Decrease remaining by elapsed time
        remaining = self.remaining_duration[task_id] - elapsed_since_update
        return max(0.0, remaining)

    def render(self, task: Task) -> Text:
        """Render the remaining audio duration."""
        with self.lock:
            remaining = self._get_current_remaining(task.id)
            if remaining > 0:
                return Text(format_duration_hhmmss(remaining), style="bold")
            else:
                return Text("00:00:00", style="bold")


def create_progress_bar():
    """Create a rich progress bar with custom columns."""
    remaining_duration_column = RemainingAudioDurationColumn()
    progress = Progress(
        SpinnerColumn(spinner_name="dots12", style="cyan"),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=40, style="cyan", complete_style="green"),
        TaskProgressColumn(),
        "•",
        TimeElapsedColumn(),
        "•",
        remaining_duration_column,
        console=console,
        refresh_per_second=1,
    )
    return progress, remaining_duration_column


class DebugTracker:
    """Track debug statistics for file transcription."""

    def __init__(self, debug_enabled: bool = False):
        self.enabled = debug_enabled
        # Overall stats
        self.total_files = 0
        self.current_file_idx = 0
        self.total_audio_duration_all = 0
        self.total_audio_processed = 0
        self.overall_elapsed_time = 0.0

        # Current file stats
        self.file_name = ""
        self.audio_duration = 0
        self.audio_position = 0
        self.current_elapsed_time = 0.0
        self.previous_elapsed_time = 0.0  # From checkpoint

        # Per-file tracking for multiple files
        self.files_data = {}  # file_idx -> file data

    def add_segment(self, start: float, end: float, process_time: float):
        """Add a segment's data to the tracker."""
        # We still need this method to be called but don't need to store data
        pass

    def set_previous_elapsed_time(self, elapsed_time: float):
        """Set the elapsed time from previous sessions."""
        self.previous_elapsed_time = elapsed_time

    def update_current_elapsed_time(self, elapsed_time: float):
        """Update the current session elapsed time."""
        self.current_elapsed_time = elapsed_time

    def update_overall_stats(
        self,
        total_files: int,
        file_idx: int,
        total_audio_duration: float,
        total_processed: float,
    ):
        """Update overall statistics."""
        self.total_files = total_files
        self.current_file_idx = file_idx
        self.total_audio_duration_all = total_audio_duration
        self.total_audio_processed = total_processed

    def update_file_stats(
        self,
        file_idx: int,
        file_name: str,
        duration: float,
        position: float,
        elapsed: float,
    ):
        """Update statistics for a specific file."""
        self.files_data[file_idx] = {
            "name": file_name,
            "duration": duration,
            "position": position,
            "elapsed": elapsed,
        }

    def build_table(self) -> Table:
        """Build the debug statistics table."""
        # Calculate overall remaining
        overall_remaining = (
            self.total_audio_duration_all
            - self.total_audio_processed
            - self.audio_position
        )
        total_elapsed_time = self.previous_elapsed_time + self.current_elapsed_time

        # Create debug table
        title = (
            f"🔧 Debug Statistics - {self.file_name}"
            if self.total_files == 1
            else "🔧 Debug Statistics"
        )
        debug_table = Table(
            title=title,
            show_header=True,
            header_style="bold magenta",
            border_style="magenta",
        )

        debug_table.add_column("Metric", style="cyan", width=35)
        debug_table.add_column("Value", style="yellow", justify="right")

        # Show overall statistics (especially useful for multiple files)
        if self.total_files > 1:
            debug_table.add_row(
                "Files Processing", f"{self.current_file_idx + 1}/{self.total_files}"
            )
            debug_table.add_row(
                "Total Audio Duration (All Files)",
                f"{self.total_audio_duration_all:.2f}s",
            )
            debug_table.add_row(
                "Total Audio Processed",
                f"{self.total_audio_processed + self.audio_position:.2f}s",
            )
            debug_table.add_row(
                "Overall Remaining Duration", f"{overall_remaining:.2f}s"
            )

            if self.total_audio_duration_all > 0:
                overall_progress = (
                    (self.total_audio_processed + self.audio_position)
                    / self.total_audio_duration_all
                    * 100
                )
                debug_table.add_row("Overall Progress", f"{overall_progress:.1f}%")

            # Overall ETA
            if self.audio_position > 0 and self.current_elapsed_time > 0:
                rate = self.audio_position / self.current_elapsed_time
                if rate > 0:
                    overall_eta = overall_remaining / rate
                    debug_table.add_row(
                        "Overall Time Remaining",
                        f"{overall_eta:.1f}s ({overall_eta / 60:.1f}m)",
                    )

            # Add separator for current file stats
            debug_table.add_row("─" * 35, "─" * 20, style="dim")
            debug_table.add_row("Current File", self.file_name, style="bold")

        # Current file statistics
        remaining_duration = self.audio_duration - self.audio_position

        debug_table.add_row("File Audio Duration", f"{self.audio_duration:.2f}s")
        debug_table.add_row("File Current Position", f"{self.audio_position:.2f}s")
        debug_table.add_row("File Remaining Duration", f"{remaining_duration:.2f}s")

        if self.audio_duration > 0:
            progress_pct = self.audio_position / self.audio_duration * 100
            debug_table.add_row("File Progress", f"{progress_pct:.1f}%")
        else:
            debug_table.add_row("File Progress", "0.0%")

        # Format elapsed times as HH:MM:SS
        current_elapsed_formatted = format_duration_hhmmss(self.current_elapsed_time)
        total_elapsed_formatted = format_duration_hhmmss(total_elapsed_time)

        debug_table.add_row("Current Elapsed Time", current_elapsed_formatted)
        debug_table.add_row("Total Elapsed Time", total_elapsed_formatted)

        # Calculate file-specific ETA
        if self.audio_position > 0 and self.current_elapsed_time > 0:
            # Calculate rate: audio processed per second
            rate = self.audio_position / self.current_elapsed_time
            if rate > 0:
                eta_seconds = remaining_duration / rate
                debug_table.add_row(
                    "File Estimated Time Remaining",
                    f"{eta_seconds:.1f}s ({eta_seconds / 60:.1f}m)",
                )
            else:
                debug_table.add_row("File Estimated Time Remaining", "0.0s (0.0m)")
        else:
            debug_table.add_row("File Estimated Time Remaining", "0.0s (0.0m)")

        return debug_table


class ProgressWithDebug:
    """Combines progress bar and debug statistics in a single live display."""

    def __init__(self, debug_enabled: bool = False):
        self.debug_enabled = debug_enabled
        self.progress, self.remaining_duration_column = create_progress_bar()
        self.debug_tracker = DebugTracker(debug_enabled) if debug_enabled else None
        self.live = None

    def start(self):
        """Start the live display."""
        if self.debug_enabled and self.debug_tracker:
            # Create a group that combines progress and debug table
            self.live = Live(
                self.get_renderable(),
                console=console,
                refresh_per_second=1,
                transient=False,
            )
            self.live.start()
            # Return progress for context manager compatibility
            return self.progress
        else:
            # Just use progress bar without debug
            return self.progress

    def stop(self):
        """Stop the live display."""
        if self.live:
            self.live.stop()

    def get_renderable(self):
        """Get the combined renderable for live display."""
        if self.debug_enabled and self.debug_tracker:
            return Group(
                self.progress,
                "",  # Empty line for spacing
                self.debug_tracker.build_table(),
            )
        return self.progress

    def update_debug(
        self,
        file_name: str,
        audio_duration: float,
        audio_position: float,
        elapsed_time: float,
        total_files: int = 1,
        file_idx: int = 0,
        total_audio_duration_all: float = 0,
        total_audio_processed: float = 0,
    ):
        """Update debug statistics and refresh display."""
        if self.debug_enabled and self.debug_tracker:
            self.debug_tracker.file_name = file_name
            self.debug_tracker.audio_duration = audio_duration
            self.debug_tracker.audio_position = audio_position
            self.debug_tracker.update_current_elapsed_time(elapsed_time)

            # Update overall stats for multiple files
            self.debug_tracker.update_overall_stats(
                total_files, file_idx, total_audio_duration_all, total_audio_processed
            )

            if self.live:
                self.live.update(self.get_renderable())

    def __enter__(self):
        """Enter context manager."""
        if self.debug_enabled:
            return self
        else:
            return self.progress.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager."""
        if self.debug_enabled:
            self.stop()
        else:
            self.progress.__exit__(exc_type, exc_val, exc_tb)


class GlobalInterruptHandler:
    """Global interrupt handler for the entire application."""

    _instance = None

    def __init__(self):
        self.interrupted = False
        self.original_sigint = None
        self.cleanup_handlers = []

    @classmethod
    def instance(cls):
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def install(self):
        """Install the global interrupt handler."""
        self.original_sigint = signal.signal(signal.SIGINT, self._handle_interrupt)
        return self

    def uninstall(self):
        """Restore original signal handler."""
        if self.original_sigint is not None:
            signal.signal(signal.SIGINT, self.original_sigint)
            self.original_sigint = None

    def add_cleanup_handler(self, handler):
        """Add a cleanup handler to be called on interrupt."""
        self.cleanup_handlers.append(handler)

    def remove_cleanup_handler(self, handler):
        """Remove a cleanup handler."""
        if handler in self.cleanup_handlers:
            self.cleanup_handlers.remove(handler)

    def _handle_interrupt(self, signum, frame):
        """Handle interrupt signal immediately."""
        if self.interrupted:
            # Force exit on second Ctrl+C
            console.print("\n[bold red]Force quit![/bold red]")
            sys.exit(1)

        self.interrupted = True
        console.print(
            "\n[bold yellow]🛑 Interrupt received! Gracefully shutting down...[/bold yellow]"
        )

        # Execute cleanup handlers
        if self.cleanup_handlers:
            with console.status(
                "[cyan]Saving your progress...[/cyan]", spinner="dots12"
            ) as status:
                for handler in self.cleanup_handlers:
                    try:
                        handler(status)
                    except Exception as e:
                        console.print(
                            f"  [yellow]⚠ Warning: Error during cleanup: {e}[/yellow]"
                        )

        # Final message
        console.print("\n[bold green]✅ Graceful shutdown complete![/bold green]")
        console.print("[green]Your progress has been saved where applicable.[/green]\n")
        sys.exit(0)


class GracefulInterruptHandler:
    """Handle graceful shutdown on interrupt for segment processing."""

    def __init__(
        self, checkpoint: TranscriptionCheckpoint, writers: Optional[Dict] = None
    ):
        self.checkpoint = checkpoint
        self.writers = writers or {}
        self.cleanup_handler = None

    def __enter__(self):
        # Create cleanup handler for this context
        def cleanup(status):
            # Close all writers to ensure data is flushed
            if self.writers:
                for fmt, writer in self.writers.items():
                    try:
                        status.update(f"[cyan]Saving {fmt.upper()} output...[/cyan]")
                        writer.close()
                        console.print(f"  [green]✓ Saved {fmt.upper()} output[/green]")
                        time.sleep(0.1)  # Small delay to make the progress visible
                    except Exception as e:
                        console.print(
                            f"  [yellow]⚠ Warning: Error closing {fmt} writer: {e}[/yellow]"
                        )

            # Save checkpoint
            status.update("[cyan]Saving checkpoint...[/cyan]")
            self.checkpoint.save()
            time.sleep(0.2)  # Small delay to make the progress visible
            console.print("  [green]✓ Checkpoint saved[/green]")

        self.cleanup_handler = cleanup
        GlobalInterruptHandler.instance().add_cleanup_handler(cleanup)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.cleanup_handler:
            GlobalInterruptHandler.instance().remove_cleanup_handler(
                self.cleanup_handler
            )


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


def find_audio_files_in_directory(directory: Path) -> List[Path]:
    """Find all audio files in a directory."""
    audio_files = []
    for file in directory.iterdir():
        if is_audio_file(file, AUDIO_EXTENSIONS):
            audio_files.append(file)
    return audio_files


def get_audio_files(input_path: Path) -> Tuple[List[Path], str]:
    """
    Get audio files from input path (file or directory).
    Returns tuple of (audio_files, input_type_display).
    """
    if input_path.is_file():
        if is_audio_file(input_path, AUDIO_EXTENSIONS):
            return [input_path], "Single file"
        else:
            raise ValueError(f"File is not a supported audio format: {input_path}")
    else:
        # Directory mode
        if not input_path.exists():
            input_path.mkdir(parents=True, exist_ok=True)
            raise FileNotFoundError(f"Created input directory: {input_path}")

        audio_files = find_audio_files_in_directory(input_path)
        if not audio_files:
            raise ValueError(f"No audio files found in {input_path}")

        return audio_files, "Directory"


def show_error_panel(message: str, title: str = "Error"):
    """Display error panel with rich formatting."""
    console.print(
        Panel(
            message,
            title=f"[bold red]{title}[/bold red]",
            border_style="red",
            padding=(1, 2),
        )
    )


def detect_device_and_compute_type(force_cpu: bool = False) -> Tuple[str, str]:
    """Detect if GPU is available and return appropriate device and compute type.

    Args:
        force_cpu: If True, force CPU usage even if GPU is available

    Returns:
        Tuple of (device, compute_type)
    """
    if force_cpu:
        console.print("[yellow]ℹ[/yellow] Using CPU (forced by --cpu flag)")
        return "cpu", "int8"

    try:
        import torch

        if torch.cuda.is_available():
            # Get GPU info
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = (
                torch.cuda.get_device_properties(0).total_memory / 1024**3
            )  # GB

            console.print(
                f"[green]✓[/green] GPU detected: {gpu_name} ({gpu_memory:.1f}GB)"
            )

            # Choose compute type based on GPU capability
            gpu_capability = torch.cuda.get_device_capability(0)

            # Use float16 for GPUs with compute capability >= 7.0 (Turing and newer)
            # This includes RTX 20xx, RTX 30xx, RTX 40xx, A100, etc.
            if gpu_capability[0] >= 7:
                return "cuda", "float16"
            else:
                # Older GPUs use int8 for better compatibility
                return "cuda", "int8"
        else:
            console.print(
                "[yellow]ℹ[/yellow] No GPU detected, using CPU with int8 quantization"
            )
            return "cpu", "int8"

    except ImportError:
        # PyTorch not installed, fallback to CPU
        console.print(
            "[yellow]ℹ[/yellow] PyTorch not found, using CPU with int8 quantization"
        )
        return "cpu", "int8"
    except Exception as e:
        # Any other error, fallback to CPU
        console.print(f"[yellow]⚠[/yellow] Error detecting GPU: {e}")
        console.print("[yellow]ℹ[/yellow] Falling back to CPU with int8 quantization")
        return "cpu", "int8"


def create_output_directory(output_path: Optional[str]) -> Optional[Path]:
    """Create output directory if specified."""
    if output_path is None:
        return None

    path = Path(output_path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def display_configuration(
    input_type: str,
    input_path: Path,
    file_count: int,
    language_display: str,
    output_format: str,
    model_size: str,
    output_location: str,
    multilingual: bool,
    debug: bool = False,
):
    """Display the transcription configuration."""
    config_table = Table(title="Transcription Configuration", show_header=False)
    config_table.add_column("Setting", style="cyan", width=20)
    config_table.add_column("Value", style="green")

    config_table.add_row("Input Type", input_type)
    config_table.add_row("Input Path", str(input_path))
    config_table.add_row("Files to Process", str(file_count))
    config_table.add_row("Language Mode", language_display)
    config_table.add_row("Output Format", output_format.upper())
    config_table.add_row("Model Size", model_size.upper())
    config_table.add_row("Output Location", output_location)

    if multilingual:
        config_table.add_row("Multilingual", "ENABLED")

    if debug:
        config_table.add_row("Debug Mode", "[bold red]ENABLED[/bold red]")

    console.print("\n")
    console.print(config_table)
    console.print("\n")


def load_whisper_model(model_size: str, force_cpu: bool = False) -> WhisperModel:
    """Load the Whisper model with progress indicator."""
    # Detect device and compute type
    device, compute_type = detect_device_and_compute_type(force_cpu)

    with console.status(
        f"[bold cyan]Loading Whisper {model_size} model on {device.upper()}...[/bold cyan]",
        spinner="dots12",
    ):
        model = WhisperModel(model_size, device=device, compute_type=compute_type)
        console.print(
            f"[green]✓[/green] Model loaded successfully on {device.upper()}!"
        )
    return model


def should_resume_checkpoint(
    checkpoint: TranscriptionCheckpoint, audio_file: Path
) -> Tuple[bool, float]:
    """Check if we should resume from checkpoint and get start position."""
    if not checkpoint.load():
        return False, 0

    # Show checkpoint info
    checkpoint_time = checkpoint.data.get("timestamp", "Unknown")
    segments_completed = checkpoint.data.get("segments_completed", 0)
    last_position = checkpoint.data.get("last_position", 0)
    elapsed_time = checkpoint.data.get("elapsed_time", 0)
    elapsed_formatted = format_duration_hhmmss(elapsed_time)

    console.print(
        Panel(
            f"[yellow]Found previous transcription for {audio_file.name}[/yellow]\n"
            f"[white]Time: {checkpoint_time}[/white]\n"
            f"[white]Segments completed: {segments_completed}[/white]\n"
            f"[white]Progress: {last_position:.1f}s processed[/white]\n"
            f"[white]Elapsed time: {elapsed_formatted}[/white]",
            title="[bold cyan]Resume Transcription?[/bold cyan]",
            border_style="cyan",
        )
    )

    # Ensure we have a clean prompt without any live displays
    try:
        # Check if stdin is available for interactive input
        if not sys.stdin.isatty():
            console.print(
                "[yellow]Non-interactive mode detected. Resuming automatically.[/yellow]"
            )
            return True, last_position

        resume = Confirm.ask(
            f"Resume transcription of {audio_file.name}?", default=True
        )
    except Exception as e:
        console.print(
            f"[yellow]Could not get user input: {e}. Resuming automatically.[/yellow]"
        )
        return True, last_position

    if not resume:
        checkpoint.clear()
        return False, 0

    return True, last_position


def create_output_writers(
    audio_file: Path,
    formats: List[str],
    use_input_dir: bool,
    output_dir: Optional[Path],
    detected_language: Optional[str],
    multilingual: bool,
) -> Dict[str, StreamingTranscriptionWriter]:
    """Create writers for all requested output formats."""
    writers = {}

    for fmt in formats:
        ext = "txt" if fmt == "txt" else fmt

        if use_input_dir:
            output_file = audio_file.parent / f"{audio_file.stem}.{ext}"
        else:
            output_file = output_dir / f"{audio_file.stem}.{ext}"

        try:
            writers[fmt] = StreamingTranscriptionWriter(
                output_file, fmt, detected_language, multilingual
            )
        except Exception as e:
            console.print(f"[red]Error creating {fmt} writer: {e}[/red]")

    return writers


def get_formats_to_process(format_arg: str) -> List[str]:
    """Get list of formats to process based on argument."""
    return ["txt", "srt", "vtt"] if format_arg == "all" else [format_arg]


def calculate_segment_rate(
    segment_start: float, segment_end: float, process_time: float
) -> Optional[float]:
    """Calculate processing rate for a segment."""
    segment_duration = segment_end - segment_start
    if process_time > 0 and segment_duration > 0:
        return segment_duration / process_time
    return None


def process_single_file(
    audio_file: Path,
    model: WhisperModel,
    checkpoint: TranscriptionCheckpoint,
    resume_from_checkpoint: bool,
    checkpoint_start_position: float,
    args,
    use_input_dir_as_output: bool,
    output_path: Optional[Path],
    selected_language: Optional[str],
    file_idx: int,
    total_files: int,
    progress: Progress,
    main_task,
    segment_task,
    total_audio_processed: float,
    progress_with_debug: Optional[ProgressWithDebug] = None,
    remaining_duration_column: Optional[RemainingAudioDurationColumn] = None,
    estimated_total_duration: float = 0,
) -> Tuple[bool, float, float, int]:
    """
    Process a single audio file.
    Returns: (success, audio_duration, process_time, segment_count)
    """
    # Check if file exists
    if not check_file_exists_with_retry(audio_file):
        console.print(f"[red]✗[/red] Input file not found: {audio_file.name}")
        return False, 0, 0, 0

    # Update checkpoint position
    checkpoint.update_position(checkpoint_start_position)

    start_time = time.perf_counter()

    # Initialize return values
    audio_duration = 0
    process_time = 0
    total_segments = 0

    try:
        # Transcribe
        segments_preview, info = model.transcribe(
            str(audio_file),
            language=selected_language,
            beam_size=5,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
            multilingual=args.multilingual,
        )

        audio_duration = info.duration
        detected_lang = info.language if selected_language is None else None

        # Create output writers
        formats_to_process = get_formats_to_process(args.format)
        writers = create_output_writers(
            audio_file,
            formats_to_process,
            use_input_dir_as_output,
            output_path,
            detected_lang,
            args.multilingual,
        )

        # Process segments with interrupt handler
        with GracefulInterruptHandler(checkpoint, writers):
            success, total_segments = process_segments(
                segments_preview,
                info,
                writers,
                checkpoint,
                resume_from_checkpoint,
                checkpoint_start_position,
                file_idx,
                total_files,
                progress,
                main_task,
                segment_task,
                total_audio_processed,
                audio_file.name,
                progress_with_debug if args.debug else None,
                remaining_duration_column,
                estimated_total_duration,
            )

        # Close writers
        for writer in writers.values():
            writer.close()

        process_time = time.perf_counter() - start_time

        if success:
            checkpoint.clear()
            show_file_summary(
                audio_file, audio_duration, process_time, total_segments, detected_lang
            )

        return success, audio_duration, process_time, total_segments

    except Exception as e:
        console.print(f"[red]✗[/red] Error processing {audio_file.name}: {e}")
        process_time = time.perf_counter() - start_time
        return False, audio_duration, process_time, 0


def process_segments(
    segments_preview,
    info,
    writers: Dict[str, StreamingTranscriptionWriter],
    checkpoint: TranscriptionCheckpoint,
    resume_from_checkpoint: bool,
    checkpoint_start_position: float,
    file_idx: int,
    total_files: int,
    progress: Progress,
    main_task,
    segment_task,
    total_audio_processed: float,
    file_name: str,
    progress_with_debug: Optional[ProgressWithDebug] = None,
    remaining_duration_column: Optional[RemainingAudioDurationColumn] = None,
    estimated_total_duration: float = 0,
) -> Tuple[bool, int]:
    """Process all segments. Returns (success, segment_count)."""
    segment_count = 0
    audio_position = checkpoint_start_position if resume_from_checkpoint else 0
    rate_queue = deque(maxlen=10)

    # Track elapsed time
    session_start_time = time.perf_counter()
    previous_elapsed_time = (
        checkpoint.get_elapsed_time() if resume_from_checkpoint else 0.0
    )

    # Initialize debug tracker
    debug_tracker = (
        DebugTracker(progress_with_debug.debug_enabled) if progress_with_debug else None
    )

    # Set previous elapsed time in debug tracker
    if debug_tracker and progress_with_debug and progress_with_debug.debug_enabled:
        debug_tracker.set_previous_elapsed_time(previous_elapsed_time)
        if progress_with_debug.debug_tracker:
            progress_with_debug.debug_tracker.set_previous_elapsed_time(
                previous_elapsed_time
            )

    # Initialize debug display with initial values
    if progress_with_debug and progress_with_debug.debug_enabled:
        progress_with_debug.update_debug(
            file_name,
            info.duration,
            audio_position,
            0.0,
            total_files,
            file_idx,
            estimated_total_duration,
            total_audio_processed,
        )

    # Initialize progress for this file
    if file_idx == 0:
        # First file - update with actual duration if different from estimate
        current_total = progress.tasks[main_task].total
        actual_total = total_audio_processed + info.duration

        # For single file, use exact audio duration for main task
        if total_files == 1:
            progress.update(
                main_task,
                total=info.duration,
                completed=audio_position,  # Start from checkpoint position
                description=f"[cyan]File {file_idx + 1}/{total_files}",
                visible=True,
            )
        else:
            # Multiple files
            if actual_total > current_total:
                # Actual duration is more than estimate, update total
                progress.update(main_task, total=actual_total)
            progress.update(
                main_task,
                description=f"[cyan]File {file_idx + 1}/{total_files}",
                visible=True,
            )
    else:
        # For subsequent files, check if we need to expand the total
        current_total = progress.tasks[main_task].total
        actual_total = total_audio_processed + info.duration
        if actual_total > current_total:
            # Actual duration is more than estimate, update total
            progress.update(main_task, total=actual_total)
        progress.update(
            main_task,
            description=f"[cyan]File {file_idx + 1}/{total_files}",
            visible=True,
        )

    # Set initial remaining audio duration for current file
    # Use the progress bar's total for accurate remaining calculation
    progress_total = progress.tasks[main_task].total
    current_position = total_audio_processed + audio_position
    initial_remaining = progress_total - current_position

    # For single file, use actual file remaining for both tasks
    if total_files == 1:
        file_remaining = info.duration - audio_position
        if remaining_duration_column:
            remaining_duration_column.set_remaining_duration(main_task, file_remaining)
            remaining_duration_column.set_remaining_duration(
                segment_task, file_remaining
            )
    else:
        # Multiple files: track separately
        if remaining_duration_column:
            remaining_duration_column.set_remaining_duration(
                main_task, initial_remaining
            )
            # Also set for segment task to show remaining for current file
            file_remaining = info.duration - audio_position
            remaining_duration_column.set_remaining_duration(
                segment_task, file_remaining
            )

    max(1, int(info.duration / SEGMENT_ESTIMATE_SECONDS))

    # For both single and multiple files, make segment task track audio position
    progress.update(
        segment_task,
        total=info.duration,
        completed=audio_position,  # Start from checkpoint position
    )

    # Update segment task description
    progress.update(
        segment_task,
        description=f"[yellow]Processing: {file_name}",
        visible=True,
    )

    try:
        segment_iterator = iter(segments_preview)
        while True:
            try:
                # Start timing BEFORE getting the segment (this is where transcription happens)
                segment_process_start = time.perf_counter()

                # Get next segment - this triggers the actual transcription
                segment = next(segment_iterator)

                # Skip already processed segments
                if resume_from_checkpoint and segment.end <= checkpoint_start_position:
                    continue

                segment_count += 1

                # Write segment
                for fmt, writer in writers.items():
                    try:
                        writer.write_segment(segment)
                    except Exception as e:
                        console.print(
                            f"[yellow]Warning: Error writing {fmt} segment: {e}[/yellow]"
                        )

                # Calculate rate with actual transcription time included
                segment_process_time = time.perf_counter() - segment_process_start
                audio_position = segment.end

                # Update remaining audio duration based on total progress
                total_position = total_audio_processed + audio_position
                progress_total = progress.tasks[main_task].total
                total_remaining = progress_total - total_position

                # For single file processing, use actual file remaining for both tasks
                if total_files == 1:
                    file_remaining = info.duration - audio_position
                    if remaining_duration_column:
                        remaining_duration_column.update_remaining_duration(
                            main_task, file_remaining
                        )
                        remaining_duration_column.update_remaining_duration(
                            segment_task, file_remaining
                        )
                else:
                    # Multiple files: track separately
                    if remaining_duration_column:
                        remaining_duration_column.update_remaining_duration(
                            main_task, total_remaining
                        )
                        # Also update for segment task (current file remaining)
                        file_remaining = info.duration - audio_position
                        remaining_duration_column.update_remaining_duration(
                            segment_task, file_remaining
                        )

                # Track debug data
                if debug_tracker:
                    debug_tracker.add_segment(
                        segment.start, segment.end, segment_process_time
                    )

                segment_rate = calculate_segment_rate(
                    segment.start, segment.end, segment_process_time
                )
                if segment_rate:
                    rate_queue.append(segment_rate)
                    sum(rate_queue) / len(rate_queue) if rate_queue else 0

                # Update progress bars
                if total_files == 1:
                    # Single file: both bars track audio position
                    progress.update(segment_task, completed=audio_position)
                    progress.update(main_task, completed=audio_position)
                else:
                    # Multiple files: segment task tracks current file audio position, main task tracks total
                    progress.update(segment_task, completed=audio_position)
                    progress.update(
                        main_task, completed=total_audio_processed + audio_position
                    )

                # Update checkpoint
                checkpoint.update_position(audio_position)
                # Update elapsed time
                current_elapsed_time = time.perf_counter() - session_start_time
                checkpoint.update_elapsed_time(
                    previous_elapsed_time + current_elapsed_time
                )
                checkpoint.add_segment(
                    {
                        "text": segment.text,
                        "start": segment.start,
                        "end": segment.end,
                        "language": getattr(segment, "language", None),
                    }
                )

                # Update debug statistics continuously
                if progress_with_debug and progress_with_debug.debug_enabled:
                    current_elapsed_time = time.perf_counter() - session_start_time
                    progress_with_debug.update_debug(
                        file_name,
                        info.duration,
                        audio_position,
                        current_elapsed_time,
                        total_files,
                        file_idx,
                        estimated_total_duration,
                        total_audio_processed,
                    )

            except StopIteration:
                # No more segments
                break

        # Save final elapsed time to checkpoint
        final_elapsed_time = time.perf_counter() - session_start_time
        checkpoint.update_elapsed_time(previous_elapsed_time + final_elapsed_time)
        checkpoint.save()  # Save the final state

        # Set remaining duration to 0 when complete
        if remaining_duration_column:
            remaining_duration_column.set_remaining_duration(main_task, 0)
            remaining_duration_column.set_remaining_duration(segment_task, 0)

        # Ensure progress bars show 100% completion
        if total_files == 1:
            # Single file: both bars should be at 100%
            progress.update(segment_task, completed=info.duration)
            progress.update(main_task, completed=info.duration)
        else:
            # Multiple files: segment task shows 100% for current file
            progress.update(segment_task, completed=info.duration)

        # Final debug display update
        if progress_with_debug and progress_with_debug.debug_enabled:
            progress_with_debug.update_debug(
                file_name,
                info.duration,
                audio_position,
                final_elapsed_time,
                total_files,
                file_idx,
                estimated_total_duration,
                total_audio_processed,
            )

        return True, segment_count

    except Exception as e:
        console.print(f"[red]Error processing segments: {e}[/red]")
        return False, segment_count


def show_file_summary(
    audio_file: Path,
    duration: float,
    process_time: float,
    segment_count: int,
    detected_language: Optional[str],
):
    """Display summary for a processed file."""
    speed = duration / process_time if process_time > 0 else 0
    console.print(
        f"[green]✓[/green] {audio_file.name} "
        f"[dim]({duration:.1f}s @ {speed:.1f}x speed, {segment_count} segments)[/dim]"
    )
    if detected_language:
        console.print(f"  [dim]Language: {detected_language}[/dim]")


def display_final_results(
    successful_files: int,
    failed_files: int,
    total_duration: float,
    total_process_time: float,
    overall_time: float,
    output_format: str,
    language_display: str,
    multilingual: bool,
    use_input_dir_as_output: bool,
    output_path: Optional[str],
):
    """Display final results and statistics."""
    console.print("\n")

    if successful_files > 0:
        output_location = (
            "same folder as input files"
            if use_input_dir_as_output
            else str(output_path)
        )
        console.print(
            Panel(
                f"[green]Transcription complete![/green]\n"
                f"Files saved to: [cyan]{output_location}[/cyan]",
                title="[bold green]Success[/bold green]",
                border_style="green",
            )
        )

    # Session summary
    summary_table = Table(
        title="📊 Session Summary", show_header=True, header_style="bold cyan"
    )
    summary_table.add_column("Metric", style="white", width=30)
    summary_table.add_column("Value", style="yellow", justify="right")

    summary_table.add_row("Files Processed", f"{successful_files} successful")
    if failed_files > 0:
        summary_table.add_row("Failed Files", f"[red]{failed_files}[/red]")
    summary_table.add_row("Output Format", output_format.upper())
    summary_table.add_row("Language Mode", language_display)
    if multilingual:
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

        hour_estimate = (total_process_time / total_duration) * 3600 / 60
        perf_table.add_row("Est. for 1 Hour Audio", f"{hour_estimate:.1f} minutes")

        console.print("\n")
        console.print(perf_table)

    console.print("\n")


def estimate_audio_duration(file_path: Path) -> float:
    """Estimate audio duration based on file size (rough approximation)."""
    # Conservative estimate: 1MB ≈ 10 seconds for typical audio
    # This is just for initial progress estimation and will be corrected
    # with actual duration once transcription starts
    try:
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        # Use a more conservative estimate to avoid huge overestimates
        return file_size_mb * 10
    except Exception:
        return 60  # Default to 1 minute if can't read file


def run_file_transcription(args):
    """Run file transcription with streaming output and resume capability."""
    # Install global interrupt handler immediately
    global_handler = GlobalInterruptHandler.instance()
    global_handler.install()

    try:
        selected_language = LANGUAGE_MAP[args.language]
        language_display = LANGUAGE_DISPLAY[args.language]

        # Handle output directory
        use_input_dir_as_output = args.output is None

        # Parse input path
        input_path = Path(args.input)

        # Get audio files
        try:
            audio_files, input_type_display = get_audio_files(input_path)
        except ValueError as e:
            show_error_panel(
                f"{str(e)}\nSupported formats: {', '.join(sorted(AUDIO_EXTENSIONS))}",
                "Invalid Audio File",
            )
            return
        except FileNotFoundError as e:
            show_error_panel(
                f"{str(e)}\nPlease place your audio files in this directory and run the script again.",
                "No Input Directory",
            )
            return

        # Create output directory if specified
        try:
            output_path = create_output_directory(args.output)
        except Exception as e:
            show_error_panel(f"Failed to create output directory: {e}", "Output Error")
            return

        # Load the Whisper model with progress
        model = load_whisper_model(args.model, getattr(args, "cpu", False))

        # Display configuration
        display_configuration(
            input_type_display,
            input_path,
            len(audio_files),
            language_display,
            args.format,
            args.model,
            "same folder as input files"
            if use_input_dir_as_output
            else str(args.output),
            args.multilingual,
            args.debug,
        )

        # Process each audio file
        console.print("\n[bold cyan]Starting transcription...[/bold cyan]\n")

        # Statistics tracking
        total_duration = 0
        total_process_time = 0
        successful_files = 0
        failed_files = 0
        overall_start_time = time.perf_counter()

        # Check all checkpoints BEFORE starting the progress display
        checkpoints_info = []
        estimated_total_duration = 0
        for audio_file in audio_files:
            checkpoint = TranscriptionCheckpoint(audio_file)
            resume_from_checkpoint, checkpoint_start_position = (
                should_resume_checkpoint(checkpoint, audio_file)
            )
            checkpoints_info.append(
                {
                    "checkpoint": checkpoint,
                    "resume": resume_from_checkpoint,
                    "start_position": checkpoint_start_position,
                }
            )
            # Estimate duration for progress tracking
            estimated_total_duration += estimate_audio_duration(audio_file)

        # Create progress with debug display
        progress_with_debug = ProgressWithDebug(args.debug)

        with progress_with_debug as display:
            # Get the progress object and remaining duration column
            if args.debug:
                progress = progress_with_debug.start()
                remaining_duration_column = (
                    progress_with_debug.remaining_duration_column
                )
            else:
                progress = display
                remaining_duration_column = (
                    progress_with_debug.remaining_duration_column
                )

            # Main task for overall progress
            main_task = progress.add_task(
                "[cyan]Preparing transcription...",
                total=estimated_total_duration,  # Use estimated total
                completed=0,
                visible=True,  # Show from the start
            )

            # Secondary task for current file segments
            segment_task = progress.add_task(
                "[yellow]Current file segments", visible=False
            )

            # Track overall progress
            total_audio_processed = 0

            for file_idx, (audio_file, checkpoint_info) in enumerate(
                zip(audio_files, checkpoints_info)
            ):
                # Get checkpoint info
                checkpoint = checkpoint_info["checkpoint"]
                resume_from_checkpoint = checkpoint_info["resume"]
                checkpoint_start_position = checkpoint_info["start_position"]

                # Process the file
                success, audio_duration, process_time, segment_count = (
                    process_single_file(
                        audio_file,
                        model,
                        checkpoint,
                        resume_from_checkpoint,
                        checkpoint_start_position,
                        args,
                        use_input_dir_as_output,
                        output_path,
                        selected_language,
                        file_idx,
                        len(audio_files),
                        progress,
                        main_task,
                        segment_task,
                        total_audio_processed,
                        progress_with_debug if args.debug else None,
                        remaining_duration_column,
                        estimated_total_duration,
                    )
                )

                if success:
                    # Update statistics
                    total_duration += audio_duration
                    total_process_time += process_time
                    successful_files += 1

                    # Update total audio processed
                    total_audio_processed += audio_duration

                    # Update main progress to completion for this file
                    progress.update(main_task, completed=total_audio_processed)
                    progress.update(segment_task, visible=False)

                else:
                    failed_files += 1

                    # Skip this file's duration in progress if available
                    if audio_duration > 0:
                        total_audio_processed += audio_duration
                        progress.update(main_task, completed=total_audio_processed)

        # Calculate overall statistics
        overall_time = time.perf_counter() - overall_start_time

        # Display final results
        display_final_results(
            successful_files,
            failed_files,
            total_duration,
            total_process_time,
            overall_time,
            args.format,
            language_display,
            args.multilingual,
            use_input_dir_as_output,
            output_path,
        )

    except Exception as e:
        console.print(f"[red]Error in run_file_transcription: {e}[/red]")
    finally:
        # Uninstall global handler
        global_handler.uninstall()


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
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode for detailed segment processing statistics",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU usage even if GPU is available",
    )
    args = parser.parse_args()
    run_file_transcription(args)
