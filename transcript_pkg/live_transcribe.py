"""Live transcription module for capturing and transcribing system audio."""

import argparse
import json
import queue
import subprocess
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Optional, TextIO

import numpy as np
import scipy.signal as scipy_signal
import sounddevice as sd
from faster_whisper import WhisperModel
from rich import box
from rich.align import Align
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.prompt import IntPrompt
from rich.table import Table
from rich.text import Text

from .file_transcribe import (
    GlobalInterruptHandler,
    detect_device_and_compute_type,
    format_duration_hhmmss,
    format_timestamp,
    show_error_panel,
)

console = Console()

# Constants
SAMPLE_RATE = 16000
BUFFER_DURATION = 5  # seconds of audio to buffer before transcribing
BUFFER_SIZE = int(SAMPLE_RATE * BUFFER_DURATION)
SILENCE_THRESHOLD = 0.001
TARGET_MONITOR = (
    "alsa_output.usb-Generic_TAE2146_20210726905926-00.analog-stereo.monitor"
)

LANGUAGE_MAP = {
    "en": "en",
    "pt": "pt",
    "auto": None,  # None means auto-detect
}

LANGUAGE_DISPLAY = {"en": "English", "pt": "Portuguese", "auto": "Auto-detect"}


class LiveSessionCheckpoint:
    """Manage live transcription session checkpoints for recovery."""

    def __init__(self, output_dir: Optional[Path] = None):
        """Initialize checkpoint for live session."""
        # Create checkpoint filename based on session start time
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_dir = output_dir if output_dir else Path.cwd()
        self.checkpoint_file = (
            checkpoint_dir / f".live_session_{session_id}_checkpoint.json"
        )
        self.data = {
            "session_id": session_id,
            "start_time": datetime.now().isoformat(),
            "last_update": None,
            "total_duration": 0,
            "transcription_count": 0,
            "segment_count": 0,
            "languages_detected": {},
            "transcriptions": [],  # Store last N transcriptions
        }
        self.max_transcriptions = 100  # Keep last 100 transcriptions

    def update(
        self,
        duration: float,
        transcription_count: int,
        segment_count: int,
        languages: dict[str, int],
        last_transcription: Optional[str] = None,
    ):
        """Update checkpoint data."""
        self.data["last_update"] = datetime.now().isoformat()
        self.data["total_duration"] = duration
        self.data["transcription_count"] = transcription_count
        self.data["segment_count"] = segment_count
        self.data["languages_detected"] = languages

        if last_transcription:
            self.data["transcriptions"].append(
                {"timestamp": time.time(), "text": last_transcription}
            )
            # Keep only last N transcriptions
            if len(self.data["transcriptions"]) > self.max_transcriptions:
                self.data["transcriptions"] = self.data["transcriptions"][
                    -self.max_transcriptions :
                ]

    def save(self):
        """Save checkpoint data."""
        try:
            with open(self.checkpoint_file, "w") as f:
                json.dump(self.data, f, indent=2)
        except Exception:
            pass  # Silently fail for checkpoints

    def clear(self):
        """Remove checkpoint file."""
        try:
            if self.checkpoint_file.exists():
                self.checkpoint_file.unlink()
        except Exception:
            pass


class StreamingLiveWriter:
    """Handles streaming output of live transcription data to multiple formats."""

    def __init__(
        self,
        output_file: Path,
        format_type: str = "txt",
        include_timestamps: bool = True,
        detected_language: Optional[str] = None,
        multilingual: bool = False,
    ):
        self.output_file = output_file
        self.format_type = format_type
        self.include_timestamps = include_timestamps
        self.detected_language = detected_language
        self.multilingual = multilingual
        self.file_handle: Optional[TextIO] = None
        self.start_time = time.time()
        self.segment_count = 0
        self._initialize_file()

    def _initialize_file(self):
        """Initialize the output file with appropriate headers."""
        try:
            # Ensure parent directory exists
            self.output_file.parent.mkdir(parents=True, exist_ok=True)

            self.file_handle = open(
                self.output_file, "w", encoding="utf-8", buffering=1
            )

            if self.format_type == "txt":
                # Write header for text format
                header = f"Live Transcription Session - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                if self.detected_language and not self.multilingual:
                    header += f"Language: {self.detected_language}\n"
                elif self.multilingual:
                    header += (
                        "Multilingual transcription - language shown in brackets\n"
                    )
                header += "=" * 60 + "\n\n"
                self.file_handle.write(header)
                self.file_handle.flush()

            elif self.format_type == "vtt":
                self.file_handle.write("WEBVTT\n\n")
                if self.multilingual:
                    self.file_handle.write("NOTE Multilingual transcription\n\n")
                self.file_handle.flush()

            elif self.format_type == "srt":
                # SRT doesn't need a header
                pass

        except Exception as e:
            console.print(
                f"[red]Error creating output file {self.output_file}: {e}[/red]"
            )
            raise

    def write_segment(
        self,
        text: str,
        language: Optional[str] = None,
        timestamp: Optional[float] = None,
    ):
        """Write a transcription segment in the appropriate format."""
        if not self.file_handle:
            return

        try:
            self.segment_count += 1

            if timestamp is None:
                timestamp = time.time() - self.start_time

            if self.format_type == "txt":
                if self.include_timestamps:
                    time_str = f"[{timestamp:06.1f}s]"
                    if self.multilingual and language:
                        line = f"{time_str} [{language}] {text}\n"
                    else:
                        line = f"{time_str} {text}\n"
                else:
                    if self.multilingual and language:
                        line = f"[{language}] {text}\n"
                    else:
                        line = f"{text}\n"
                self.file_handle.write(line)

            elif self.format_type == "srt":
                # SRT format
                self.file_handle.write(f"{self.segment_count}\n")
                start_time = format_timestamp(timestamp).replace(".", ",")
                # Estimate end time (5 seconds later or next segment)
                end_time = format_timestamp(timestamp + 5).replace(".", ",")
                self.file_handle.write(f"{start_time} --> {end_time}\n")
                if self.multilingual and language:
                    self.file_handle.write(f"[{language}] {text}\n\n")
                else:
                    self.file_handle.write(f"{text}\n\n")

            elif self.format_type == "vtt":
                # WebVTT format
                start_time = format_timestamp(timestamp)
                end_time = format_timestamp(timestamp + 5)
                self.file_handle.write(f"{start_time} --> {end_time}\n")
                if self.multilingual and language:
                    self.file_handle.write(f"[{language}] {text}\n\n")
                else:
                    self.file_handle.write(f"{text}\n\n")

            self.file_handle.flush()

        except Exception as e:
            console.print(f"[yellow]Warning: Error writing segment: {e}[/yellow]")

    def write_summary(self, summary_lines: list):
        """Write session summary to the file (only for TXT format)."""
        if not self.file_handle or self.format_type != "txt":
            return

        try:
            self.file_handle.write("\n" + "=" * 60 + "\n")
            self.file_handle.write("Session Summary\n")
            self.file_handle.write("=" * 60 + "\n\n")

            for line in summary_lines:
                # Remove rich formatting
                clean_line = line
                for tag in [
                    "[cyan]",
                    "[/cyan]",
                    "[bold cyan]",
                    "[/bold cyan]",
                    "[bold]",
                    "[/bold]",
                    "[green]",
                    "[/green]",
                    "[yellow]",
                    "[/yellow]",
                ]:
                    clean_line = clean_line.replace(tag, "")
                self.file_handle.write(clean_line + "\n")

            self.file_handle.flush()

        except Exception as e:
            console.print(f"[yellow]Warning: Error writing summary: {e}[/yellow]")

    def close(self):
        """Close the output file."""
        if self.file_handle:
            try:
                self.file_handle.close()
            except Exception:
                pass


class DebugTracker:
    """Track debug statistics for live transcription."""

    def __init__(self, debug_enabled: bool = False):
        self.enabled = debug_enabled
        self.session_start = time.time()
        self.last_transcription_time = 0
        self.transcription_times = deque(maxlen=20)  # Keep last 20 transcription times
        self.audio_buffer_sizes = deque(maxlen=50)  # Track buffer sizes
        self.silence_periods = 0
        self.active_periods = 0
        self.current_languages = {}
        self.transcription_rates = deque(maxlen=10)  # Audio seconds per real second

    def add_transcription_time(self, duration: float):
        """Add a transcription processing time."""
        self.transcription_times.append(duration)
        self.last_transcription_time = duration

    def add_buffer_size(self, size: int):
        """Track audio buffer size."""
        self.audio_buffer_sizes.append(size)

    def add_silence_period(self):
        """Increment silence period count."""
        self.silence_periods += 1

    def add_active_period(self):
        """Increment active audio period count."""
        self.active_periods += 1

    def add_transcription_rate(self, audio_duration: float, real_duration: float):
        """Add transcription rate (audio seconds processed per real second)."""
        if real_duration > 0:
            rate = audio_duration / real_duration
            self.transcription_rates.append(rate)

    def build_table(self, stats: dict) -> Table:
        """Build the debug statistics table."""
        debug_table = Table(
            title="🔧 Live Transcription Debug",
            show_header=True,
            header_style="bold magenta",
            border_style="magenta",
        )

        debug_table.add_column("Metric", style="cyan", width=35)
        debug_table.add_column("Value", style="yellow", justify="right")

        # Session info
        session_duration = time.time() - self.session_start
        debug_table.add_row(
            "Session Duration", format_duration_hhmmss(session_duration)
        )
        debug_table.add_row("Total Transcriptions", str(stats.get("transcriptions", 0)))
        debug_table.add_row("Total Segments", str(stats.get("segments", 0)))

        # Audio processing
        debug_table.add_row("─" * 35, "─" * 20, style="dim")
        debug_table.add_row("Audio Buffer Duration", f"{BUFFER_DURATION}s")

        if self.audio_buffer_sizes:
            avg_buffer = sum(self.audio_buffer_sizes) / len(self.audio_buffer_sizes)
            debug_table.add_row(
                "Avg Buffer Fill", f"{avg_buffer / BUFFER_SIZE * 100:.1f}%"
            )

        total_periods = self.silence_periods + self.active_periods
        if total_periods > 0:
            silence_pct = self.silence_periods / total_periods * 100
            debug_table.add_row(
                "Silence Periods", f"{self.silence_periods} ({silence_pct:.1f}%)"
            )
            debug_table.add_row(
                "Active Periods", f"{self.active_periods} ({100 - silence_pct:.1f}%)"
            )

        # Performance metrics
        debug_table.add_row("─" * 35, "─" * 20, style="dim")
        if self.transcription_times:
            avg_time = sum(self.transcription_times) / len(self.transcription_times)
            debug_table.add_row("Avg Transcription Time", f"{avg_time:.2f}s")
            debug_table.add_row(
                "Last Transcription Time", f"{self.last_transcription_time:.2f}s"
            )

        if self.transcription_rates:
            avg_rate = sum(self.transcription_rates) / len(self.transcription_rates)
            debug_table.add_row("Avg Processing Rate", f"{avg_rate:.1f}x realtime")

        # Language detection
        if stats.get("languages"):
            debug_table.add_row("─" * 35, "─" * 20, style="dim")
            lang_str = ", ".join([f"{k}:{v}" for k, v in stats["languages"].items()])
            debug_table.add_row("Languages Detected", lang_str)

        return debug_table


class TranscriptionUI:
    """Manages the live transcription UI with optional debug mode."""

    def __init__(self, debug_enabled: bool = False):
        self.debug_enabled = debug_enabled
        self.layout = Layout()
        self.transcription_text = Text()
        self.status_text = "Initializing..."
        self.status_style = "yellow"
        self.stats = {
            "duration": 0,
            "transcriptions": 0,
            "segments": 0,
            "languages": {},
            "last_update": None,
        }
        self.debug_tracker = DebugTracker(debug_enabled) if debug_enabled else None
        self.setup_layout()

    def setup_layout(self):
        """Setup the layout structure."""
        if self.debug_enabled:
            # Layout with debug panel
            self.layout.split_column(
                Layout(name="header", size=3),
                Layout(name="main", ratio=1),
                Layout(name="status", size=3),
                Layout(name="bottom", size=16),  # Stats + Debug
            )
            # Split bottom into stats and debug
            self.layout["bottom"].split_row(
                Layout(name="stats", ratio=1),
                Layout(name="debug", ratio=1),
            )
        else:
            # Standard layout without debug
            self.layout.split_column(
                Layout(name="header", size=3),
                Layout(name="main", ratio=1),
                Layout(name="status", size=3),
                Layout(name="stats", size=8),
            )

    def update_header(self):
        """Update the header panel."""
        header = Panel(
            Align.center(
                Text("🎙️  LIVE TRANSCRIPTION", style="bold bright_cyan"),
                vertical="middle",
            ),
            border_style="bright_cyan",
            box=box.DOUBLE,
        )
        self.layout["header"].update(header)

    def update_transcription(self, text, language=None):
        """Update the transcription display."""
        if language:
            styled_text = f"[bright_green][{language}][/bright_green] {text}"
        else:
            styled_text = text

        # Keep last 5 transcriptions (or more in debug mode)
        max_lines = 10 if self.debug_enabled else 5
        lines = self.transcription_text.plain.split("\n")
        if len(lines) > max_lines:
            lines = lines[-max_lines:]
        lines.append(styled_text)

        self.transcription_text = Text("\n".join(lines))

        panel = Panel(
            self.transcription_text,
            title="[bold]Transcription[/bold]",
            border_style="green",
            padding=(1, 2),
        )
        self.layout["main"].update(panel)

    def update_status(self, status, style="yellow"):
        """Update the status display."""
        self.status_text = status
        self.status_style = style

        status_panel = Panel(
            Align.center(
                Text(self.status_text, style=self.status_style), vertical="middle"
            ),
            border_style=self.status_style,
            height=3,
        )
        self.layout["status"].update(status_panel)

    def update_stats(
        self, duration=None, transcriptions=None, segments=None, language=None
    ):
        """Update statistics."""
        if duration is not None:
            self.stats["duration"] = duration
        if transcriptions is not None:
            self.stats["transcriptions"] = transcriptions
        if segments is not None:
            self.stats["segments"] = segments
        if language:
            self.stats["languages"][language] = (
                self.stats["languages"].get(language, 0) + 1
            )

        self.stats["last_update"] = datetime.now().strftime("%H:%M:%S")

        # Create stats table
        stats_table = Table(show_header=False, box=None, padding=(0, 1))
        stats_table.add_column(style="cyan", width=20)
        stats_table.add_column(style="yellow")

        stats_table.add_row(
            "📊 Duration", format_duration_hhmmss(self.stats["duration"])
        )
        stats_table.add_row("🎯 Transcriptions", str(self.stats["transcriptions"]))
        stats_table.add_row("📝 Segments", str(self.stats["segments"]))

        if self.stats["languages"]:
            lang_str = ", ".join(
                [f"{lang}: {count}" for lang, count in self.stats["languages"].items()]
            )
            stats_table.add_row("🌐 Languages", lang_str)

        stats_table.add_row("🕐 Last Update", self.stats["last_update"])

        stats_panel = Panel(
            stats_table, title="[bold]Statistics[/bold]", border_style="cyan"
        )
        self.layout["stats"].update(stats_panel)

        # Update debug panel if enabled
        if self.debug_enabled and self.debug_tracker:
            debug_table = self.debug_tracker.build_table(self.stats)
            self.layout["debug"].update(debug_table)

    def get_layout(self):
        """Get the current layout."""
        self.update_header()
        return self.layout


def display_configuration(
    language_display: str,
    output_format: str,
    model_size: str,
    output_files: dict[str, StreamingLiveWriter],
    multilingual: bool,
    debug: bool = False,
    no_timestamps: bool = False,
    device_name: str = "",
):
    """Display the live transcription configuration."""
    config_table = Table(title="Live Transcription Configuration", show_header=False)
    config_table.add_column("Setting", style="cyan", width=20)
    config_table.add_column("Value", style="green")

    config_table.add_row("Audio Device", device_name)
    config_table.add_row("Language Mode", language_display)
    config_table.add_row("Output Format", output_format.upper())
    config_table.add_row("Model Size", model_size.upper())

    if output_files:
        output_paths = ", ".join([str(w.output_file) for w in output_files.values()])
        config_table.add_row("Output Files", output_paths)
    else:
        config_table.add_row("Output Files", "None (display only)")

    config_table.add_row("Timestamps", "Disabled" if no_timestamps else "Enabled")

    if multilingual:
        config_table.add_row("Multilingual", "ENABLED")

    if debug:
        config_table.add_row("Debug Mode", "[bold red]ENABLED[/bold red]")

    console.print("\n")
    console.print(config_table)
    console.print("\n")


def get_available_input_devices() -> list[tuple[int, str, int, int]]:
    """Get list of available input devices.

    Returns:
        List of tuples: (index, name, channels, sample_rate)
    """
    devices = []
    try:
        all_devices = sd.query_devices()
        for i, device in enumerate(all_devices):
            # Only include devices with input channels
            if device["max_input_channels"] > 0:
                devices.append(
                    (
                        i,
                        device["name"],
                        device["max_input_channels"],
                        int(device["default_samplerate"]),
                    )
                )
    except Exception as e:
        console.print(f"[red]Error querying audio devices: {e}[/red]")

    return devices


def display_device_selection_menu(devices: list[tuple[int, str, int, int]]) -> int:
    """Display interactive device selection menu and get user choice.

    Args:
        devices: List of available input devices

    Returns:
        Selected device index
    """
    # Create a table to display devices
    table = Table(
        title="🎤 Available Audio Input Devices",
        show_header=True,
        header_style="bold cyan",
        border_style="cyan",
        box=box.ROUNDED,
    )

    table.add_column("Index", style="yellow", width=8, justify="center")
    table.add_column("Device Name", style="white", width=50)
    table.add_column("Channels", style="green", justify="center")
    table.add_column("Sample Rate", style="blue", justify="right")

    for idx, name, channels, sample_rate in devices:
        # Highlight special devices
        if "monitor" in name.lower():
            name_style = "[bold cyan]" + name + " [dim](System Audio)[/dim][/bold cyan]"
        elif "TAE2146" in name:
            name_style = (
                "[bold green]" + name + " [dim](Recommended)[/dim][/bold green]"
            )
        else:
            name_style = name

        table.add_row(str(idx), name_style, str(channels), f"{sample_rate:,} Hz")

    console.print("\n")
    console.print(table)
    console.print("\n")

    # Check for common audio monitoring devices
    monitor_devices = [d for d in devices if "monitor" in d[1].lower()]
    if monitor_devices:
        console.print(
            "[yellow]💡 Tip:[/yellow] Devices with 'monitor' in the name can capture system audio.\n"
        )

    # Get user selection
    while True:
        try:
            device_indices = [d[0] for d in devices]
            selected = IntPrompt.ask(
                "[bold cyan]Select device by index[/bold cyan]",
                choices=[str(i) for i in device_indices],
                show_choices=False,
            )

            if selected in device_indices:
                # Find the device name for confirmation
                selected_device = next(d for d in devices if d[0] == selected)
                console.print(
                    f"\n[green]✓[/green] Selected device: [bold]{selected_device[1]}[/bold]\n"
                )
                return selected
            else:
                console.print(
                    "[red]Invalid selection. Please choose from the available indices.[/red]"
                )
        except KeyboardInterrupt:
            console.print("\n[yellow]Device selection cancelled.[/yellow]")
            raise


def get_audio_device(
    device_arg: Optional[int] = None, prefer_tae2146: bool = True
) -> tuple[int, str]:
    """Get audio device based on user preference or interactive selection.

    Args:
        device_arg: Device index from command line argument
        prefer_tae2146: If True and no device specified, try TAE2146 first

    Returns:
        Tuple of (device_index, device_name)
    """
    # If device index is provided via command line, use it
    if device_arg is not None:
        try:
            device_info = sd.query_devices(device_arg)
            if device_info["max_input_channels"] > 0:
                console.print(
                    f"[green]✓[/green] Using device {device_arg}: {device_info['name']}"
                )
                return device_arg, device_info["name"]
            else:
                console.print(
                    f"[red]Error: Device {device_arg} has no input channels.[/red]"
                )
        except Exception as e:
            console.print(f"[red]Error: Invalid device index {device_arg}: {e}[/red]")

    # If prefer_tae2146 is True, try to find TAE2146 first
    if prefer_tae2146 and device_arg is None:
        # Try to get TAE2146 device
        device_index, device_name = get_tae2146_monitor_device()
        if device_index != "pulse":  # If we found a specific device
            # Ask user if they want to use TAE2146 or select another device
            console.print(
                Panel(
                    f"[green]Found TAE2146 device:[/green] {device_name}\n\n"
                    "[yellow]Would you like to use this device or select another?[/yellow]",
                    title="[bold]Device Found[/bold]",
                    border_style="green",
                )
            )

            from rich.prompt import Confirm

            use_tae = Confirm.ask("Use TAE2146 device?", default=True)
            if use_tae:
                return device_index, device_name

    # Show device selection menu
    console.print("[cyan]Please select an audio input device for transcription.[/cyan]")

    devices = get_available_input_devices()
    if not devices:
        raise ValueError("No audio input devices found!")

    selected_index = display_device_selection_menu(devices)
    selected_device = next(d for d in devices if d[0] == selected_index)

    return selected_device[0], selected_device[1]


# Get the TAE2146 monitor device
def get_tae2146_monitor_device() -> tuple[str, str]:
    """Get the TAE2146 monitor device specifically."""
    # First, try to use the exact name directly
    device_index, device_name = try_exact_device_match()
    if device_index is not None:
        return device_index, device_name

    # If not found by exact name, search through devices
    device_index, device_name = search_device_by_name()
    if device_index is not None:
        return device_index, device_name

    # If still not found, return pulse as fallback
    return "pulse", "pulse (TAE2146 not found)"


def try_exact_device_match() -> tuple[Optional[str], Optional[str]]:
    """Try to match device by exact name."""
    try:
        sd.query_devices(TARGET_MONITOR)
        console.print("[green]✓[/green] Found TAE2146 monitor device directly")
        return TARGET_MONITOR, TARGET_MONITOR
    except Exception:
        return None, None


def search_device_by_name() -> tuple[Optional[int], Optional[str]]:
    """Search for TAE2146 device in available devices."""
    try:
        devices = sd.query_devices()
        for i, device in enumerate(devices):
            # Look for TAE2146 monitor in device name
            if "TAE2146" in device["name"] and "monitor" in device["name"].lower():
                console.print(
                    f"[green]✓[/green] Found TAE2146 monitor at index {i}: {device['name']}"
                )
                return i, device["name"]
            # Also check for partial matches
            elif "TAE2146" in device["name"] and device["max_input_channels"] > 0:
                console.print(
                    f"[green]✓[/green] Found TAE2146 input device at index {i}: {device['name']}"
                )
                return i, device["name"]
    except Exception as e:
        console.print(f"[yellow]⚠[/yellow] Error searching for TAE2146: {e}")

    return None, None


def setup_pulseaudio_source():
    """Try to set PulseAudio default source to TAE2146 monitor."""
    try:
        subprocess.run(["pactl", "set-default-source", TARGET_MONITOR], check=True)
        console.print(
            "[green]✓[/green] Set PulseAudio default source to TAE2146 monitor"
        )
    except Exception as e:
        console.print(f"[yellow]Note:[/yellow] Could not set default source: {e}")


def get_device_configuration(device_index) -> tuple[int, int, float]:
    """Get device configuration for audio capture."""
    try:
        device_info = sd.query_devices(device_index)
        channels = min(2, device_info["max_input_channels"])  # Use stereo if available
        device_sample_rate = int(device_info["default_samplerate"])
        resample_ratio = device_sample_rate / SAMPLE_RATE

        if device_sample_rate != SAMPLE_RATE:
            console.print(
                f"[cyan]Device sample rate:[/cyan] {device_sample_rate}Hz → "
                f"will resample to {SAMPLE_RATE}Hz"
            )

        return channels, device_sample_rate, resample_ratio
    except Exception:
        # Default values for common system audio
        return 2, 48000, 48000 / SAMPLE_RATE


def get_formats_to_process(format_arg: str) -> list[str]:
    """Get list of formats to process based on argument."""
    return ["txt", "srt", "vtt"] if format_arg == "all" else [format_arg]


def initialize_file_writers(
    output_path: Optional[str],
    formats: list[str],
    no_timestamps: bool,
    detected_language: Optional[str] = None,
    multilingual: bool = False,
) -> dict[str, StreamingLiveWriter]:
    """Initialize file writers for all requested formats."""
    if not output_path:
        return {}

    output_file = Path(output_path)
    # Remove extension if provided
    if output_file.suffix:
        output_file = output_file.parent / output_file.stem

    writers = {}
    for fmt in formats:
        ext = fmt
        output_file_with_ext = output_file.parent / f"{output_file.name}.{ext}"

        try:
            writers[fmt] = StreamingLiveWriter(
                output_file_with_ext,
                format_type=fmt,
                include_timestamps=not no_timestamps,
                detected_language=detected_language,
                multilingual=multilingual,
            )
            console.print(
                f"[green]✓[/green] Saving {fmt.upper()} to: {output_file_with_ext}"
            )
        except Exception as e:
            console.print(f"[red]Failed to create {fmt} output file: {e}[/red]")
            # Clean up any created writers on failure
            for writer in writers.values():
                writer.close()
            raise

    return writers


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


def display_session_summary(
    start_time: float,
    transcription_count: int,
    total_segments: int,
    languages_detected: dict[str, int],
    total_transcription_time: float,
    args,
    file_writers: Optional[dict[str, StreamingLiveWriter]],
) -> list[str]:
    """Display and return session summary."""
    total_time = time.time() - start_time

    summary_content = []
    summary_content.append(
        f"[cyan]Total duration:[/cyan] {format_duration_hhmmss(total_time)}"
    )
    summary_content.append(f"[cyan]Total transcriptions:[/cyan] {transcription_count}")
    summary_content.append(f"[cyan]Total segments:[/cyan] {total_segments}")

    if transcription_count > 0:
        avg_time = total_transcription_time / transcription_count
        summary_content.append(f"[cyan]Avg transcription time:[/cyan] {avg_time:.2f}s")

        # Calculate average processing rate
        avg_rate = (transcription_count * BUFFER_DURATION) / total_transcription_time
        summary_content.append(
            f"[cyan]Avg processing rate:[/cyan] {avg_rate:.1f}x realtime"
        )

    if languages_detected:
        summary_content.append("\n[bold cyan]Languages detected:[/bold cyan]")
        total_lang_segments = sum(languages_detected.values())
        for lang, count in sorted(
            languages_detected.items(), key=lambda x: x[1], reverse=True
        ):
            percentage = (count / total_lang_segments) * 100
            summary_content.append(f"  • {lang}: {count} segments ({percentage:.1f}%)")

    summary_content.append(f"\n[cyan]Model:[/cyan] {args.model}")
    summary_content.append(
        f"[cyan]Language mode:[/cyan] {LANGUAGE_DISPLAY[args.language]}"
    )
    if args.multilingual:
        summary_content.append("[cyan]Multilingual:[/cyan] Enabled")

    if file_writers:
        summary_content.append("[cyan]Output files:[/cyan]")
        for fmt, writer in file_writers.items():
            summary_content.append(f"  • {fmt.upper()}: {writer.output_file}")

    console.print("\n")
    console.print(
        Panel(
            "\n".join(summary_content),
            title="[bold green]📊 Session Summary[/bold green]",
            border_style="green",
            padding=(1, 2),
        )
    )
    console.print("\n")

    return summary_content


def show_ready_panel(
    language_display: str, file_writers: dict[str, StreamingLiveWriter]
):
    """Display the ready panel before starting transcription."""
    panel_content = "[bold green]Listening to system audio...[/bold green]\n"
    panel_content += f"[cyan]Language:[/cyan] {language_display}\n"

    if file_writers:
        formats = ", ".join([fmt.upper() for fmt in file_writers.keys()])
        panel_content += f"[cyan]Output formats:[/cyan] {formats}\n"
        if len(file_writers) == 1:
            writer = list(file_writers.values())[0]
            panel_content += f"[cyan]Output file:[/cyan] {writer.output_file}\n"
    else:
        panel_content += "[cyan]Output:[/cyan] Display only (no file)\n"

    panel_content += "[dim]Press Ctrl+C to stop[/dim]"

    console.print(
        Panel(
            panel_content,
            title="[bold]Ready[/bold]",
            border_style="green",
        )
    )


def process_transcription_segments(
    segments_list: list,
    args,
    ui: TranscriptionUI,
    file_writers: dict[str, StreamingLiveWriter],
    languages_detected: dict[str, int],
    timestamp: float,
):
    """Process transcription segments based on mode."""
    if args.multilingual:
        # Multilingual mode - show each segment with its language
        for segment in segments_list:
            if hasattr(segment, "language"):
                lang = segment.language
                languages_detected[lang] = languages_detected.get(lang, 0) + 1
                ui.update_stats(language=lang)
                ui.update_transcription(segment.text.strip(), lang)
                # Write to all file formats
                for writer in file_writers.values():
                    writer.write_segment(segment.text.strip(), lang, timestamp)
            else:
                ui.update_transcription(segment.text.strip())
                # Write to all file formats
                for writer in file_writers.values():
                    writer.write_segment(segment.text.strip(), None, timestamp)
    else:
        # Regular mode
        text = " ".join([segment.text.strip() for segment in segments_list])

        # Get detected language if auto-detecting
        lang_info = None
        if (
            args.language == "auto"
            and segments_list
            and hasattr(segments_list[0], "language")
        ):
            lang = segments_list[0].language
            languages_detected[lang] = languages_detected.get(lang, 0) + 1
            ui.update_stats(language=lang)
            lang_info = lang

        if text.strip():
            ui.update_transcription(text, lang_info)
            # Write to all file formats
            for writer in file_writers.values():
                writer.write_segment(text, lang_info, timestamp)


class AudioTranscriber:
    """Handles the audio transcription in a separate thread."""

    def __init__(
        self,
        model: WhisperModel,
        ui: TranscriptionUI,
        file_writers: dict[str, StreamingLiveWriter],
        selected_language: Optional[str],
        args,
        checkpoint: Optional[LiveSessionCheckpoint] = None,
    ):
        self.model = model
        self.ui = ui
        self.file_writers = file_writers
        self.selected_language = selected_language
        self.args = args
        self.checkpoint = checkpoint
        self.audio_buffer = []
        self.buffer_lock = threading.Lock()
        self.transcription_queue = queue.Queue()
        self.running = True
        self.transcribing = False

        # Statistics
        self.start_time = time.time()
        self.total_segments = 0
        self.total_transcription_time = 0
        self.languages_detected = {}
        self.transcription_count = 0

    def add_audio(self, audio_data: np.ndarray):
        """Add audio data to buffer and queue it for transcription when full."""
        with self.buffer_lock:
            self.audio_buffer.extend(audio_data)
            if len(self.audio_buffer) >= BUFFER_SIZE:
                chunk = np.array(self.audio_buffer[:BUFFER_SIZE], dtype=np.float32)
                self.transcription_queue.put(chunk)
                self.audio_buffer = self.audio_buffer[BUFFER_SIZE:]

                if self.ui.debug_tracker:
                    self.ui.debug_tracker.add_buffer_size(len(self.audio_buffer))

    def stop(self):
        """Stop the transcription thread."""
        self.running = False

    def transcribe_audio(self):
        """Main transcription loop."""
        checkpoint_counter = 0

        while self.running:
            try:
                # Wait for a chunk of audio from the queue.
                # Timeout to check self.running periodically.
                audio_data = self.transcription_queue.get(timeout=1)
            except queue.Empty:
                self.ui.update_status("👂 Listening for audio...", "green")
                continue

            # Skip if audio is too quiet
            if np.max(np.abs(audio_data)) < SILENCE_THRESHOLD:
                self.ui.update_status("🔇 Silence detected, skipping...", "dim")
                if self.ui.debug_tracker:
                    self.ui.debug_tracker.add_silence_period()
                continue

            if self.ui.debug_tracker:
                self.ui.debug_tracker.add_active_period()

            self.transcribing = True
            self.ui.update_status("🎯 Transcribing...", "bright_yellow")

            trans_start = time.time()
            timestamp = trans_start - self.start_time

            try:
                # Transcribe the audio
                segments, info = self.model.transcribe(
                    audio_data,
                    language=self.selected_language,
                    beam_size=5,
                    vad_filter=True,
                    vad_parameters=dict(min_silence_duration_ms=500),
                    multilingual=self.args.multilingual,
                )

                # Process segments
                segments_list = list(segments)
                segment_count = len(segments_list)

                # Update statistics
                trans_time = time.time() - trans_start
                self.total_transcription_time += trans_time
                self.total_segments += segment_count
                self.transcription_count += 1

                # Update debug tracker
                if self.ui.debug_tracker:
                    self.ui.debug_tracker.add_transcription_time(trans_time)
                    # Transcription rate: audio duration / real time
                    self.ui.debug_tracker.add_transcription_rate(
                        BUFFER_DURATION, trans_time
                    )

                # Update UI stats
                elapsed = time.time() - self.start_time
                self.ui.update_stats(
                    duration=elapsed,
                    transcriptions=self.transcription_count,
                    segments=self.total_segments,
                )

                # Process segments
                process_transcription_segments(
                    segments_list,
                    self.args,
                    self.ui,
                    self.file_writers,
                    self.languages_detected,
                    timestamp,
                )

                # Update checkpoint periodically (every 10 transcriptions)
                checkpoint_counter += 1
                if self.checkpoint and checkpoint_counter % 10 == 0:
                    last_text = " ".join([s.text.strip() for s in segments_list])
                    self.checkpoint.update(
                        elapsed,
                        self.transcription_count,
                        self.total_segments,
                        self.languages_detected,
                        last_text,
                    )
                    self.checkpoint.save()

                self.ui.update_status("✅ Ready", "green")

            except Exception as e:
                self.ui.update_status(f"❌ Error: {str(e)}", "red")
                console.print(f"[red]Transcription error: {e}[/red]")

            self.transcribing = False


def run_live_transcription(args):
    """Run live transcription with the given arguments."""
    # Install global interrupt handler immediately
    global_handler = GlobalInterruptHandler.instance()
    global_handler.install()

    try:
        selected_language = LANGUAGE_MAP[args.language]
        language_display = LANGUAGE_DISPLAY[args.language]

        # Get audio device (interactive or from command line)
        try:
            device_index, device_name = get_audio_device(
                device_arg=args.device,
                prefer_tae2146=not args.no_tae2146,  # Add option to disable TAE2146 preference
            )
        except (KeyboardInterrupt, ValueError) as e:
            if isinstance(e, KeyboardInterrupt):
                console.print("\n[yellow]Transcription cancelled.[/yellow]")
            else:
                show_error_panel(str(e), "Device Selection Error")
            return

        # Get formats to process
        formats_to_process = get_formats_to_process(args.format)

        # Initialize file writers for all formats
        try:
            file_writers = initialize_file_writers(
                args.output,
                formats_to_process,
                args.no_timestamps,
                multilingual=args.multilingual,
            )
        except Exception:
            return

        # Initialize checkpoint
        output_dir = Path(args.output).parent if args.output else None
        checkpoint = LiveSessionCheckpoint(output_dir) if args.output else None

        # Initialize UI with debug mode
        ui = TranscriptionUI(debug_enabled=args.debug)

        # Load model
        model = load_whisper_model(args.model, getattr(args, "cpu", False))

        # If using pulse with TAE2146, try to set the default source
        if device_index == "pulse" and "TAE2146" in device_name:
            setup_pulseaudio_source()

        # Display configuration
        display_configuration(
            language_display,
            args.format,
            args.model,
            file_writers,
            args.multilingual,
            args.debug,
            args.no_timestamps,
            device_name,
        )

        # Get device configuration
        channels, device_sample_rate, resample_ratio = get_device_configuration(
            device_index
        )

        # Initialize transcriber with checkpoint
        transcriber = AudioTranscriber(
            model, ui, file_writers, selected_language, args, checkpoint
        )

        # Setup cleanup handler for this session
        def cleanup_handler(status):
            transcriber.stop()

            # Save final checkpoint
            if checkpoint:
                try:
                    status.update("[cyan]Saving session checkpoint...[/cyan]")
                    checkpoint.update(
                        time.time() - transcriber.start_time,
                        transcriber.transcription_count,
                        transcriber.total_segments,
                        transcriber.languages_detected,
                    )
                    checkpoint.save()
                    time.sleep(0.1)
                except Exception as e:
                    console.print(
                        f"  [yellow]⚠ Warning: Error saving checkpoint: {e}[/yellow]"
                    )

            if file_writers:
                try:
                    # Display summary first
                    summary_content = display_session_summary(
                        transcriber.start_time,
                        transcriber.transcription_count,
                        transcriber.total_segments,
                        transcriber.languages_detected,
                        transcriber.total_transcription_time,
                        args,
                        file_writers,
                    )

                    # Write summary to TXT files
                    status.update("[cyan]Writing session summary to files...[/cyan]")
                    for fmt, writer in file_writers.items():
                        if fmt == "txt":
                            writer.write_summary(summary_content)
                    time.sleep(0.2)

                    # Close all writers
                    status.update("[cyan]Saving transcription files...[/cyan]")
                    for fmt, writer in file_writers.items():
                        writer.close()
                        console.print(
                            f"  [green]✓ Saved {fmt.upper()} to: {writer.output_file}[/green]"
                        )
                    time.sleep(0.1)
                except Exception as e:
                    console.print(
                        f"  [yellow]⚠ Warning: Error closing files: {e}[/yellow]"
                    )
            else:
                # Just display summary if no file writers
                display_session_summary(
                    transcriber.start_time,
                    transcriber.transcription_count,
                    transcriber.total_segments,
                    transcriber.languages_detected,
                    transcriber.total_transcription_time,
                    args,
                    None,
                )
                time.sleep(0.5)  # Give time to see the summary

            # Clear checkpoint on successful completion
            if checkpoint:
                checkpoint.clear()

        # Add cleanup handler to global handler
        global_handler.add_cleanup_handler(cleanup_handler)

        # Audio callback
        def audio_callback(indata, frames, time_info, status):
            """Callback function to capture audio data."""
            if status:
                console.print(f"[yellow]Audio callback status:[/yellow] {status}")

            # Convert to mono if needed
            if indata.shape[1] > 1:
                audio_data = np.mean(indata, axis=1)
            else:
                audio_data = indata.flatten()

            # Resample to 16kHz if needed
            if resample_ratio and resample_ratio != 1.0:
                audio_data = scipy_signal.resample(
                    audio_data, int(len(audio_data) / resample_ratio)
                )

            transcriber.add_audio(audio_data)

        # Start transcription thread
        console.print("\n[cyan]Starting transcription thread...[/cyan]")
        transcription_thread = threading.Thread(
            target=transcriber.transcribe_audio, daemon=True
        )
        transcription_thread.start()

        # Clear console and show ready panel
        console.clear()
        show_ready_panel(language_display, file_writers)
        time.sleep(2)
        console.clear()

        # Start audio capture
        try:
            with sd.InputStream(
                device=device_index,
                channels=channels,
                samplerate=device_sample_rate,
                callback=audio_callback,
                blocksize=1024,
            ):
                # Run UI
                with Live(ui.get_layout(), refresh_per_second=4, console=console):
                    running = True
                    while running:
                        time.sleep(0.25)
                        if not transcriber.transcribing:
                            ui.update_status("👂 Listening for audio...", "green")

                        # Update elapsed time
                        elapsed = time.time() - transcriber.start_time
                        ui.update_stats(duration=elapsed)

                        # Check if interrupted
                        if global_handler.interrupted:
                            running = False
                            break

        except KeyboardInterrupt:
            # This should never happen now with GlobalInterruptHandler
            pass

        except Exception as e:
            console.print(f"\n[red]Error:[/red] {e}")
            show_troubleshooting_tips()

    except Exception as e:
        show_error_panel(str(e), "Live Transcription Error")
    finally:
        # Uninstall global handler
        global_handler.uninstall()


def show_troubleshooting_tips():
    """Display troubleshooting tips for audio issues."""
    console.print("\n[yellow]Troubleshooting tips:[/yellow]")
    console.print("1. Make sure your selected device is properly connected")
    console.print("2. Check that audio is playing through the selected device")
    console.print("3. For system audio capture, look for 'monitor' devices")
    console.print(
        "4. Try running `transcript live` without --device to see all available devices"
    )
    console.print("5. Ensure PulseAudio/PipeWire is running for system audio capture")


# For backward compatibility when running directly
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Transcribe system audio from selected input device"
    )
    parser.add_argument(
        "--language",
        "-l",
        choices=["en", "pt", "auto"],
        default="en",
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
        "--device",
        "-d",
        type=int,
        help="Audio input device index (skip interactive selection)",
    )
    parser.add_argument(
        "--no-tae2146",
        action="store_true",
        help="Don't prefer TAE2146 device (go straight to device selection)",
    )
    parser.add_argument(
        "--multilingual",
        action="store_true",
        help="Enable language detection for each segment (useful for audio with multiple languages)",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output file base name (extension will be added based on format)",
    )
    parser.add_argument(
        "--format",
        "-f",
        choices=["txt", "srt", "vtt", "all"],
        default="txt",
        help="Output format: txt (plain text), srt (subtitles), vtt (WebVTT), or all",
    )
    parser.add_argument(
        "--no-timestamps",
        action="store_true",
        help="Do not include timestamps in transcription (only for TXT format)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode for detailed transcription statistics",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU usage even if GPU is available",
    )
    args = parser.parse_args()
    run_live_transcription(args)
