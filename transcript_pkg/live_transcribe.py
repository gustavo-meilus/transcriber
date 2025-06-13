import argparse
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, TextIO, Tuple

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
from rich import box
from rich.align import Align
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from scipy import signal as scipy_signal

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


class StreamingLiveWriter:
    """Handles streaming output of live transcription data to file."""

    def __init__(self, output_file: Path, include_timestamps: bool = True):
        self.output_file = output_file
        self.include_timestamps = include_timestamps
        self.file_handle: Optional[TextIO] = None
        self.start_time = time.time()
        self._initialize_file()

    def _initialize_file(self):
        """Initialize the output file."""
        try:
            # Ensure parent directory exists
            self.output_file.parent.mkdir(parents=True, exist_ok=True)

            self.file_handle = open(
                self.output_file, "w", encoding="utf-8", buffering=1
            )

            # Write header
            header = f"Live Transcription Session - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            header += "=" * 60 + "\n\n"
            self.file_handle.write(header)
            self.file_handle.flush()

        except Exception as e:
            console.print(
                f"[red]Error creating output file {self.output_file}: {e}[/red]"
            )
            raise

    def write_transcription(self, text: str, language: Optional[str] = None):
        """Write a transcription to the file."""
        if not self.file_handle:
            return

        try:
            if self.include_timestamps:
                elapsed = time.time() - self.start_time
                timestamp = f"[{elapsed:06.1f}s]"

                if language:
                    line = f"{timestamp} [{language}] {text}\n"
                else:
                    line = f"{timestamp} {text}\n"
            else:
                if language:
                    line = f"[{language}] {text}\n"
                else:
                    line = f"{text}\n"

            self.file_handle.write(line)
            self.file_handle.flush()

        except Exception as e:
            console.print(f"[yellow]Warning: Error writing to file: {e}[/yellow]")

    def write_summary(self, summary_lines: list):
        """Write session summary to the file."""
        if not self.file_handle:
            return

        try:
            self.file_handle.write("\n" + "=" * 60 + "\n")
            self.file_handle.write("Session Summary\n")
            self.file_handle.write("=" * 60 + "\n\n")

            for line in summary_lines:
                # Remove rich formatting
                clean_line = line.replace("[cyan]", "").replace("[/cyan]", "")
                clean_line = clean_line.replace("[bold cyan]", "").replace(
                    "[/bold cyan]", ""
                )
                clean_line = clean_line.replace("[bold]", "").replace("[/bold]", "")
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


class TranscriptionUI:
    """Manages the live transcription UI."""

    def __init__(self):
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
        self.setup_layout()

    def setup_layout(self):
        """Setup the layout structure."""
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

        # Keep last 5 transcriptions
        lines = self.transcription_text.plain.split("\n")
        if len(lines) > 5:
            lines = lines[-5:]
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

        duration_min = self.stats["duration"] / 60
        stats_table.add_row(
            "📊 Duration", f"{self.stats['duration']:.1f}s ({duration_min:.1f} min)"
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

    def get_layout(self):
        """Get the current layout."""
        self.update_header()
        return self.layout


# Get the TAE2146 monitor device
def get_tae2146_monitor_device() -> Tuple[str, str]:
    """Get the TAE2146 monitor device specifically."""
    # First, try to use the exact name directly
    device_index, device_name = try_exact_device_match()
    if device_index is not None:
        return device_index, device_name

    # If not found by exact name, search through devices
    device_index, device_name = search_device_by_name()
    if device_index is not None:
        return device_index, device_name

    # If still not found, use PulseAudio
    console.print(
        "[yellow]⚠[/yellow] TAE2146 monitor not found in sounddevice, "
        "attempting to use PulseAudio directly..."
    )
    return "pulse", "pulse (will use TAE2146 monitor via PulseAudio)"


def try_exact_device_match() -> Tuple[Optional[str], Optional[str]]:
    """Try to match device by exact name."""
    try:
        sd.query_devices(TARGET_MONITOR)
        console.print("[green]✓[/green] Found TAE2146 monitor device directly")
        return TARGET_MONITOR, TARGET_MONITOR
    except Exception:
        return None, None


def search_device_by_name() -> Tuple[Optional[int], Optional[str]]:
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


def get_device_configuration(device_index) -> Tuple[int, int, float]:
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


def initialize_file_writer(
    output_path: Optional[str], no_timestamps: bool
) -> Optional[StreamingLiveWriter]:
    """Initialize file writer if output is specified."""
    if not output_path:
        return None

    output_file = Path(output_path)
    try:
        file_writer = StreamingLiveWriter(
            output_file, include_timestamps=not no_timestamps
        )
        console.print(f"[green]✓[/green] Saving transcriptions to: {output_file}")
        return file_writer
    except Exception as e:
        console.print(f"[red]Failed to create output file: {e}[/red]")
        raise


def load_whisper_model(model_size: str) -> WhisperModel:
    """Load the Whisper model with progress indicator."""
    with console.status(
        "[bold cyan]Loading Whisper model...[/bold cyan]", spinner="dots12"
    ):
        model = WhisperModel(model_size, device="cpu", compute_type="int8")
        console.print("[green]✓[/green] Model loaded successfully!")
    return model


def display_session_summary(
    start_time: float,
    transcription_count: int,
    total_segments: int,
    languages_detected: Dict[str, int],
    total_transcription_time: float,
    args,
    file_writer: Optional[StreamingLiveWriter],
) -> List[str]:
    """Display and return session summary."""
    total_time = time.time() - start_time

    summary_content = []
    summary_content.append(
        f"[cyan]Total duration:[/cyan] {total_time:.1f}s ({total_time / 60:.1f} min)"
    )
    summary_content.append(f"[cyan]Total transcriptions:[/cyan] {transcription_count}")
    summary_content.append(f"[cyan]Total segments:[/cyan] {total_segments}")

    if transcription_count > 0:
        avg_time = total_transcription_time / transcription_count
        summary_content.append(f"[cyan]Avg transcription time:[/cyan] {avg_time:.2f}s")

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
    if file_writer:
        summary_content.append(f"[cyan]Output file:[/cyan] {file_writer.output_file}")

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


def show_ready_panel(language_display: str, file_writer: Optional[StreamingLiveWriter]):
    """Display the ready panel before starting transcription."""
    panel_content = "[bold green]Listening to TAE2146 system audio...[/bold green]\n"
    panel_content += f"[cyan]Language:[/cyan] {language_display}\n"
    if file_writer:
        panel_content += f"[cyan]Output file:[/cyan] {file_writer.output_file}\n"
    panel_content += (
        "[yellow]Make sure audio is playing through your TAE2146 device![/yellow]\n"
    )
    panel_content += "[dim]Press Ctrl+C to stop[/dim]"

    console.print(
        Panel(
            panel_content,
            title="[bold]Ready[/bold]",
            border_style="green",
        )
    )


def process_audio_buffer(
    audio_buffer: List[float], buffer_lock: threading.Lock, buffer_size: int
) -> Optional[np.ndarray]:
    """Process audio buffer and return audio data if ready."""
    with buffer_lock:
        if len(audio_buffer) < buffer_size:
            return None

        # Get audio data and clear buffer
        audio_data = np.array(audio_buffer[:buffer_size], dtype=np.float32)
        audio_buffer[:buffer_size] = []

    # Skip if audio is too quiet
    if np.max(np.abs(audio_data)) < SILENCE_THRESHOLD:
        return None

    return audio_data


def process_transcription_segments(
    segments_list: List,
    args,
    ui: TranscriptionUI,
    file_writer: Optional[StreamingLiveWriter],
    languages_detected: Dict[str, int],
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
                # Write to file if enabled
                if file_writer:
                    file_writer.write_transcription(segment.text.strip(), lang)
            else:
                ui.update_transcription(segment.text.strip())
                # Write to file if enabled
                if file_writer:
                    file_writer.write_transcription(segment.text.strip())
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
            # Write to file if enabled
            if file_writer:
                file_writer.write_transcription(text, lang_info)


class AudioTranscriber:
    """Handles the audio transcription in a separate thread."""

    def __init__(
        self,
        model: WhisperModel,
        ui: TranscriptionUI,
        file_writer: Optional[StreamingLiveWriter],
        selected_language: Optional[str],
        args,
    ):
        self.model = model
        self.ui = ui
        self.file_writer = file_writer
        self.selected_language = selected_language
        self.args = args
        self.audio_buffer = []
        self.buffer_lock = threading.Lock()
        self.running = True
        self.transcribing = False

        # Statistics
        self.start_time = time.time()
        self.total_segments = 0
        self.total_transcription_time = 0
        self.languages_detected = {}
        self.transcription_count = 0

    def add_audio(self, audio_data: np.ndarray):
        """Add audio data to buffer."""
        with self.buffer_lock:
            self.audio_buffer.extend(audio_data)

    def stop(self):
        """Stop the transcription thread."""
        self.running = False

    def transcribe_audio(self):
        """Main transcription loop."""
        while self.running:
            time.sleep(BUFFER_DURATION)

            # Process audio buffer
            audio_data = process_audio_buffer(
                self.audio_buffer, self.buffer_lock, BUFFER_SIZE
            )
            if audio_data is None:
                self.ui.update_status("🔇 Waiting for audio...", "dim")
                continue

            self.transcribing = True
            self.ui.update_status("🎯 Transcribing...", "bright_yellow")

            trans_start = time.time()

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
                    self.file_writer,
                    self.languages_detected,
                )

                self.ui.update_status("✅ Ready", "green")

            except Exception as e:
                self.ui.update_status(f"❌ Error: {str(e)}", "red")

            self.transcribing = False


def run_live_transcription(args):
    """Run live transcription with the given arguments."""
    selected_language = LANGUAGE_MAP[args.language]
    language_display = LANGUAGE_DISPLAY[args.language]

    # Initialize file writer if output is specified
    try:
        file_writer = initialize_file_writer(args.output, args.no_timestamps)
    except Exception:
        return

    # Initialize UI
    ui = TranscriptionUI()

    # Load model
    model = load_whisper_model(args.model)

    console.print(f"[cyan]Language mode:[/cyan] {language_display}")
    if args.multilingual:
        console.print("[cyan]Multilingual mode:[/cyan] [green]ENABLED[/green]")

    # Get TAE2146 monitor device
    device_index, device_name = get_tae2146_monitor_device()
    console.print(f"[cyan]Using audio device:[/cyan] {device_name}")

    # If using pulse, try to set the default source
    if device_index == "pulse":
        setup_pulseaudio_source()

    # Get device configuration
    channels, device_sample_rate, resample_ratio = get_device_configuration(
        device_index
    )

    # Initialize transcriber
    transcriber = AudioTranscriber(model, ui, file_writer, selected_language, args)

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

    # Setup signal handler
    running = True

    def signal_handler(signum, frame):
        nonlocal running
        running = False
        transcriber.stop()
        console.print(
            "\n[bold yellow]🛑 Interrupt received! Gracefully shutting down...[/bold yellow]"
        )

        # Show spinner while saving
        with console.status(
            "[cyan]Saving your session...[/cyan]", spinner="dots12"
        ) as status:
            if file_writer:
                try:
                    # Display summary first
                    summary_content = display_session_summary(
                        transcriber.start_time,
                        transcriber.transcription_count,
                        transcriber.total_segments,
                        transcriber.languages_detected,
                        transcriber.total_transcription_time,
                        args,
                        file_writer,
                    )

                    # Write summary to file
                    status.update("[cyan]Writing session summary to file...[/cyan]")
                    file_writer.write_summary(summary_content)
                    time.sleep(0.2)  # Small delay to make the progress visible

                    # Close the file
                    status.update("[cyan]Saving transcription file...[/cyan]")
                    file_writer.close()
                    time.sleep(0.1)  # Small delay to make the progress visible
                    console.print(
                        f"  [green]✓ Saved transcription to: {file_writer.output_file}[/green]"
                    )
                except Exception as e:
                    console.print(
                        f"  [yellow]⚠ Warning: Error closing file: {e}[/yellow]"
                    )
            else:
                # Just display summary if no file writer
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

        # Final message
        console.print("\n[bold green]✅ Graceful shutdown complete![/bold green]\n")
        sys.exit(0)

    original_sigint = signal.signal(signal.SIGINT, signal_handler)

    # Clear console and show ready panel
    console.clear()
    show_ready_panel(language_display, file_writer)
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
                while running:
                    time.sleep(0.25)
                    if not transcriber.transcribing:
                        ui.update_status("👂 Listening for audio...", "green")

                    # Update elapsed time
                    elapsed = time.time() - transcriber.start_time
                    ui.update_stats(duration=elapsed)

    except KeyboardInterrupt:
        running = False
        transcriber.stop()
        console.print("\n[yellow]Stopping transcription...[/yellow]")

        # Display summary
        summary_content = display_session_summary(
            transcriber.start_time,
            transcriber.transcription_count,
            transcriber.total_segments,
            transcriber.languages_detected,
            transcriber.total_transcription_time,
            args,
            file_writer,
        )

        # Write summary to file
        if file_writer:
            file_writer.write_summary(summary_content)

    except Exception as e:
        console.print(f"\n[red]Error:[/red] {e}")
        show_troubleshooting_tips()

    finally:
        # Restore signal handler
        signal.signal(signal.SIGINT, original_sigint)

        # Ensure file is closed
        if file_writer:
            try:
                file_writer.close()
                console.print(
                    f"[green]✓[/green] Transcription saved to: {file_writer.output_file}"
                )
            except Exception as e:
                console.print(f"[yellow]Warning: Error closing file: {e}[/yellow]")


def show_troubleshooting_tips():
    """Display troubleshooting tips for audio issues."""
    console.print("\n[yellow]Troubleshooting tips:[/yellow]")
    console.print("1. Make sure TAE2146 is set as your default audio output device")
    console.print("2. Check that audio is playing through the TAE2146 device")
    console.print(
        f"3. Try running: [cyan]pactl set-default-source {TARGET_MONITOR}[/cyan]"
    )
    console.print("4. Ensure PulseAudio/PipeWire is running")


# For backward compatibility when running directly
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Transcribe system audio from TAE2146 device"
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
        "--multilingual",
        action="store_true",
        help="Enable language detection for each segment (useful for audio with multiple languages)",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output file for transcription",
    )
    parser.add_argument(
        "--no-timestamps",
        action="store_true",
        help="Do not include timestamps in transcription",
    )
    args = parser.parse_args()
    run_live_transcription(args)
