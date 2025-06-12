import argparse
import queue
import subprocess
import threading
import time

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
from scipy import signal


# Get the TAE2146 monitor device
def get_tae2146_monitor_device():
    """Get the TAE2146 monitor device specifically."""
    target_monitor = (
        "alsa_output.usb-Generic_TAE2146_20210726905926-00.analog-stereo.monitor"
    )

    # First, check if we can use the exact name directly
    try:
        # Try to use the exact device name
        test = sd.query_devices(target_monitor)
        print(f"Found TAE2146 monitor device directly")
        return target_monitor, target_monitor
    except:
        pass

    # If not found by exact name, search through devices
    try:
        devices = sd.query_devices()
        for i, device in enumerate(devices):
            # Look for TAE2146 monitor in device name
            if "TAE2146" in device["name"] and "monitor" in device["name"].lower():
                print(f"Found TAE2146 monitor at index {i}: {device['name']}")
                return i, device["name"]
            # Also check for partial matches
            elif "TAE2146" in device["name"] and device["max_input_channels"] > 0:
                print(f"Found TAE2146 input device at index {i}: {device['name']}")
                return i, device["name"]
    except Exception as e:
        print(f"Error searching for TAE2146: {e}")

    # If still not found, try using pactl to set up proper routing
    print(
        "TAE2146 monitor not found in sounddevice, attempting to use PulseAudio directly..."
    )

    # Use the pulse device with specific source selection
    return "pulse", "pulse (will use TAE2146 monitor via PulseAudio)"


def run_live_transcription(args):
    """Run live transcription with the given arguments."""
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

    # Load the Whisper model
    print(f"Loading Whisper model ({args.model})...")
    model = WhisperModel(args.model, device="cpu", compute_type="int8")
    print(f"Language mode: {language_display}")
    if args.multilingual:
        print("Multilingual mode: ENABLED (will detect language changes)")

    # Audio parameters
    SAMPLE_RATE = 16000
    CHANNELS = 1
    BUFFER_DURATION = 5  # seconds of audio to buffer before transcribing
    BUFFER_SIZE = int(SAMPLE_RATE * BUFFER_DURATION)

    # Get TAE2146 monitor device
    device_index, device_name = get_tae2146_monitor_device()
    print(f"Using audio device: {device_name} (index: {device_index})")

    # If using pulse, try to set the default source to TAE2146 monitor
    if device_index == "pulse":
        try:
            # Set PulseAudio to record from TAE2146 monitor
            subprocess.run(
                [
                    "pactl",
                    "set-default-source",
                    "alsa_output.usb-Generic_TAE2146_20210726905926-00.analog-stereo.monitor",
                ],
                check=True,
            )
            print("Set PulseAudio default source to TAE2146 monitor")
        except Exception as e:
            print(f"Note: Could not set default source: {e}")

    # Audio buffer
    audio_buffer = []
    buffer_lock = threading.Lock()
    transcribing = False
    resample_ratio = None

    # Statistics tracking
    start_time = time.time()
    total_segments = 0
    total_transcription_time = 0
    languages_detected = {}
    transcription_count = 0

    def audio_callback(indata, frames, time, status):
        """Callback function to capture audio data."""
        nonlocal resample_ratio

        if status:
            print(f"Audio callback status: {status}")

        # Convert to mono if needed
        if indata.shape[1] > 1:
            audio_data = np.mean(indata, axis=1)
        else:
            audio_data = indata.flatten()

        # Resample to 16kHz if needed
        if resample_ratio and resample_ratio != 1.0:
            audio_data = signal.resample(
                audio_data, int(len(audio_data) / resample_ratio)
            )

        with buffer_lock:
            audio_buffer.extend(audio_data)

    def transcribe_audio():
        """Thread to transcribe audio periodically."""
        nonlocal \
            transcribing, \
            total_segments, \
            total_transcription_time, \
            transcription_count

        while True:
            time.sleep(BUFFER_DURATION)

            with buffer_lock:
                if len(audio_buffer) < BUFFER_SIZE:
                    continue

                # Get audio data and clear buffer
                audio_data = np.array(audio_buffer[:BUFFER_SIZE], dtype=np.float32)
                audio_buffer[:BUFFER_SIZE] = []

            # Skip if audio is too quiet (no sound playing)
            if np.max(np.abs(audio_data)) < 0.001:
                continue

            transcribing = True
            print("\n[Transcribing...]", end="", flush=True)

            trans_start = time.time()

            try:
                # Transcribe the audio
                segments, info = model.transcribe(
                    audio_data,
                    language=selected_language,  # None for auto-detect
                    beam_size=5,
                    vad_filter=True,  # Voice activity detection
                    vad_parameters=dict(min_silence_duration_ms=500),
                    multilingual=args.multilingual,
                )

                # Process segments
                segments_list = list(segments)
                segment_count = len(segments_list)

                # Update statistics
                trans_time = time.time() - trans_start
                total_transcription_time += trans_time
                total_segments += segment_count
                transcription_count += 1

                if args.multilingual:
                    # Multilingual mode - show each segment with its language
                    texts = []
                    for segment in segments_list:
                        if hasattr(segment, "language"):
                            lang = segment.language
                            languages_detected[lang] = (
                                languages_detected.get(lang, 0) + 1
                            )
                            texts.append(f"[{lang}] {segment.text.strip()}")
                        else:
                            texts.append(segment.text.strip())

                    if texts:
                        print(f"\r[Transcription]: {' '.join(texts)}")
                    else:
                        print("\r[No speech detected]", end="", flush=True)
                else:
                    # Regular mode
                    text = " ".join([segment.text.strip() for segment in segments_list])

                    # Get detected language if auto-detecting
                    if selected_language is None and hasattr(info, "language"):
                        lang = info.language
                        languages_detected[lang] = languages_detected.get(lang, 0) + 1
                        lang_info = f" [{lang}]"
                    else:
                        lang_info = ""

                    if text.strip():
                        print(f"\r[Transcription{lang_info}]: {text}")
                    else:
                        print("\r[No speech detected]", end="", flush=True)

            except Exception as e:
                print(f"\r[Error transcribing]: {e}")

            transcribing = False

    # Start the transcription thread
    print("Starting transcription thread...")
    transcription_thread = threading.Thread(target=transcribe_audio, daemon=True)
    transcription_thread.start()

    # Get device info for proper configuration
    try:
        device_info = sd.query_devices(device_index)
        channels = min(
            2, device_info["max_input_channels"]
        )  # Use stereo if available, else mono

        # Some monitor devices report high sample rates, but we'll resample to 16kHz
        device_sample_rate = int(device_info["default_samplerate"])
        resample_ratio = device_sample_rate / SAMPLE_RATE
        if device_sample_rate != SAMPLE_RATE:
            print(
                f"Device sample rate is {device_sample_rate}Hz, will resample to {SAMPLE_RATE}Hz"
            )
    except:
        channels = 2  # Default to stereo
        device_sample_rate = 48000  # Common sample rate for system audio
        resample_ratio = device_sample_rate / SAMPLE_RATE

    # Start capturing audio from the system
    print(f"\nListening to TAE2146 system audio...")
    print(f"Transcription language: {language_display}")
    print("Make sure audio is playing through your TAE2146 device!")
    print("Press Ctrl+C to stop.\n")

    try:
        with sd.InputStream(
            device=device_index,
            channels=channels,
            samplerate=device_sample_rate,
            callback=audio_callback,
            blocksize=1024,
        ):
            while True:
                time.sleep(0.1)
                if not transcribing:
                    print("\r[Listening for audio...]", end="", flush=True)
    except KeyboardInterrupt:
        print("\n\nStopping...")

        # Display summary statistics
        total_time = time.time() - start_time

        print("\n" + "=" * 50)
        print("TRANSCRIPTION SESSION SUMMARY")
        print("=" * 50)
        print(
            f"Total session duration: {total_time:.1f} seconds ({total_time / 60:.1f} minutes)"
        )
        print(f"Total transcriptions: {transcription_count}")
        print(f"Total segments processed: {total_segments}")

        if transcription_count > 0:
            print(
                f"Average transcription time: {total_transcription_time / transcription_count:.2f} seconds"
            )

        if languages_detected:
            print("\nLanguages detected:")
            for lang, count in sorted(
                languages_detected.items(), key=lambda x: x[1], reverse=True
            ):
                percentage = (count / sum(languages_detected.values())) * 100
                print(f"  {lang}: {count} segments ({percentage:.1f}%)")

        print("\nSettings used:")
        print(f"  Model: {args.model}")
        print(f"  Language mode: {language_display}")
        if args.multilingual:
            print(f"  Multilingual: Enabled")
        print("=" * 50)

    except Exception as e:
        print(f"\nError: {e}")
        print("\nTroubleshooting tips:")
        print("1. Make sure TAE2146 is set as your default audio output device")
        print("2. Check that audio is playing through the TAE2146 device")
        print(
            "3. Try running: pactl set-default-source alsa_output.usb-Generic_TAE2146_20210726905926-00.analog-stereo.monitor"
        )
        print("4. Ensure PulseAudio/PipeWire is running")


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
    args = parser.parse_args()
    run_live_transcription(args)
