# Transcript CLI Usage Guide

## Quick Start

```bash
# Using uv (recommended)
uv run ./transcript [command] [options]

# Or as Python module
python -m src.cli [command] [options]
```

## Commands

### Live Transcription

Transcribe audio in real-time from your system audio:

### Basic Usage

```bash
transcript live
```

This will:

1. Show a list of available audio input devices
2. Let you select which device to transcribe from
3. Display transcriptions in real-time with language tags (English + multilingual by default)

### Understanding the Defaults

By default, live transcription uses:

- **Primary language**: English
- **Multilingual mode**: Enabled (detects and labels other languages)
- **Output**: Display only (no file)

This means you'll see output like:

```
[en] Hello, welcome to our podcast.
[pt] Olá, bem-vindo ao nosso podcast.
[en] Today we'll discuss technology.
```

### Device Selection

#### Interactive Selection (Default)

When you run `transcript live` without specifying a device, you'll see an interactive menu:

```
🎤 Available Audio Input Devices
┌────────┬──────────────────────────────────────────────────┬──────────┬─────────────┐
│ Index  │ Device Name                                      │ Channels │ Sample Rate │
├────────┼──────────────────────────────────────────────────┼──────────┼─────────────┤
│   0    │ Built-in Microphone                              │    2     │   48,000 Hz │
│   1    │ USB Headset Microphone                           │    1     │   44,100 Hz │
│   2    │ Monitor of Built-in Audio (System Audio)         │    2     │   48,000 Hz │
│   3    │ TAE2146 USB Audio Device (Recommended)           │    2     │   48,000 Hz │
└────────┴──────────────────────────────────────────────────┴──────────┴─────────────┘

💡 Tip: Devices with 'monitor' in the name can capture system audio.

Select device by index:
```

#### Direct Device Selection

If you know your device index, skip the menu:

```bash
# Use device index 2 directly
transcript live --device 2

# Short form
transcript live -d 2
```

#### TAE2146 Device

If you have a TAE2146 audio device, it will be detected automatically and you'll be asked if you want to use it. To skip this prompt:

```bash
# Skip TAE2146 preference and go straight to device selection
transcript live --no-tae2146
```

### Save Transcription to File

```bash
# Save to text file
transcript live -o session.txt

# Save with timestamps disabled (cleaner output)
transcript live -o session.txt --no-timestamps
```

### Multiple Output Formats

```bash
# Save as SRT subtitles
transcript live -o session --format srt

# Save as WebVTT
transcript live -o session --format vtt

# Save in all formats simultaneously
transcript live -o session --format all
```

### Language Options

```bash
# Default: English with multilingual detection
transcript live

# English only (no language tags)
transcript live --no-multilingual

# Portuguese with multilingual detection
transcript live --language pt

# Portuguese only (no language tags)
transcript live --language pt --no-multilingual

# Auto-detect primary language (with multilingual)
transcript live --language auto

# Auto-detect single language (no tags)
transcript live --language auto --no-multilingual
```

### Model Selection

```bash
# Use tiny model (fastest, least accurate)
transcript live --model tiny

# Use large model (slowest, most accurate)
transcript live --model large
```

### Advanced Options

```bash
# Enable debug mode for performance statistics
transcript live --debug

# Force CPU usage even if GPU is available
transcript live --cpu

# Combine multiple options
transcript live -d 2 -o meeting --format all --language auto --model medium --debug
```

### System Audio Capture Tips

- Look for devices with "monitor" in the name for system audio capture
- On Linux with PulseAudio: Use "Monitor of" devices
- On Windows: May need to enable "Stereo Mix" in sound settings
- On macOS: May need additional software like BlackHole or Loopback

### Examples

#### Record a Meeting

```bash
# Capture system audio with auto language detection
transcript live -o meeting --language auto --format all
```

#### Transcribe Microphone Input

```bash
# Select microphone interactively and save with timestamps
transcript live -o interview.txt
```

#### Debug Audio Issues

```bash
# Run without saving to test device selection
transcript live --debug
```

### File Transcription

Batch process audio files:

```bash
# Basic usage (auto-detect language)
uv run ./transcript file

# English files
uv run ./transcript file --language en

# Portuguese files with subtitles
uv run ./transcript file --language pt --format srt

# All output formats
uv run ./transcript file --format all

# Custom directories
uv run ./transcript file --input /path/to/audio --output /path/to/output

# Multilingual with large model
uv run ./transcript file --multilingual --model large
```

## Options Reference

### Global Options

- `--help` - Show help message
- `--version` - Show version

### Live Mode Options

- `--language {en,pt,auto}` - Language (default: en)
- `--model {tiny,base,small,medium,large}` - Model size (default: base)
- `--device, -d INDEX` - Audio input device index (skip interactive selection)
- `--no-tae2146` - Don't prefer TAE2146 device (go straight to device selection)
- `--multilingual` - Enable language detection for each segment (default: enabled)
- `--no-multilingual` - Disable multilingual mode
- `--output, -o PATH` - Save transcription to file base name
- `--format, -f {txt,srt,vtt,all}` - Output format (default: txt)
- `--no-timestamps` - Do not include timestamps in output file (TXT only)
- `--debug` - Enable debug mode with detailed statistics
- `--cpu` - Force CPU usage even if GPU is available

### File Mode Options

- `--language {en,pt,auto}` - Language (default: en)
- `--model {tiny,base,small,medium,large}` - Model size (default: base)
- `--input PATH` - Input file or folder (default: ./input)
- `--output PATH` - Output folder (default: same as input)
- `--format {txt,srt,vtt,all}` - Output format (default: txt)
- `--multilingual` - Enable language detection for each segment (default: enabled)
- `--no-multilingual` - Disable multilingual mode
- `--debug` - Enable debug mode for developers
- `--cpu` - Force CPU usage even if GPU is available

## Examples

```bash
# Live English transcription
uv run ./transcript live

# Live multilingual with auto-detection
uv run ./transcript live --language auto --multilingual

# Transcribe Portuguese podcast files to subtitles
uv run ./transcript file --language pt --format srt

# Process multilingual videos with all formats
uv run ./transcript file --multilingual --format all --model medium

# Custom folders with large model
uv run ./transcript file -i ~/Videos -o ~/Transcriptions --model large
```

## Tips

1. **Model Selection**:

   - `base` is good for most use cases
   - Use `large` for best accuracy
   - Use `tiny` for fastest processing

2. **Language Detection**:

   - Use `--language auto` for unknown content
   - Use `--multilingual` for mixed-language content

3. **Output Formats**:

   - `txt` - Plain text
   - `srt` - Subtitles with timestamps
   - `vtt` - Web video subtitles
   - `all` - Generate all formats

4. **Performance**:
   - Processing speed varies by model size
   - Expect ~10-15x realtime with base model on CPU
   - Large models are ~2-3x realtime on CPU
   - GPU can be 3-5x faster than CPU
   - Use `--cpu` to force CPU if GPU is busy
