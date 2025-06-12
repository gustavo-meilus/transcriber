# Transcript - Audio Transcription Tools

A unified CLI tool for audio transcription with support for English and Portuguese, including automatic language detection and multilingual capabilities.

## 🎯 Features

- **Live Audio Transcription** - Real-time transcription of system audio
- **File Batch Transcription** - Process multiple audio files at once
- **Multilingual Support** - Automatic detection of language switches
- **Multiple Output Formats** - TXT, SRT, VTT for subtitles
- **Performance Statistics** - Detailed metrics and time estimates
- **Unified CLI** - Single command interface for all features

## 🚀 Quick Start

### Installation

#### For Users (Recommended)

Install as a global command-line tool:

```bash
# Clone and install
git clone https://github.com/gustavo-meilus/transcriber
cd transcriber
uv tool install .

# Now use from anywhere
transcript --help
```

#### For Development

```bash
# Clone the repository
git clone https://github.com/gustavo-meilus/transcriber
cd transcriber

# Install dependencies
uv sync

# Run in development mode
uv run ./transcript --help
```

See [INSTALL.md](INSTALL.md) for detailed installation instructions.

### Usage

The `transcript` command provides two main modes: `live` for real-time transcription and `file` for batch processing.

```bash
# If installed as UV tool (recommended):
transcript [command] [options]

# For development:
uv run ./transcript [command] [options]
# or
python -m transcript_pkg.cli [command] [options]
```

## 🎤 Live Audio Transcription

Captures and transcribes audio from your TAE2146 audio device in real-time.

### Basic Usage

```bash
# English transcription (default)
transcript live

# Portuguese transcription
transcript live --language pt

# Auto-detect language
transcript live --language auto

# Multilingual (detects language changes)
transcript live --multilingual
```

### Advanced Options

```bash
# Use a larger model for better accuracy
transcript live --model large

# Combine options
transcript live --language auto --multilingual --model small
```

### Features

- **Real-time transcription** of system audio
- **Language detection** (English, Portuguese, or automatic)
- **Multilingual mode** - detects and labels language changes within the audio
- **Session summary** - displays comprehensive statistics when stopped (Ctrl+C)
- **Multiple model sizes** - balance speed vs accuracy

### Session Summary

When you stop the transcription (Ctrl+C), you'll see detailed statistics:

```
==================================================
TRANSCRIPTION SESSION SUMMARY
==================================================
Total session duration: 125.3 seconds (2.1 minutes)
Total transcriptions: 24
Total segments processed: 156
Average transcription time: 1.85 seconds

Languages detected:
  en: 89 segments (57.1%)
  pt: 67 segments (42.9%)

Settings used:
  Model: base
  Language mode: Auto-detect
  Multilingual: Enabled
==================================================
```

## 📁 File Transcription

Batch transcribe audio files with detailed performance metrics.

### Basic Usage

```bash
# Auto-detect language (default)
transcript file

# English files
transcript file --language en

# Portuguese files
transcript file --language pt

# Multilingual files
transcript file --multilingual
```

### Advanced Options

```bash
# Change input/output directories
transcript file --input /path/to/audio --output /path/to/transcriptions

# Different output formats
transcript file --format srt    # Subtitle format
transcript file --format vtt    # WebVTT format
transcript file --format all    # All formats

# Use larger model
transcript file --model large

# Combine options
transcript file --language pt --model medium --format all --multilingual
```

### Features

- **Batch processing** - transcribe entire folders
- **Multiple formats** - TXT, SRT, VTT outputs
- **Progress tracking** - real-time progress with speed metrics
- **Performance statistics** - detailed timing and estimates
- **Wide format support** - MP3, WAV, MP4, and many more

### Performance Statistics

```
==================================================
PERFORMANCE STATISTICS
--------------------------------------------------
Total audio duration: 45.8 seconds (0.8 minutes)
Total processing time: 4.0 seconds (0.1 minutes)
Overall time (including loading): 4.0 seconds

Average processing speed: 11.5x realtime
Time per minute of audio: 5.2 seconds
Estimated time for 1 hour of audio: 5.2 minutes
==================================================
```

## 🌐 Multilingual Support

Both modes support multilingual transcription - perfect for:

- Bilingual conversations
- Language learning content
- International meetings
- Code-switching scenarios

When enabled with `--multilingual`, transcriptions show language tags:

```
[en] Hello, welcome to our podcast.
[pt] Olá, bem-vindo ao nosso podcast.
[en] Today we'll discuss technology.
[pt] Hoje vamos discutir tecnologia.
```

## 📋 CLI Reference

### Global Options

- `--version` - Show version information
- `--help` - Show help message

### Live Mode Options

```
transcript live [options]

Options:
  --language, -l {en,pt,auto}  Language for transcription (default: en)
  --model, -m {tiny,base,small,medium,large}  Model size (default: base)
  --multilingual  Enable per-segment language detection
```

### File Mode Options

```
transcript file [options]

Options:
  --language, -l {en,pt,auto}  Language for transcription (default: auto)
  --model, -m {tiny,base,small,medium,large}  Model size (default: base)
  --input, -i PATH  Input folder (default: ./input)
  --output, -o PATH  Output folder (default: ./output)
  --format, -f {txt,srt,vtt,all}  Output format (default: txt)
  --multilingual  Enable per-segment language detection
```

## ⚙️ Configuration

### Model Sizes

- `tiny` - Fastest, least accurate (~39MB)
- `base` - Good balance (default) (~74MB)
- `small` - Better accuracy (~244MB)
- `medium` - High accuracy (~769MB)
- `large` - Best accuracy (~1550MB)

### Audio Device

The live transcription is configured for the TAE2146 audio device. To use a different device, modify the device selection in `transcript_pkg/live_transcribe.py`.

## 📁 Project Structure

```
transcriber/
├── transcript_pkg/
│   ├── __init__.py
│   ├── cli.py              # Main CLI entry point
│   ├── live_transcribe.py  # Live audio transcription
│   └── file_transcribe.py  # File batch transcription
├── input/                  # Default input folder for files
├── output/                 # Default output folder
├── pyproject.toml          # Project configuration
├── README.md               # This file
└── README_file_transcription.md  # Detailed file transcription docs
```

## 📋 Requirements

- Python 3.9+
- PulseAudio or PipeWire (for system audio capture)
- TAE2146 audio device (or modify for your device)

## 🤝 Contributing

Feel free to submit issues and enhancement requests!

## 📄 License

[Your License Here]
