# Transcript - Audio Transcription Tools

A unified CLI tool for audio transcription with support for English and Portuguese, including automatic language detection and multilingual capabilities.

**🚀 New: [Google Colab Notebook](transcript_colab.ipynb)** - Run transcriptions directly in your browser with free GPU!

## 🎯 Features

- **Live Audio Transcription** - Real-time transcription of system audio
- **File Batch Transcription** - Process multiple audio files at once
- **Single File Transcription** - Transcribe individual audio files with smart path handling
- **Multilingual Support** - Automatic detection of language switches
- **Multiple Output Formats** - TXT, SRT, VTT for subtitles
- **Performance Statistics** - Detailed metrics and time estimates
- **Unified CLI** - Single command interface for all features
- **Beautiful UI** - Animated progress bars and colored output using Rich library
- **Streaming Output** - See results as they're generated in real-time
- **Resume Capability** - Continue interrupted transcriptions from checkpoint
- **Graceful Shutdown** - Save progress and outputs on Ctrl+C
- **Smart Output Location** - Defaults to input directory when not specified
- **Flexible Input** - Support for both single files and directories
- **Live Transcription Export** - Save live sessions to file with optional timestamps
- **High-Quality Models** - Support for models from tiny (39MB) to large (1.5GB)
- **GPU Acceleration** - Automatic GPU detection and usage for faster processing
- **Debug Mode**: Detailed statistics for developers to diagnose timing issues

## 🚀 Quick Start

The Transcript tool provides a professional CLI experience with beautiful visual feedback, streaming output, and production-ready reliability.

### Option 1: Google Colab (No Installation!)

Run transcriptions directly in your browser using our [Google Colab notebook](transcript_colab.ipynb):

- No installation required
- Free GPU acceleration
- Direct Google Drive integration
- [📓 Open in Colab](https://colab.research.google.com/github/gustavo-meilus/transcriber/blob/main/transcript_colab.ipynb)

**Getting Started:**

- [🚀 Quick Start Guide](COLAB_QUICK_START.md) - 5 simple steps
- [📘 Complete Setup Guide](COLAB_SETUP_GUIDE.md) - Detailed instructions
- [📖 Colab Features Overview](COLAB_README.md) - What's included

### Option 2: Local Installation

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

Captures and transcribes audio from your selected audio input device in real-time.

### Basic Usage

```bash
# Start with interactive device selection (English + multilingual by default)
transcript live

# Use a specific device by index
transcript live --device 2

# Disable multilingual mode (English only)
transcript live --no-multilingual

# Auto-detect primary language
transcript live --language auto

# Portuguese transcription
transcript live --language pt

# Pure single-language mode (no language tags)
transcript live --language pt --no-multilingual
```

### Device Selection

When you run `transcript live` without the `--device` flag, you'll see an interactive menu to select your audio input device:

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

If you have a TAE2146 device, it will be detected automatically and you'll be asked if you want to use it.

### Advanced Options

```bash
# Use a specific device directly (skip menu)
transcript live --device 2

# Skip TAE2146 auto-detection
transcript live --no-tae2146

# Use a larger model for better accuracy
transcript live --model large

# Save transcription to file
transcript live --output session.txt

# Save without timestamps
transcript live --output session.txt --no-timestamps

# Save in multiple formats
transcript live --output session --format all

# Enable debug mode for performance stats
transcript live --debug

# Combine options
transcript live -d 2 --language auto --multilingual --model small --output meeting
```

### Features

- **Interactive device selection** - Choose from all available audio input devices
- **Direct device selection** - Skip menu with `--device` flag
- **Real-time transcription** of selected audio source
- **Language detection** (English, Portuguese, or automatic)
- **Multilingual mode** - detects and labels language changes within the audio
- **Multiple output formats** - TXT, SRT, VTT, or all formats simultaneously
- **Session summary** - displays comprehensive statistics when stopped (Ctrl+C)
- **Multiple model sizes** - balance speed vs accuracy
- **Live dashboard** - animated status display with real-time stats
- **Debug mode** - detailed performance statistics for troubleshooting
- **Streaming write** - output files updated in real-time as segments are transcribed
- **Graceful shutdown** - transcripts are saved even when interrupted (Ctrl+C)
- **Session checkpoints** - automatic progress saving every 10 transcriptions

### System Audio Capture

To capture system audio (what's playing through your speakers):

- Look for devices with "monitor" in the name
- On Linux: Use "Monitor of" devices (PulseAudio/PipeWire)
- On Windows: May need to enable "Stereo Mix" in sound settings
- On macOS: May need additional software like BlackHole or Loopback

### Session Summary

When you stop the transcription (Ctrl+C), you'll see detailed statistics:

```
╭───────────── 📊 Session Summary ─────────────╮
│ Total duration: 125.3s (2.1 min)             │
│ Total transcriptions: 24                      │
│ Total segments: 156                           │
│ Avg transcription time: 1.85s                 │
│                                               │
│ Languages detected:                           │
│   • en: 89 segments (57.1%)                   │
│   • pt: 67 segments (42.9%)                   │
│                                               │
│ Model: base                                   │
│ Language mode: Auto-detect                    │
│ Multilingual: Enabled                         │
│ Output file: session.txt                      │
╰───────────────────────────────────────────────╯

✓ Transcription saved to: session.txt
```

## 📁 File Transcription

Transcribe individual audio files or batch process entire folders with detailed performance metrics.

### Basic Usage

```bash
# Transcribe all files in default input folder
transcript file

# Transcribe a single audio file
transcript file --input /path/to/audio.mp3

# Transcribe all files in a specific folder
transcript file --input /path/to/audio/folder

# English files
transcript file --language en

# Portuguese files
transcript file --language pt

# Multilingual files
transcript file --multilingual
```

### Advanced Options

```bash
# Single file with specific output location
transcript file --input audio.mp3 --output /path/to/transcriptions

# Change input directory (output defaults to same folder)
transcript file --input /path/to/audio

# Specify different output directory
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

### Debug Mode (Developer Feature)

For developers and troubleshooting timing estimation issues, a debug mode is available:

```bash
# Enable debug statistics during transcription
transcript file --input audio.mp3 --debug

# Debug mode with other options
transcript file --input /path/to/folder --model tiny --debug
```

The debug mode displays detailed statistics every 5 segments:

```
🔧 Debug Statistics - aula_containers.wav
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━┓
┃ Metric                         ┃                Value ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━┩
│ Total Audio Duration           │             8432.50s │
│ Current Position               │             1250.30s │
│ Remaining Duration             │             7182.20s │
│ Progress                       │                14.8% │
│ Mean Processing Rate           │               13.50x │
│ Estimated Time Remaining       │        532.0s (8.9m) │
│                                │                      │
│ ────────────────────────────── │ ──────────────────── │
│                                │                      │
│ Last 10 Segments               │                      │
│   Total Segment Duration       │              115.00s │
│   Total Process Time           │                8.50s │
│   Average Rate                 │               13.53x │
│                                │                      │
│   Segment 1                    │       11.20s @ 14.0x │
│   Segment 2                    │        9.80s @ 14.0x │
│   Segment 3                    │       12.50s @ 13.9x │
│   Segment 4                    │       10.30s @ 13.7x │
│   Segment 5                    │       13.10s @ 13.1x │
│   Segment 6                    │        8.90s @ 13.7x │
│   Segment 7                    │       11.70s @ 13.8x │
│   Segment 8                    │       14.20s @ 12.9x │
│   Segment 9                    │       10.50s @ 13.1x │
│   Segment 10                   │       12.80s @ 13.5x │
└────────────────────────────────┴──────────────────────┘
```

This helps identify:

- When processing rate drops or becomes undefined
- Individual segments that take unusually long
- Inconsistent processing rates
- Progress tracking synchronization issues

### Features

- **Single file support** - transcribe individual audio files
- **Batch processing** - transcribe entire folders
- **Multiple formats** - TXT, SRT, VTT outputs
- **Progress tracking** - real-time progress with speed metrics
- **Performance statistics** - detailed timing and estimates
- **Wide format support** - MP3, WAV, MP4, and many more
- **Streaming output** - results written as segments are processed
- **Resume capability** - continue from where you left off
- **Graceful interruption** - save progress with Ctrl+C
- **Smart output location** - defaults to same folder as input files

### Input Types

The file transcription mode intelligently handles both files and folders:

- **Single File**: When you specify a file path (e.g., `--input audio.mp3`), only that file will be transcribed
- **Folder**: When you specify a folder path (e.g., `--input /audio/folder`), all supported audio files in that folder will be transcribed

### 🔄 Streaming Output & Resume

#### Streaming Output

Files are written segment-by-segment as they're processed:

- View partial results immediately
- No need to wait for complete file processing
- Progress bar shows actual segments completed
- **Transcripts saved on interruption** - Even if you press Ctrl+C, all processed segments are saved

#### Checkpoint System

Automatic progress saving enables resumable transcriptions:

- Progress saved automatically every 10 segments
- Graceful shutdown on Ctrl+C saves current state
- Resume prompt when previous session detected
- **All writers are properly closed** - Ensures data is flushed to disk on interruption

```
╭─────────────── Resume Session? ───────────────╮
│ Found previous transcription session           │
│ Time: 2024-01-15T14:30:22                    │
│ Files completed: 3                             │
│ Current file: interview.mp3                    │
╰────────────────────────────────────────────────╯
Do you want to resume from where you left off? (Y/n):
```

#### File Resilience

The system handles file issues gracefully:

- Continues processing if input files are deleted
- Handles output write failures without stopping
- Retries file operations for temporary failures

### Performance Statistics

```
╔═══════════════════════════════════════════════════════════════╗
║  ⚡ Performance Statistics                                      ║
╠═══════════════════════════════════════════════════════════════╣
║ Metric                        │ Value                          ║
╟───────────────────────────────┼────────────────────────────────╢
║ Total Audio Duration          │ 45.8s (0.8 min)                ║
║ Total Processing Time         │ 4.0s (0.1 min)                 ║
║ Overall Time                  │ 4.0s (0.1 min)                 ║
║ Average Speed                 │ 11.5x realtime                 ║
║ Time per Audio Minute         │ 5.2s                           ║
║ Est. for 1 Hour Audio         │ 5.2 minutes                    ║
╚═══════════════════════════════════════════════════════════════╝
```

### Example Outputs

#### Portuguese Transcription (from classroom recording):

```
Boa noite a todos e todas, eu sou a Luísa César, o mediador da aula de hoje.
Sejam muito bem-vindos ao módulo introdutório com a aula de
Comprimização de Serviços Docker com o professor Hélder Prado Santos.
```

#### SRT Format with Timestamps:

```
33
00:01:08,310 --> 00:01:10,190
em segundo plano, instalando o material de Docker,

34
00:01:10,410 --> 00:01:12,390
o principal, que é a instalação do Docker Desktop.
```

#### Graceful Shutdown Messages:

```
[yellow]Interrupt received! Saving progress...[/yellow]
[green]✓ Saved txt output[/green]
[green]✓ Saved srt output[/green]
[green]✓ Saved vtt output[/green]
[green]Progress saved. You can resume later.[/green]
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
  --device, -d INDEX  Audio input device index (skip interactive selection)
  --no-tae2146  Don't prefer TAE2146 device (go straight to device selection)
  --multilingual  Enable language detection for each segment (default: enabled)
  --no-multilingual  Disable multilingual mode
  --output, -o PATH  Save transcription to file base name
  --format, -f {txt,srt,vtt,all}  Output format (default: txt)
  --no-timestamps  Do not include timestamps in output file (TXT only)
  --debug  Enable debug mode with detailed statistics
  --cpu  Force CPU usage even if GPU is available
```

### File Mode Options

```
transcript file [options]

Options:
  --language, -l {en,pt,auto}  Language for transcription (default: en)
  --model, -m {tiny,base,small,medium,large}  Model size (default: base)
  --input, -i PATH  Input file or folder (default: ./input)
  --output, -o PATH  Output folder (default: same as input)
  --format, -f {txt,srt,vtt,all}  Output format (default: txt)
  --multilingual  Enable language detection for each segment (default: enabled)
  --no-multilingual  Disable multilingual mode
  --debug  Enable debug mode with detailed statistics (for developers)
  --cpu  Force CPU usage even if GPU is available
```

## ⚙️ Configuration

### Model Sizes

- `tiny` - Fastest, least accurate (~39MB)
- `base` - Good balance (default) (~74MB)
- `small` - Better accuracy (~244MB)
- `medium` - High accuracy (~769MB)
- `large` - Best accuracy (~1550MB)

### GPU Support

The tool automatically detects and uses GPU acceleration when available:

- **Automatic Detection** - Detects NVIDIA GPUs with CUDA support
- **Smart Compute Type** - Uses float16 for modern GPUs (RTX 20xx+), int8 for older
- **Fallback to CPU** - Gracefully falls back if GPU unavailable or PyTorch missing
- **Force CPU Option** - Use `--cpu` flag to force CPU usage even with GPU available

#### GPU Requirements

- NVIDIA GPU with CUDA support
- PyTorch installed with CUDA support: `pip install torch torchvision torchaudio`

#### Performance Comparison

- **CPU (int8)**: ~10-15x realtime on modern processors
- **GPU (float16)**: ~30-50x realtime on RTX 3060 or better
- **GPU (int8)**: ~20-30x realtime on older GPUs

Example:

```bash
# Auto-detect GPU (default)
transcript file --model large

# Force CPU usage
transcript file --model large --cpu
```

### Audio Device Selection

The live transcription supports all available audio input devices on your system:

- **Interactive Selection** - Choose from a menu of available devices when starting
- **Direct Selection** - Use `--device INDEX` to select a specific device
- **TAE2146 Support** - Automatically detects and offers TAE2146 devices when available
- **System Audio** - Look for "monitor" devices to capture system audio

## 📊 Tested Performance

The tool has been extensively tested with real-world recordings:

### Test Case: Portuguese Classroom Recording

- **File**: 1.1GB WAV file (long classroom session)
- **Language**: Portuguese
- **Model**: Large (1.5GB)
- **Results**:
  - ✅ Accurate transcription of technical terms (Docker, VS Code)
  - ✅ Proper name recognition (Luísa César, Hélder Prado Santos)
  - ✅ Clean handling of interruptions with data preservation
  - ✅ All output formats (TXT, SRT, VTT) generated correctly
  - ✅ Checkpoint system successfully saved progress
  - ✅ Memory usage: ~2.8GB with large model
  - ✅ Processing continues even after Ctrl+C

### Performance Characteristics

- **Model Loading**: 30-60 seconds for large model
- **Streaming Output**: Transcripts appear within seconds of processing
- **Interrupt Handling**: Clean shutdown with confirmation messages
- **Resume Speed**: Instant detection of previous sessions
- **Output Quality**: Professional-grade transcriptions suitable for subtitles

## 📁 Project Structure

```
transcriber/
├── transcript_pkg/
│   ├── __init__.py
│   ├── cli.py              # Main CLI entry point
│   ├── live_transcribe.py  # Live audio transcription
│   └── file_transcribe.py  # File batch transcription
├── input/                  # Default input folder for files
├── output/                 # Default output folder (when specified)
├── pyproject.toml          # Project configuration
├── README.md               # This file
├── VISUAL_ENHANCEMENTS.md  # Detailed visual features documentation
├── INSTALL.md              # Installation guide
└── CLI_USAGE.md            # Command-line usage examples
```

## 📋 Requirements

- Python 3.9+
- PulseAudio or PipeWire (for system audio capture on Linux)
- Any audio input device (microphone, system audio monitor, etc.)

## 🤝 Contributing

Feel free to submit issues and enhancement requests!

## 🔧 Key Technical Improvements

### Visual Enhancements

- Rich library integration for beautiful terminal UI
- Animated progress bars with dual tracking (files + segments)
- Color-coded output for better readability
- Live dashboard for real-time transcription
- Professional error handling and status messages

### Streaming & Reliability

- True streaming output - segments written immediately
- Checkpoint system saves every 10 segments automatically
- Graceful interrupt handler ensures no data loss
- Smart file closing on any exit condition
- Resilient file operations with retry logic

### User Experience

- Smart defaults (output to input directory when not specified)
- Flexible input handling (single file or directory)
- Clear feedback at every step
- Resume prompts with session information
- Comprehensive statistics and performance metrics

For detailed information about visual enhancements and implementation details, see [VISUAL_ENHANCEMENTS.md](VISUAL_ENHANCEMENTS.md).

## 📄 License

[Your License Here]
