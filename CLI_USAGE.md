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

Real-time transcription of system audio:

```bash
# Basic usage (English)
uv run ./transcript live

# Portuguese
uv run ./transcript live --language pt

# Auto-detect language
uv run ./transcript live --language auto

# Multilingual (detects language changes)
uv run ./transcript live --multilingual

# With larger model
uv run ./transcript live --model large --multilingual
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
- `--multilingual` - Enable per-segment language detection
- `--cpu` - Force CPU usage even if GPU is available

### File Mode Options

- `--language {en,pt,auto}` - Language (default: auto)
- `--model {tiny,base,small,medium,large}` - Model size (default: base)
- `--input PATH` - Input folder (default: ./input)
- `--output PATH` - Output folder (default: ./output)
- `--format {txt,srt,vtt,all}` - Output format (default: txt)
- `--multilingual` - Enable per-segment language detection
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
