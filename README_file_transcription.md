# Audio File Transcription

This script transcribes audio files from the `./input` folder and saves transcriptions to the `./output` folder.

## Usage

### Basic Usage

```bash
# Auto-detect language (default)
uv run transcribe_files.py

# Transcribe in English
./transcribe_files_en.sh

# Transcribe in Portuguese
./transcribe_files_pt.sh

# Transcribe multilingual audio (detects language changes within files)
./transcribe_files_multilingual.sh
```

### Advanced Options

```bash
# Use a different model size (tiny, base, small, medium, large)
uv run transcribe_files.py --model small

# Change input/output directories
uv run transcribe_files.py --input /path/to/audio --output /path/to/transcriptions

# Output in different formats
uv run transcribe_files.py --format srt    # Subtitle format
uv run transcribe_files.py --format vtt    # WebVTT format
uv run transcribe_files.py --format all    # All formats (txt, srt, vtt)

# Combine options
uv run transcribe_files.py --language pt --model large --format all

# Enable multilingual mode (detects language changes within audio)
uv run transcribe_files.py --multilingual

# Multilingual with specific output format
uv run transcribe_files.py --multilingual --format srt
```

## Supported Audio Formats

- MP3, WAV, FLAC, OGG, M4A
- MP4, AAC, WMA, OPUS, WEBM
- MKV, AVI, MOV, M4V

## Output Formats

- **txt**: Plain text transcription
- **srt**: SubRip subtitle format with timestamps
- **vtt**: WebVTT subtitle format for web videos
- **all**: Generates all three formats

## How It Works

1. Place audio files in the `./input` directory
2. Run the script with your preferred options
3. Find transcriptions in the `./output` directory with the same filename

## Examples

```bash
# Transcribe all files in Portuguese with subtitles
uv run transcribe_files.py --language pt --format srt

# Use large model for better accuracy with all output formats
uv run transcribe_files.py --model large --format all

# Auto-detect language and save as plain text
uv run transcribe_files.py --language auto
```

## Multilingual Support

The script can handle audio files that contain multiple languages (e.g., a podcast that switches between English and Portuguese):

```bash
# Enable multilingual detection
./transcribe_files_multilingual.sh

# Or manually:
uv run transcribe_files.py --multilingual
```

When multilingual mode is enabled:

- Language is detected for each segment individually
- The output shows language tags like `[en]` or `[pt]` before each segment
- Works with all output formats (txt, srt, vtt)
- Particularly useful for:
  - Bilingual conversations
  - Language learning materials
  - International content with code-switching

## Performance

The script provides detailed performance statistics:

### Per-file statistics:

- Processing speed (e.g., "11.4x realtime" means 11.4x faster than real-time)
- Detected language (when using auto-detect)
- Duration of each audio file
- Individual processing time

### Overall statistics (shown at the end):

- **Total audio duration**: Combined duration of all processed files
- **Total processing time**: Time spent actually transcribing (excludes model loading)
- **Overall time**: Complete execution time including model loading
- **Average processing speed**: Overall speed across all files
- **Time per minute of audio**: How many seconds needed to process 1 minute of audio
- **Estimated time for 1 hour**: Projection for longer content

Example output:

```
==================================================
PERFORMANCE STATISTICS
--------------------------------------------------
Total audio duration: 45.8 seconds (0.8 minutes)
Total processing time: 4.0 seconds (0.1 minutes)
Overall time (including loading): 4.0 seconds (0.1 minutes)

Average processing speed: 11.5x realtime
Time per minute of audio: 5.2 seconds

Estimated time for 1 hour of audio: 5.2 minutes
==================================================
```
