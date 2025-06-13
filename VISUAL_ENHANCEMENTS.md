# Visual Enhancements for Transcript CLI

This document describes the visual improvements and animations added to the Transcript CLI tool to make it more visually appealing and responsive.

## 🎨 Overview of Enhancements

The CLI has been completely redesigned with the following improvements:

1. **Animated Progress Bars** - Smooth animations with real-time updates
2. **Color-Coded Output** - Different colors for different types of information
3. **Live Status Updates** - Real-time feedback during processing
4. **Styled Tables** - Clean, formatted tables for statistics and summaries
5. **Responsive UI** - Dynamic updates that make the CLI feel more interactive

## 🚀 Key Features

### 1. Enhanced Main Interface

- **Colored Examples**: Help text uses color coding to make examples more readable
- **Clean Layout**: Better spacing and organization of information

### 2. File Transcription Mode

#### Progress Indicators

- **Animated Spinner**: Shows activity during model loading and processing
- **Dual Progress Bars**:
  - Overall progress for all files
  - Individual progress for current file
- **Real-time Stats**: Shows processing speed and time estimates

#### Visual Feedback

```
✅ Model loaded successfully!
✓ test.wav (70.6s @ 2.6x speed)
```

#### Styled Tables

- Configuration table showing settings before processing
- Session summary with emojis for better readability
- Performance statistics with clear metrics

### 3. Live Transcription Mode

#### Live Dashboard

The live mode features a real-time dashboard with four sections:

1. **Header Panel**: Shows "🎙️ LIVE TRANSCRIPTION" with double borders
2. **Main Transcription Area**: Displays the last 5 transcriptions with language tags
3. **Status Bar**: Animated status indicators:
   - 🔇 Waiting for audio...
   - 👂 Listening for audio...
   - 🎯 Transcribing...
   - ✅ Ready
4. **Statistics Panel**: Real-time stats including:
   - 📊 Duration
   - 🎯 Transcriptions count
   - 📝 Segments processed
   - 🌐 Languages detected
   - 🕐 Last update time

#### Status Animations

The status bar changes color based on the current state:

- **Dim gray**: Waiting for audio
- **Green**: Listening
- **Bright yellow**: Actively transcribing
- **Green**: Ready/Complete
- **Red**: Error state

### 4. Color Scheme

The CLI uses a consistent color scheme:

- **Cyan**: Headers, configuration items, and important labels
- **Green**: Success messages and positive feedback
- **Yellow**: Warnings, current activity, and values
- **Red**: Errors and failed operations
- **White**: General text
- **Dim**: Less important information

## 🔄 Advanced Features (New!)

### 5. Streaming Output & Real-time Progress

The file transcription mode now features:

#### Streaming Transcription

- **Real-time Writing**: Segments are written to output files as they're processed
- **Live Progress**: Progress bar shows actual segments completed, not estimates
- **Immediate Results**: Users can start viewing partial results immediately

#### Segment-level Progress

```
Processing: audio.mp3 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 127/156 • 81%
```

### 6. Checkpoint & Resume System

#### Automatic Checkpointing

- **Progress Saving**: Automatically saves progress every 10 segments
- **Session Information**: Stores completed files, current file, and processed segments
- **Settings Preservation**: Remembers the exact settings used

#### Resume Prompt

When a previous session is detected:

```
╭─────────────── Resume Session? ───────────────╮
│ Found previous transcription session           │
│ Time: 2024-01-15T14:30:22                    │
│ Files completed: 3                             │
│ Current file: interview.mp3                    │
╰────────────────────────────────────────────────╯
Do you want to resume from where you left off? (Y/n):
```

### 7. Graceful Shutdown

#### Interrupt Handling

- **Ctrl+C Support**: Cleanly saves progress when interrupted
- **Visual Feedback**: Shows saving status

```
Interrupt received! Saving progress...
Progress saved. You can resume later.
```

### 8. File Resilience

#### Robust File Handling

- **Retry Logic**: Automatically retries file operations if temporary failures occur
- **Missing File Detection**: Gracefully handles deleted input files
- **Output Protection**: Continues with other files if one output fails

#### Error Recovery

```
✗ Input file not found: deleted_file.mp3
✗ Error writing srt output: Permission denied
```

### 9. Smart Output Location

#### Intelligent Default Behavior

- **Same Folder Output**: When no output directory is specified, transcriptions are saved alongside the source audio files
- **Clear Configuration Display**: Shows output location in the configuration table
- **Checkpoint Location**: Checkpoint files are stored in the appropriate location based on output choice

#### Output Location Display

```
         Transcription Configuration
┌──────────────────────┬─────────────────────┐
│ Files to Process     │ 3                   │
│ Language Mode        │ Auto-detect         │
│ Output Format        │ TXT                 │
│ Model Size           │ BASE                │
│ Output Location      │ Same as input files │
└──────────────────────┴─────────────────────┘
```

When output is specified:

```
│ Output Location      │ /path/to/output     │
```

### 10. Flexible Input Handling

#### Single File vs Directory Mode

The file transcription command now intelligently handles both individual files and directories:

- **Single File Mode**: When input is a file path, only that specific file is transcribed
- **Directory Mode**: When input is a directory path, all audio files in the directory are transcribed

#### Input Type Display

The configuration table clearly shows what type of input is being processed:

For single file:

```
         Transcription Configuration
┌──────────────────────┬─────────────────────┐
│ Input Type           │ Single file         │
│ Input Path           │ audio/podcast.mp3   │
│ Files to Process     │ 1                   │
│ Language Mode        │ Auto-detect         │
│ Output Format        │ TXT                 │
│ Model Size           │ BASE                │
│ Output Location      │ Same as input files │
└──────────────────────┴─────────────────────┘
```

For directory:

```
         Transcription Configuration
┌──────────────────────┬─────────────────────┐
│ Input Type           │ Directory           │
│ Input Path           │ input               │
│ Files to Process     │ 5                   │
│ Language Mode        │ Auto-detect         │
│ Output Format        │ TXT                 │
│ Model Size           │ BASE                │
│ Output Location      │ Same as input files │
└──────────────────────┴─────────────────────┘
```

#### Error Handling for Invalid Files

When a non-audio file is specified:

```
╭─────────────── Invalid Audio File ───────────────╮
│ File is not a supported audio format: doc.pdf    │
│ Supported formats: .aac, .avi, .flac, .m4a...    │
╰───────────────────────────────────────────────────╯
```

## 🛠️ Technical Implementation

### Dependencies Added

- **rich**: For terminal styling, progress bars, and live updates
- **click**: For better CLI argument handling (optional)

### Key Components

1. **Rich Console**: Central console object for all output
2. **Progress Bars**: Using Rich's Progress class with custom columns
3. **Live Display**: Real-time updates using Rich's Live feature
4. **Panels & Tables**: Structured output using Rich's layout components
5. **Checkpoint System**: JSON-based progress tracking
6. **Signal Handlers**: Graceful interrupt handling with signal module
7. **Streaming Writers**: Buffered file writers with immediate flush
8. **Smart Path Handling**: Intelligent output location determination
9. **Input Type Detection**: Automatic detection of file vs directory input

## 📸 Visual Examples

### File Transcription Progress

```
  Transcribing files ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% • 0:00:27 • 0:00:00
  Processing: audio.mp3 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 85% • 0:00:15
```

### Live Transcription Status

```
╔══════════════════════════════════════════════════════════╗
║           🎙️  LIVE TRANSCRIPTION                         ║
╚══════════════════════════════════════════════════════════╝
┌─────────────── Transcription ────────────────┐
│ [en] Hello, this is a test                   │
│ [pt] Olá, isto é um teste                    │
│ [en] The system is working perfectly          │
└───────────────────────────────────────────────┘
```

### Resume Configuration

```
     Transcription Configuration
┌──────────────────────┬─────────────┐
│ Files to Process     │ 7           │
│ Language Mode        │ Auto-detect │
│ Output Format        │ ALL         │
│ Model Size           │ BASE        │
│ Resume Mode          │ ENABLED     │
└──────────────────────┴─────────────┘
```

## 🎯 User Experience Improvements

1. **Immediate Feedback**: Users see activity immediately, no more wondering if the app is working
2. **Clear Progress**: Exact progress percentage and time estimates
3. **Better Error Messages**: Colored and formatted error messages with troubleshooting tips
4. **Professional Appearance**: The CLI now looks like a polished, professional tool
5. **Responsive Feel**: Animations and live updates make the CLI feel modern and responsive
6. **Resume Capability**: Never lose progress, even with unexpected interruptions
7. **Streaming Output**: See results as they're generated, not just at the end
8. **Resilient Operation**: Continues working even if files are deleted or become inaccessible
9. **Convenient Defaults**: Output files default to the same location as input files
10. **Flexible Input**: Can process individual files or entire directories with clear visual feedback

## 🏃 Running the Demo

To see all the visual enhancements in action, run:

```bash
uv run python transcript_pkg/demo_animations.py
```

This will demonstrate:

- Various spinner animations
- Progress bar styles
- Live updating displays
- Styled tables and panels

## 🔧 Customization

The visual elements can be easily customized by modifying:

- Color schemes in the console.print() calls
- Spinner styles in console.status() and Progress()
- Table styles and borders
- Panel borders and padding

All visual components use the Rich library's theming system, making it easy to maintain consistency across the application.

## 💾 Checkpoint File Structure

The checkpoint file (`.transcription_checkpoint.json`) stores:

```json
{
  "files_completed": ["file1.mp3", "file2.wav"],
  "current_file": "file3.mp3",
  "current_file_segments": [
    {
      "text": "Hello world",
      "start": 0.0,
      "end": 2.5,
      "language": "en"
    }
  ],
  "timestamp": "2024-01-15T14:30:22",
  "settings": {
    "language": "auto",
    "model": "base",
    "format": "all",
    "multilingual": true,
    "output_same_as_input": true,
    "is_single_file": false
  }
}
```

## Future Enhancement Ideas

- Add export functionality from the UI
- Add audio waveform visualization
- Support for custom color themes
- Integration with cloud transcription services
- Real-time collaboration features

## Live Transcription File Output

The live transcription mode now supports saving transcriptions to a file in real-time:

### Features

- **Streaming Output**: Transcriptions are written to file immediately as they're generated
- **Timestamped Entries**: Each transcription includes elapsed time from session start
- **Language Tags**: In multilingual mode, each line shows the detected language
- **Session Summary**: Complete statistics are appended at the end of the session
- **Header Information**: File starts with session timestamp and metadata

### Usage

```bash
# Save with timestamps (default)
transcript live --output session.txt

# Save without timestamps
transcript live --output session.txt --no-timestamps

# Multilingual with output
transcript live --multilingual --output meeting.txt
```

### Output Format

With timestamps:

```
Live Transcription Session - 2024-01-15 14:30:00
============================================================

[0012.5s] Hello, welcome to our meeting
[0025.3s] [pt] Olá, bem-vindo à nossa reunião
[0038.7s] Let's discuss the project status
```

Without timestamps:

```
Live Transcription Session - 2024-01-15 14:30:00
============================================================

Hello, welcome to our meeting
[pt] Olá, bem-vindo à nossa reunião
Let's discuss the project status
```

The file is written with line buffering (`buffering=1`) and explicit `flush()` calls to ensure real-time updates. You can `tail -f` the output file to watch transcriptions as they happen.

## Graceful Shutdown & Data Preservation

Both file and live transcription modes now ensure that all transcribed data is saved even when interrupted:

### File Transcription Mode

- **Enhanced Interrupt Handler**: The `GracefulInterruptHandler` now tracks all active writers
- **Automatic Writer Closing**: On Ctrl+C, all format writers (txt, srt, vtt) are properly closed
- **Data Flushing**: Ensures all buffered data is written to disk before exit
- **Progress Messages**: Shows which outputs were successfully saved

### Live Transcription Mode

- **Signal Handling**: Proper signal handler for SIGINT (Ctrl+C)
- **Finally Block**: Ensures file writer is closed even on unexpected errors
- **Summary Writing**: Session summary is written to file before closing
- **Status Feedback**: Clear messages about file saving status

### Usage Benefits

- **Never lose data**: All transcripts processed up to the interruption point are saved
- **Clean shutdown**: No corrupted files or missing data
- **Clear feedback**: Users see confirmation that their data was saved
- **Reliable operation**: Works even with unexpected errors or system issues

This ensures a professional, production-ready experience where users can trust that their transcription data is safe.
