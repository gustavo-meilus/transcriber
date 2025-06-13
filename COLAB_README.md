# Google Colab Notebook for Audio Transcription

This repository includes a Google Colab notebook (`transcript_colab.ipynb`) that allows you to transcribe audio files directly from your Google Drive using GPU acceleration.

## 🚀 Quick Start

1. **Open in Colab**:

   - Click on `transcript_colab.ipynb` in this repository
   - Click "Open in Colab" button
   - Or use this direct link: [Open in Colab](https://colab.research.google.com/github/gustavo-meilus/transcriber/blob/main/transcript_colab.ipynb)

2. **Enable GPU**:

   - Go to Runtime → Change runtime type
   - Select GPU (T4 is available in free tier)
   - Save

3. **Run the notebook**:
   - Execute cells in order (Shift+Enter)
   - Authorize Google Drive access when prompted
   - Modify the configuration cell with your audio file path
   - Run the transcription!

## 📋 Features

- **GPU Acceleration**: Automatically uses Colab's free GPU for 3-5x faster processing
- **Google Drive Integration**: Read/write files directly from your Drive
- **Multiple Languages**: English, Portuguese, and auto-detection
- **Multiple Formats**: TXT, SRT, VTT outputs
- **Batch Processing**: Transcribe single files or entire folders
- **Interactive UI**: Optional widget-based file selector

## 📁 Usage Examples

### Single File

```python
AUDIO_PATH = "/content/drive/MyDrive/audio/interview.mp3"
OUTPUT_FORMAT = "txt"
```

### Entire Folder

```python
AUDIO_PATH = "/content/drive/MyDrive/podcasts/"
OUTPUT_FORMAT = "all"  # Generate all formats
```

### Portuguese with Large Model

```python
AUDIO_PATH = "/content/drive/MyDrive/aula.wav"
MODEL_SIZE = "large"
LANGUAGE = "pt"
```

## 🛠️ Configuration Options

- **MODEL_SIZE**: `tiny`, `base`, `small`, `medium`, `large`
- **LANGUAGE**: `en`, `pt`, or `None` (auto-detect)
- **OUTPUT_FORMAT**: `txt`, `srt`, `vtt`, `all`
- **MULTILINGUAL**: `True/False` (detect language changes)
- **OUTPUT_DIR**: Custom output directory or `None` (same as input)

## 📊 Performance

With Colab's T4 GPU:

- **Base model**: ~30-50x realtime
- **Large model**: ~10-15x realtime

## 🔧 Troubleshooting

1. **No GPU detected**: Enable GPU in Runtime settings
2. **File not found**: Check path and Drive mount
3. **Out of memory**: Use smaller model or restart runtime
4. **Import errors**: Re-run installation cell

## 📝 Notes

- Free Colab has usage limits - for heavy usage consider Colab Pro
- Very long audio files may timeout - split them if needed
- Models are downloaded on first use (cached for session)

## 📖 Detailed Setup Guide

For comprehensive step-by-step instructions, see [COLAB_SETUP_GUIDE.md](COLAB_SETUP_GUIDE.md)

## 🤝 Contributing

Feel free to submit issues or improvements to the notebook!
