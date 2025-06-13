# 📘 Complete Guide: Using the Transcriber Colab Notebook

This guide will walk you through everything you need to know to use the audio transcription notebook on Google Colab.

## 📥 Getting the Notebook into Colab

### Method 1: Direct from GitHub (Recommended)

1. **Direct Link Method:**

   - Simply click this link: [Open in Colab](https://colab.research.google.com/github/gustavo-meilus/transcriber/blob/main/transcript_colab.ipynb)
   - The notebook will open directly in Colab

2. **Manual GitHub Method:**
   - Go to Google Colab: https://colab.research.google.com/
   - Click "GitHub" tab
   - Enter the repository URL: `https://github.com/gustavo-meilus/transcriber`
   - Select `transcript_colab.ipynb` from the list
   - Click to open

### Method 2: Upload from Local Computer

1. **Download the notebook:**

   ```bash
   wget https://raw.githubusercontent.com/gustavo-meilus/transcriber/main/transcript_colab.ipynb
   ```

   Or download manually from the GitHub repository

2. **Upload to Colab:**
   - Go to https://colab.research.google.com/
   - Click "Upload" tab
   - Drag and drop `transcript_colab.ipynb` or click "Choose file"
   - The notebook will open automatically

### Method 3: Save to Your Google Drive

1. **First time setup:**

   - Open the notebook using Method 1 or 2
   - Go to File → Save a copy in Drive
   - This creates your own copy in `/My Drive/Colab Notebooks/`

2. **Future access:**
   - Go to https://colab.research.google.com/
   - Click "Google Drive" tab
   - Navigate to your saved copy
   - Open and use

## 🚀 Step-by-Step Usage Guide

### Step 1: Enable GPU (Important!)

1. **Before running any code:**

   - Go to menu: Runtime → Change runtime type
   - Hardware accelerator: GPU
   - GPU type: T4 (free tier)
   - Click "Save"

2. **Verify GPU is enabled:**
   - Run the first cell (Shift+Enter)
   - You should see: "✅ GPU is available!"
   - If not, repeat step 1

### Step 2: Run Setup Cells

Run cells in order by pressing **Shift+Enter** on each:

1. **Cell 1: GPU Check**

   ```
   ✅ GPU is available!
   GPU Details:
     Tesla T4
   ```

2. **Cell 2: Install Dependencies**

   ```
   📦 Installing dependencies...
   ✅ Dependencies installed successfully!
   ```

3. **Cell 3: Mount Google Drive**

   - You'll see a prompt to authorize Google Drive
   - Click the link and authorize
   - Copy the authorization code
   - Paste it back in Colab

   ```
   ✅ Google Drive mounted successfully!
   ```

4. **Cell 4: Load Functions**
   ```
   ✅ Transcription functions loaded
   ```

### Step 3: Configure Your Transcription

In the **Configuration cell**, modify these variables:

```python
# CHANGE THIS to your audio file or folder path
AUDIO_PATH = "/content/drive/MyDrive/audio/sample.mp3"

# Optional: Change output directory (default: same as input)
OUTPUT_DIR = "/content/drive/MyDrive/transcriptions/"

# Choose model size: "tiny", "base", "small", "medium", "large"
MODEL_SIZE = "base"  # Recommended for balance

# Language: "en", "pt", or None for auto-detect
LANGUAGE = None

# Output format: "txt", "srt", "vtt", "all"
OUTPUT_FORMAT = "txt"
```

### Step 4: Run Transcription

Run the main transcription cell. You'll see:

1. **File Detection:**

   ```
   Found 1 audio file(s):
     🎵 sample.mp3
   ```

2. **Model Loading:**

   ```
   ✓ GPU detected: Tesla T4 (15.0GB)
   Loading Whisper base model on CUDA...
   ✓ Model loaded successfully on CUDA!
   ```

3. **Transcription Progress:**

   ```
   Transcribing: sample.mp3
   Processing... ━━━━━━━━━━━━━━━━━━━━ 100% 0:00:45
   ```

4. **Results:**

   ```
   ✓ Transcription complete!
     Audio duration: 180.5s
     Processing time: 45.2s
     Speed: 4.0x realtime
     Segments: 25
     Language: en

   Output files:
     📄 /content/drive/MyDrive/audio/sample.txt
   ```

## 📁 Preparing Your Audio Files

### Option 1: Upload to Google Drive (Web Interface)

1. Go to https://drive.google.com
2. Create a folder (e.g., "audio")
3. Upload your audio files:
   - Drag and drop files
   - Or click "New" → "File upload"

### Option 2: Upload via Colab

Add this code cell to upload files directly:

```python
from google.colab import files

# Upload files
uploaded = files.upload()

# Move to Drive
import shutil
for filename in uploaded.keys():
    shutil.move(filename, f'/content/drive/MyDrive/audio/{filename}')
    print(f"Moved {filename} to Drive")
```

### Option 3: Download from URL

Add this code to download audio from URLs:

```python
!wget -P /content/drive/MyDrive/audio/ "https://example.com/audio.mp3"
```

## 💡 Common Use Cases

### 1. Single File Transcription

```python
AUDIO_PATH = "/content/drive/MyDrive/recordings/interview.mp3"
OUTPUT_FORMAT = "txt"
```

### 2. Batch Process a Folder

```python
AUDIO_PATH = "/content/drive/MyDrive/podcasts/"  # All files in folder
OUTPUT_FORMAT = "all"  # Generate txt, srt, and vtt
```

### 3. Portuguese Transcription

```python
AUDIO_PATH = "/content/drive/MyDrive/aula.wav"
LANGUAGE = "pt"
MODEL_SIZE = "large"  # Better for Portuguese
```

### 4. Multilingual Content

```python
AUDIO_PATH = "/content/drive/MyDrive/meeting.mp4"
MULTILINGUAL = True  # Detect language changes
LANGUAGE = None  # Auto-detect
```

### 5. Custom Output Location

```python
AUDIO_PATH = "/content/drive/MyDrive/raw_audio/"
OUTPUT_DIR = "/content/drive/MyDrive/transcriptions/project1/"
```

## 🔧 Troubleshooting

### Issue: "No GPU detected"

**Solution:**

1. Runtime → Change runtime type → GPU → Save
2. Runtime → Restart runtime
3. Run cells again

### Issue: "Audio file not found"

**Solution:**

1. Check the file path is correct (case-sensitive!)
2. Ensure Drive is mounted (run cell 3)
3. Use forward slashes `/` not backslashes `\`
4. Try listing files:
   ```python
   import os
   os.listdir('/content/drive/MyDrive/audio/')
   ```

### Issue: "Out of memory"

**Solution:**

1. Use a smaller model (`tiny` or `base`)
2. Process files one at a time
3. Runtime → Restart runtime (clears memory)

### Issue: "Drive not mounted"

**Solution:**

1. Re-run the Drive mount cell
2. Click the authorization link
3. Allow access to your Google account
4. Copy and paste the authorization code

### Issue: Slow processing

**Check:**

1. GPU is enabled (should be ~3-5x faster than CPU)
2. Model size (larger = slower but more accurate)
3. File size (very long files take time)

## 📊 Performance Expectations

With Colab's T4 GPU:

| Model Size | Speed (vs realtime) | Quality | Use Case          |
| ---------- | ------------------- | ------- | ----------------- |
| tiny       | ~50x                | Basic   | Quick drafts      |
| base       | ~30x                | Good    | General use       |
| small      | ~20x                | Better  | Important content |
| medium     | ~10x                | Great   | Professional      |
| large      | ~5x                 | Best    | Publication       |

## 🎯 Pro Tips

1. **Organize Your Files:**

   ```
   /My Drive/
   ├── audio/
   │   ├── project1/
   │   ├── project2/
   │   └── raw/
   └── transcriptions/
       ├── project1/
       └── project2/
   ```

2. **Process Large Batches:**

   - Break into folders of 10-20 files
   - Use the notebook's session time efficiently
   - Save outputs to organized folders

3. **Use Appropriate Models:**

   - `tiny/base` for drafts and quick checks
   - `small/medium` for general transcription
   - `large` for final/professional output

4. **Monitor GPU Usage:**

   - Check GPU RAM: `!nvidia-smi`
   - Free tier has ~15GB GPU memory
   - Large model uses ~3-4GB

5. **Save Your Work:**
   - Outputs auto-save to Drive
   - Save notebook: File → Save
   - Make a copy for different projects

## 🔄 Workflow Example

Here's a complete workflow for transcribing podcast episodes:

1. **Organize files in Drive:**

   ```
   /My Drive/podcasts/season1/
   ├── episode01.mp3
   ├── episode02.mp3
   └── episode03.mp3
   ```

2. **Configure notebook:**

   ```python
   AUDIO_PATH = "/content/drive/MyDrive/podcasts/season1/"
   OUTPUT_DIR = "/content/drive/MyDrive/transcriptions/season1/"
   MODEL_SIZE = "small"  # Good quality/speed balance
   OUTPUT_FORMAT = "all"  # Get all formats
   ```

3. **Run and monitor:**

   - Total time estimate: 3 files × 30 min each ÷ 20x speed ≈ 5 minutes

4. **Check results:**
   ```
   /My Drive/transcriptions/season1/
   ├── episode01.txt
   ├── episode01.srt
   ├── episode01.vtt
   ├── episode02.txt
   ├── episode02.srt
   └── ...
   ```

## 📝 Notes

- **Free Colab Limits:** ~12 hour sessions, GPU availability varies
- **Colab Pro:** More GPU time, better GPUs, longer sessions
- **File Size:** No strict limit, but very large files (>2GB) may timeout
- **Privacy:** Your audio files stay in your Google Drive

## 🆘 Getting Help

1. **Check the notebook's built-in examples** (Cell 7)
2. **Use the interactive file browser** (Cell 8)
3. **Review error messages** - they're designed to be helpful
4. **GitHub Issues:** Report problems with reproduction steps

Happy transcribing! 🎉
