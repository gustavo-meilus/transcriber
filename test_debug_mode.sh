#!/bin/bash

# Test script for debug mode
echo "=== Whisper Transcription Debug Mode Test ==="
echo "============================================="

# Check if input file is provided
if [ $# -eq 0 ]; then
    echo "Usage: ./test_debug_mode.sh <audio_file> [model_size]"
    echo ""
    echo "Examples:"
    echo "  ./test_debug_mode.sh input/aula_containers.wav"
    echo "  ./test_debug_mode.sh input/test.mp3 tiny"
    echo "  ./test_debug_mode.sh input/podcast.wav base"
    echo ""
    exit 1
fi

AUDIO_FILE="$1"
MODEL_SIZE="${2:-tiny}"  # Default to tiny if not specified

# Check if file exists
if [ ! -f "$AUDIO_FILE" ]; then
    echo "Error: Audio file '$AUDIO_FILE' not found!"
    exit 1
fi

# Display test configuration
echo ""
echo "Test Configuration:"
echo "  Audio File: $AUDIO_FILE"
echo "  Model Size: $MODEL_SIZE"
echo "  Debug Mode: ENABLED"
echo ""
echo "Starting transcription with debug statistics..."
echo "============================================="
echo ""

# Run transcription with debug mode
python -m transcript_pkg.cli file -i "$AUDIO_FILE" -m "$MODEL_SIZE" --language auto --debug

echo ""
echo "Debug test complete!" 