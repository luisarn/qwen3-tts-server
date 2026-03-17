#!/bin/bash
# Run Qwen3-TTS server and generate sample audio

echo "=========================================="
echo "Qwen3-TTS Server with Audio Generation"
echo "=========================================="
echo ""

# Check if running on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo "⚠️  This server is designed for macOS with Apple Silicon"
    echo "   Continue anyway? (y/n)"
    read -r response
    if [[ "$response" != "y" ]]; then
        exit 1
    fi
fi

cd "$(dirname "$0")"
source .venv/bin/activate

echo "Starting Qwen3-TTS server..."
echo "Note: First run will download the Qwen3-TTS model (~1.2GB)"
echo ""

# Start server in background
python -m qwen3_tts_server.server --port 8000 &
SERVER_PID=$!

# Wait for server to be ready
echo -n "Waiting for server to start"
for i in {1..60}; do
    sleep 2
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo ""
        echo "✓ Server is ready!"
        break
    fi
    echo -n "."
done

echo ""
echo "=========================================="
echo "Generating Sample Speech"
echo "=========================================="

# Generate speech using bundled default voice
curl -s -X POST http://localhost:8000/v1/audio/speech \
    -F "input=Hello! This is a test of Qwen3-TTS voice cloning on Apple Silicon." \
    -F "model=qwen3-tts" \
    --output /tmp/sample_qwen3_tts.wav

if [ -f /tmp/sample_qwen3_tts.wav ] && [ -s /tmp/sample_qwen3_tts.wav ]; then
    FILESIZE=$(ls -lh /tmp/sample_qwen3_tts.wav | awk '{print $5}')
    echo "✓ Audio generated: /tmp/sample_qwen3_tts.wav ($FILESIZE)"
    echo ""
    echo "Playing audio now..."
    afplay /tmp/sample_qwen3_tts.wav
    echo "✓ Playback complete!"
else
    echo "✗ Failed to generate audio"
fi

echo ""
echo "=========================================="
echo "Server Information"
echo "=========================================="
echo "Server running at: http://localhost:8000"
echo "PID: $SERVER_PID"
echo ""
echo "Available endpoints:"
echo "  - GET  http://localhost:8000/health"
echo "  - GET  http://localhost:8000/v1/models"
echo "  - GET  http://localhost:8000/v1/audio/voices"
echo "  - POST http://localhost:8000/v1/audio/speech"
echo ""
echo "Generate more speech with:"
echo "  curl -X POST http://localhost:8000/v1/audio/speech \\"
echo "    -F \"input=Your text here\" \\"
echo "    --output output.wav"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Keep server running
wait $SERVER_PID
