# Qwen3-TTS Server

**OpenAI-compatible TTS API for Apple Silicon** - GPU-accelerated Text-to-Speech with Voice Cloning on Mac

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Apple Silicon](https://img.shields.io/badge/Apple-Silicon-black.svg)](https://support.apple.com/en-us/HT211814)

## Overview

A focused TTS server for **Qwen3-TTS voice cloning** using Apple's MLX framework.

**Features:**
- OpenAI-compatible `/v1/audio/speech` API
- Voice cloning with Qwen3-TTS
- Default reference audio support for consistent voice output
- GPU acceleration on Apple Silicon (M1, M2, M3, M4)

## Installation

### Using uv (Recommended)

This project uses [uv](https://docs.astral.sh/uv/) for fast, reliable Python package management.

```bash
# Clone the repository
git clone https://github.com/linguist/qwen3-tts-server.git
cd qwen3-tts-server

# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Sync dependencies and create virtual environment
uv sync

# Or with development dependencies (tests, linting)
uv sync --extra dev
```

### Using pip

```bash
# Install from source
pip install -e .

# Or install with pip
pip install git+https://github.com/linguist/qwen3-tts-server.git
```

## Quick Start

### Start the Server with uv

```bash
# Basic usage (downloads Qwen3-TTS model on first run)
uv run -m qwen3_tts_server.server --port 8000

# With default reference audio for voice cloning
uv run -m qwen3_tts_server.server --port 8000 --default-ref-audio /path/to/voice.wav

# With API key authentication
uv run -m qwen3_tts_server.server --port 8000 --api-key your-secret-key
```

Or using the CLI:

```bash
uv run qwen3-tts-server serve --port 8000
```

### Generate Speech via OpenAI-Compatible API

**Endpoint:** `POST /v1/audio/speech`

**Parameters:**
- `model` (string): `qwen3-tts` (required)
- `input` (string): Text to synthesize (required)
- `voice` (string): `voice_clone` (required for OpenAI compatibility)
- `speed` (float): Speech speed 0.5-2.0 (default: 1.0)
- `response_format` (string): `wav` (default)

**Using curl with JSON (OpenAI format):**
```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen3-tts", "input": "Hello world!", "voice": "voice_clone"}' \
  --output output.wav
```

**Using curl with form-data:**
```bash
# With reference audio for voice cloning
curl -X POST http://localhost:8000/v1/audio/speech \
  -F "model=qwen3-tts" \
  -F "input=Hello, this is a voice cloning test!" \
  -F "voice=voice_clone" \
  -F "ref_audio=@/path/to/reference_voice.wav" \
  --output output.wav

# Using default reference audio (if configured on server)
curl -X POST http://localhost:8000/v1/audio/speech \
  -F "model=qwen3-tts" \
  -F "input=Hello world!" \
  -F "voice=voice_clone" \
  --output output.wav
```

**Using OpenAI Python Client:**
```python
import openai

client = openai.OpenAI(
    base_url='http://localhost:8000/v1',
    api_key='dummy-key'  # Not required unless server has --api-key
)

response = client.audio.speech.create(
    model='qwen3-tts',
    input='Hello from the OpenAI client!',
    voice='voice_clone',
    speed=1.0
)
response.stream_to_file('output.wav')
```

### Using Python

```python
from qwen3_tts_server import clone_voice, TTSEngine

# Clone voice from reference audio
audio = clone_voice(
    "Hello, this is my cloned voice!",
    ref_audio="/path/to/reference_voice.wav"
)

# Save to file
engine = TTSEngine()
engine.save(audio, "output.wav")
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/v1/models` | GET | List available models |
| `/v1/audio/speech` | POST | Generate speech from text |
| `/v1/audio/voices` | GET | List available voices |

### POST /v1/audio/speech

**Endpoint:** `/v1/audio/speech`

**OpenAI-Compatible Parameters:**
- `model` (string): **Must be `qwen3-tts`**
- `input` (string): Text to synthesize (required)
- `voice` (string): **Must be `voice_clone`** (required for OpenAI compatibility)
- `speed` (float): Speech speed - 0.5 to 2.0 (default: 1.0)
- `response_format` (string): Output format - `wav` (default)

**Additional Parameters:**
- `ref_audio` (file): Reference audio for voice cloning (optional if server has `--default-ref-audio` configured)

**Response:** Audio file (WAV format)

## Model

**Qwen3-TTS**: Multilingual voice cloning model
- Model ID: `mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16`
- Voice cloning: Yes (requires reference audio)
- Supports voice cloning from reference audio files

## Project Structure

```
qwen3-tts-server/
├── pyproject.toml          # Project dependencies
├── qwen3_tts_server/
│   ├── __init__.py         # TTS exports
│   ├── server.py           # FastAPI TTS server
│   ├── cli.py              # CLI entry point
│   ├── audio/
│   │   ├── __init__.py
│   │   └── tts.py          # TTSEngine class
│   └── api/
│       ├── __init__.py
│       └── models.py       # Pydantic models
├── examples/
│   ├── tts_api_client.py   # API client example
│   └── tts_qwen3_voice_cloning.py  # Voice cloning example
└── tests/
    └── test_audio.py       # TTS tests
```

## Development

Common `uv` commands:

```bash
# Start the server
uv run -m qwen3_tts_server.server --port 8000

# Run tests
uv run --extra dev pytest

# Run linting
uv run --extra dev ruff check .
uv run --extra dev black --check .

# Format code
uv run --extra dev black .
uv run --extra dev ruff check --fix .

# Update dependencies
uv lock --upgrade
```

## Dependencies

- `fastapi>=0.100.0` - Web framework
- `uvicorn>=0.23.0` - ASGI server
- `python-multipart>=0.0.6` - Form data parsing
- `numpy>=1.24.0` - Numerical computing
- `mlx-audio>=0.2.9` - TTS models (via mlx-audio)
- `soundfile>=0.12.0` - Audio I/O
- `scipy>=1.10.0` - Audio processing

## License

Apache 2.0 - see [LICENSE](LICENSE) for details.

## Acknowledgments

- [MLX](https://github.com/ml-explore/mlx) - Apple's ML framework
- [mlx-audio](https://github.com/Blaizzy/mlx-audio) - Text-to-Speech models
