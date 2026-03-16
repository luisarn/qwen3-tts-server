# Qwen3-TTS Server

**OpenAI-compatible TTS API for Apple Silicon** - GPU-accelerated Text-to-Speech with Voice Cloning on Mac

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Apple Silicon](https://img.shields.io/badge/Apple-Silicon-black.svg)](https://support.apple.com/en-us/HT211814)

## Overview

A stripped-down TTS server focused on **Qwen3-TTS voice cloning** using Apple's MLX framework. This is a minimal implementation that removes all LLM/MLLM/embedding functionality from the original vllm-mlx project, keeping only the TTS capabilities.

**Features:**
- OpenAI-compatible `/v1/audio/speech` API
- Voice cloning with Qwen3-TTS and Chatterbox
- Multiple TTS models: Kokoro, VibeVoice, VoxCPM, CSM, CosyVoice
- Default reference audio support for consistent voice output
- GPU acceleration on Apple Silicon (M1, M2, M3, M4)

## Installation

### Using uv (Recommended)

This project uses [uv](https://docs.astral.sh/uv/) for fast, reliable Python package management.

```bash
# Clone the repository
git clone https://github.com/vllm-mlx/qwen3-tts-server.git
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
pip install git+https://github.com/vllm-mlx/qwen3-tts-server.git
```

## Quick Start

### Start the Server

```bash
# Basic usage (downloads Qwen3-TTS on first run)
uv run -m qwen3_tts_server.server --port 8000

# With default reference audio for voice cloning
uv run -m qwen3_tts_server.server --port 8000 --default-ref-audio /path/to/voice.wav

# With API key authentication
uv run -m qwen3_tts_server.server --port 8000 --api-key your-secret-key
```

Or if installed with pip:

```bash
qwen3-tts-server serve --port 8000
```

### Generate Speech

```bash
# Using curl with reference audio for voice cloning
curl -X POST http://localhost:8000/v1/audio/speech \
  -F "model=qwen3-tts" \
  -F "input=Hello, this is a voice cloning test!" \
  -F "ref_audio=@/path/to/reference_voice.wav" \
  --output output.wav

# Using default reference audio (if configured on server)
curl -X POST http://localhost:8000/v1/audio/speech \
  -F "model=qwen3-tts" \
  -F "input=Hello world!" \
  --output output.wav
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
| `/v1/audio/voices` | GET | List available voices for a model |

**Model aliases:** The API accepts `qwen3-tts` as an alias for the full model name `mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16`.

### POST /v1/audio/speech

**Parameters:**
- `model` (string): Model ID - `qwen3-tts` (default)
- `input` (string): Text to synthesize
- `voice` (string): Voice ID - `voice_clone` for Qwen3-TTS, or see Kokoro voices below
- `speed` (float): Speech speed - 0.5 to 2.0 (default: 1.0)
- `response_format` (string): Output format - `wav` (default)
- `ref_audio` (file): Reference audio for voice cloning (optional, required for voice cloning models if no default is set)

**Kokoro Voices:**
- American Female: `af_heart`, `af_bella`, `af_nicole`, `af_sarah`, `af_sky`
- American Male: `am_adam`, `am_michael`
- British Female: `bf_emma`, `bf_isabella`
- British Male: `bm_george`, `bm_lewis`

## Supported Models

| Model | Description | Voice Cloning | Default Model ID |
|-------|-------------|---------------|------------------|
| Qwen3-TTS | Multilingual voice cloning | Yes | `mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16` |
| Kokoro | Fast, lightweight, multiple voices | No | `mlx-community/Kokoro-82M-bf16` |
| Chatterbox | Expressive, multilingual | Yes | `mlx-community/chatterbox-turbo-fp16` |
| VibeVoice | Realtime, low latency | No | `mlx-community/VibeVoice-Realtime-0.5B-4bit` |
| VoxCPM | Chinese/English, high quality | No | `mlx-community/VoxCPM1.5` |
| CSM | Conversational speech model | No | See mlx-audio docs |
| CosyVoice | Multilingual, expressive | No | See mlx-audio docs |

## Project Structure

```
qwen3-tts-server/
├── pyproject.toml          # Minimal TTS dependencies
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
- Original [vllm-mlx](https://github.com/waybarrios/vllm-mlx) project
