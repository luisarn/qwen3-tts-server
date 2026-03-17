# Qwen3-TTS Server - Agent Guide

## Project Overview

Qwen3-TTS Server is an OpenAI-compatible Text-to-Speech (TTS) API server designed for Apple Silicon (M1/M2/M3/M4). It provides GPU-accelerated voice cloning using Apple's MLX framework via the `mlx-audio` library.

**Key Features:**
- OpenAI-compatible `/v1/audio/speech` API endpoint
- Voice cloning with Qwen3-TTS
- Default reference audio support for consistent voice output
- Optional API key authentication

**Project Origin:**
This is a stripped-down TTS server focused exclusively on TTS capabilities.

## Technology Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.10+ |
| Web Framework | FastAPI |
| ASGI Server | Uvicorn |
| ML Framework | MLX (Apple Silicon GPU acceleration) |
| TTS Backend | mlx-audio |
| Package Manager | uv |
| Build System | hatchling |

## Project Structure

```
qwen3-tts-server/
├── pyproject.toml              # Project configuration, dependencies, tool settings
├── uv.lock                     # Locked dependency versions (uv package manager)
├── .python-version             # Python version specification
├── README.md                   # User documentation
├── AGENTS.md                   # This file
├── .gitignore                  # Git ignore patterns
├── data/
│   └── reference_audio/
│       └── default_voice.wav   # Bundled default reference audio for voice cloning
├── qwen3_tts_server/           # Main package
│   ├── __init__.py             # Package exports (TTSEngine, clone_voice, generate_speech)
│   ├── server.py               # FastAPI application with TTS endpoints
│   ├── cli.py                  # CLI entry point (qwen3-tts-server serve)
│   ├── audio/
│   │   ├── __init__.py         # Audio module exports
│   │   └── tts.py              # TTSEngine class and helper functions
│   └── api/
│       ├── __init__.py         # API module exports
│       └── models.py           # Pydantic models for OpenAI-compatible API
├── tests/
│   ├── conftest.py             # pytest configuration and fixtures
│   └── test_audio.py           # TTS engine and API model tests
└── examples/
    ├── tts_api_client.py       # HTTP API client example
    └── tts_qwen3_voice_cloning.py  # Direct library usage example
```

## Build and Development Commands

### Environment Setup

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Sync dependencies and create virtual environment
uv sync

# Sync with development dependencies (tests, linting)
uv sync --extra dev
```

### Running the Server

```bash
# Development mode
uv run -m qwen3_tts_server.server --port 8000

# With default reference audio
uv run -m qwen3_tts_server.server --port 8000 --default-ref-audio /path/to/voice.wav

# With API key authentication
uv run -m qwen3_tts_server.server --port 8000 --api-key your-secret-key

# Using CLI (if installed)
qwen3-tts-server serve --port 8000
```

### Testing

```bash
# Run all tests (fast, no model loading)
uv run --extra dev pytest

# Run with slow tests (requires model downloads)
uv run --extra dev pytest --run-slow

# Run integration tests (requires running server)
uv run --extra dev pytest --server-url http://localhost:8000
```

### Linting and Formatting

```bash
# Check formatting
uv run --extra dev black --check .
uv run --extra dev ruff check .

# Auto-format code
uv run --extra dev black .
uv run --extra dev ruff check --fix .

# Type checking
uv run --extra dev mypy .
```

### Dependency Management

```bash
# Update dependencies and regenerate uv.lock
uv lock --upgrade

# Add a new dependency
uv add <package>

# Add a development dependency
uv add --dev <package>
```

## Code Style Guidelines

### License Headers

All Python files must include the SPDX license identifier:

```python
# SPDX-License-Identifier: Apache-2.0
```

### Import Style

- Use absolute imports within the package
- Group imports: stdlib, third-party, local
- Follow PEP 8 import ordering

Example:
```python
# SPDX-License-Identifier: Apache-2.0
"""Module docstring."""

import logging
from pathlib import Path
from typing import Union

import numpy as np

from .tts import TTSEngine
```

### Code Formatting

- **Line length:** 88 characters (Black default)
- **Target Python versions:** 3.10, 3.11, 3.12, 3.13
- **Formatter:** Black
- **Linter:** Ruff (with rules: E, F, W, I, N, UP, B, SIM)

### Type Hints

- Use type hints for function signatures
- Use `Union` for Python 3.10 compatibility (avoid `|` in type hints where possible)
- Run mypy for type checking

## Testing Strategy

### Test Organization

- **Unit tests:** Fast tests that don't require model loading (default)
- **Slow tests:** Tests requiring model downloads (marked with `@pytest.mark.slow`)
- **Integration tests:** Tests requiring a running server (marked with `@pytest.mark.integration`)

### Test Markers

```python
# Skip unless --run-slow is passed
@pytest.mark.skip(reason="Requires mlx-audio and models downloaded")
def test_slow_feature():
    pass
```

### Running Specific Test Categories

```bash
# Only fast tests (default)
pytest

# Include slow tests
pytest --run-slow

# Integration tests only
pytest -m integration --server-url http://localhost:8000
```

## API Endpoints

| Endpoint | Method | Description | Auth Required |
|----------|--------|-------------|---------------|
| `/health` | GET | Health check | No |
| `/v1/models` | GET | List available models | Yes (if configured) |
| `/v1/audio/speech` | POST | Generate speech from text | Yes (if configured) |
| `/v1/audio/voices` | GET | List available voices | Yes (if configured) |

### Authentication

When `--api-key` is provided, all endpoints except `/health` require:
```
Authorization: Bearer <api-key>
```

## Model

**Qwen3-TTS**: Multilingual voice cloning model
- Model ID: `mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16`
- Voice cloning: Yes (requires reference audio)

## Deployment Considerations

### Platform Requirements

- **macOS only** - Requires Apple Silicon (M1, M2, M3, M4)
- **Python 3.10+**
- **mlx-audio** handles model downloads automatically on first use

### Default Reference Audio

The server includes a bundled default reference audio at:
```
data/reference_audio/default_voice.wav
```

Priority for reference audio (highest to lowest):
1. Uploaded `ref_audio` in request
2. `--default-ref-audio` command line argument
3. Bundled `data/reference_audio/default_voice.wav`

### Environment Variables

No required environment variables. Configuration is done via command-line arguments.

## Security Considerations

1. **API Key:** Use `--api-key` for production deployments
2. **Default Reference Audio:** Verify the bundled reference audio is appropriate for your use case
3. **Model Downloads:** Models are downloaded from HuggingFace on first use; ensure network security

## Common Development Tasks

### Modifying TTS Behavior

1. Update the TTSEngine class in `qwen3_tts_server/audio/tts.py`
2. Add tests in `tests/test_audio.py`
3. Update documentation in `README.md` and this file

### Adding a New API Endpoint

1. Add Pydantic models in `qwen3_tts_server/api/models.py`
2. Export from `qwen3_tts_server/api/__init__.py`
3. Add endpoint in `qwen3_tts_server/server.py`
4. Use `Depends(verify_api_key)` for authenticated endpoints
5. Add tests in `tests/test_audio.py`

### Version Bump

1. Update version in `pyproject.toml`
2. Update version in `qwen3_tts_server/__init__.py`
3. Update version in `qwen3_tts_server/server.py`
4. Update `uv.lock` if dependencies changed: `uv lock`
