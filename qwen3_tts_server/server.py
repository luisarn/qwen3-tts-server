# SPDX-License-Identifier: Apache-2.0
"""
Qwen3-TTS server with OpenAI-compatible API.

This module provides a FastAPI server that exposes an OpenAI-compatible
API for text-to-speech (TTS) using MLX on Apple Silicon.

Usage:
    python -m qwen3_tts_server.server --port 8000 --default-ref-audio /path/to/voice.wav

The server provides:
    - POST /v1/audio/speech - Text-to-Speech (TTS with voice cloning support)
    - GET /v1/audio/voices - List available TTS voices
    - GET /v1/models - List available models
    - GET /health - Health check
"""

import argparse
import logging
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import Response

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Global state
_api_key: str | None = None
_default_ref_audio: str | None = None
_tts_engines: dict[str, Any] = {}

# Default TTS model
DEFAULT_TTS_MODEL = "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16"


def verify_api_key(request: Request) -> None:
    """Verify API key if authentication is enabled."""
    if _api_key is None:
        return

    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")

    token = auth_header[7:]  # Remove "Bearer " prefix
    if token != _api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger.info("Starting Qwen3-TTS server")
    yield
    # Cleanup
    logger.info("Shutting down Qwen3-TTS server")
    for engine in _tts_engines.values():
        try:
            engine.unload()
        except Exception:
            pass


app = FastAPI(
    title="Qwen3-TTS Server",
    description="OpenAI-compatible TTS API for Apple Silicon",
    version="0.2.6",
    lifespan=lifespan,
)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "service": "qwen3-tts"}


@app.get("/v1/models", dependencies=[Depends(verify_api_key)])
async def list_models():
    """List available TTS models."""
    from .api.models import ModelInfo, ModelsResponse

    models = [
        ModelInfo(id="qwen3-tts"),
    ]
    return ModelsResponse(data=models)


@app.post("/v1/audio/speech", dependencies=[Depends(verify_api_key)])
async def create_speech(
    model: str = Form("qwen3-tts"),
    input: str = Form(""),
    voice: str = Form("voice_clone"),
    speed: float = Form(1.0),
    response_format: str = Form("wav"),
    ref_audio: UploadFile | None = File(None),
):
    """
    Generate speech from text (OpenAI TTS API compatible).

    Supports voice cloning when a reference audio file is provided via `ref_audio`.
    If no ref_audio is provided and a default reference audio was configured at
    server startup, the default will be used.

    Model aliases:
    - qwen3-tts -> mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16
    """
    global _tts_engines

    try:
        from .audio.tts import TTSEngine

        # Map model aliases to full names
        model_map = {
            "qwen3-tts": DEFAULT_TTS_MODEL,
        }
        model_name = model_map.get(model, model)

        # Get or create engine for this model
        engine = _tts_engines.get(model_name)
        if engine is None:
            logger.info(f"Loading TTS model: {model_name}")
            engine = TTSEngine(model_name)
            engine.load()
            _tts_engines[model_name] = engine

        # Handle reference audio for voice cloning
        ref_audio_path = None
        temp_file_created = False

        if ref_audio is not None:
            # Save uploaded file temporarily with correct extension
            original_filename = ref_audio.filename or "audio.wav"
            suffix = Path(original_filename).suffix.lower()
            if suffix not in [".wav", ".mp3", ".m4a", ".flac", ".ogg"]:
                suffix = ".wav"

            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                content = await ref_audio.read()
                tmp.write(content)
                ref_audio_path = tmp.name
                temp_file_created = True
        elif _default_ref_audio is not None:
            # Use default reference audio if set
            ref_audio_path = _default_ref_audio
            logger.debug(f"Using default reference audio: {ref_audio_path}")

        try:
            # Check if voice cloning model needs reference audio
            if (
                "qwen3" in model_name.lower()
                and "tts" in model_name.lower()
                and ref_audio_path is None
            ):
                raise HTTPException(
                    status_code=400,
                    detail="Qwen3-TTS requires reference audio. Provide ref_audio file or configure --default-ref-audio",
                )

            # Generate speech with optional voice cloning
            audio = engine.generate(
                input, voice=voice, speed=speed, ref_audio=ref_audio_path
            )
            audio_bytes = engine.to_bytes(audio, format=response_format)
        except HTTPException:
            raise
        finally:
            # Clean up temp file only if we created one
            if temp_file_created and ref_audio_path and os.path.exists(ref_audio_path):
                os.unlink(ref_audio_path)

        content_type = (
            "audio/wav" if response_format == "wav" else f"audio/{response_format}"
        )
        return Response(content=audio_bytes, media_type=content_type)

    except ImportError as e:
        logger.error(f"mlx-audio not installed: {e}")
        raise HTTPException(
            status_code=503,
            detail="mlx-audio not installed. Install with: pip install mlx-audio",
        )
    except Exception as e:
        logger.error(f"TTS generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/audio/voices", dependencies=[Depends(verify_api_key)])
async def list_voices(model: str = "qwen3-tts"):
    """List available voices for a TTS model."""
    model_lower = model.lower()
    if "qwen3" in model_lower and "tts" in model_lower:
        return {"voices": ["voice_clone"], "requires_ref_audio": True}
    else:
        return {"voices": ["default"]}


def main():
    """Main entry point for the server."""
    global _api_key, _default_ref_audio

    parser = argparse.ArgumentParser(
        description="Qwen3-TTS Server - OpenAI-compatible TTS API for Apple Silicon",
    )
    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="Host to bind (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="Port to bind (default: 8000)"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key for authentication (if not set, no auth required)",
    )
    parser.add_argument(
        "--default-ref-audio",
        type=str,
        default=None,
        help="Default reference audio file for TTS voice cloning (used when no ref_audio is uploaded)",
    )

    args = parser.parse_args()

    # Configure API key
    if args.api_key:
        _api_key = args.api_key
        print("=" * 60)
        print("SECURITY CONFIGURATION")
        print("=" * 60)
        print("  Authentication: ENABLED (API key required)")
        print("=" * 60)
    else:
        print("  Authentication: DISABLED - Use --api-key to enable")

    # Configure default reference audio
    if args.default_ref_audio:
        if not os.path.exists(args.default_ref_audio):
            print(f"Error: Default reference audio file not found: {args.default_ref_audio}")
            exit(1)
        _default_ref_audio = args.default_ref_audio
        print(f"Default reference audio set: {_default_ref_audio}")

    print(f"Starting Qwen3-TTS server at http://{args.host}:{args.port}")
    print("\nEndpoints:")
    print(f"  - POST http://{args.host}:{args.port}/v1/audio/speech")
    print(f"  - GET  http://{args.host}:{args.port}/v1/audio/voices")
    print(f"  - GET  http://{args.host}:{args.port}/v1/models")
    print(f"  - GET  http://{args.host}:{args.port}/health")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
