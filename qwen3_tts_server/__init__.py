# SPDX-License-Identifier: Apache-2.0
"""
Qwen3-TTS Server for Apple Silicon

This package provides a TTS server with OpenAI-compatible API
using Apple's MLX framework and mlx-audio.

Features:
- OpenAI-compatible /v1/audio/speech API
- Voice cloning with Qwen3-TTS and Chatterbox
- Multiple TTS models: Kokoro, VibeVoice, VoxCPM, CSM, CosyVoice
- Default reference audio support
"""

__version__ = "0.2.6"

# Import TTSEngine for convenience
from .audio.tts import TTSEngine, clone_voice, generate_speech

__all__ = [
    "__version__",
    "TTSEngine",
    "clone_voice",
    "generate_speech",
]
