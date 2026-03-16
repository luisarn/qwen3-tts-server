# SPDX-License-Identifier: Apache-2.0
"""
Audio support for qwen3-tts-server using mlx-audio.

Provides:
- TTS (Text-to-Speech): Qwen3-TTS with voice cloning support
- Kokoro: Fast, lightweight TTS with multiple voices
- Chatterbox: Expressive multilingual TTS with voice cloning
"""

from .tts import TTSEngine, clone_voice, generate_speech

__all__ = [
    # TTS
    "TTSEngine",
    "generate_speech",
    "clone_voice",
]
