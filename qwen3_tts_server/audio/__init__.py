# SPDX-License-Identifier: Apache-2.0
"""
Audio support for qwen3-tts-server using mlx-audio.

Provides Qwen3-TTS voice cloning.
"""

from .tts import TTSEngine, clone_voice, generate_speech

__all__ = [
    "TTSEngine",
    "generate_speech",
    "clone_voice",
]
