# SPDX-License-Identifier: Apache-2.0
"""
API models for vllm-mlx TTS server.

This module provides Pydantic models for the OpenAI-compatible TTS API.
"""

from .models import (
    AudioSpeechRequest,
    ModelInfo,
    ModelsResponse,
)

__all__ = [
    "AudioSpeechRequest",
    "ModelInfo",
    "ModelsResponse",
]
