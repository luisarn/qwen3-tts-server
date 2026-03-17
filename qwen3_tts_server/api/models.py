# SPDX-License-Identifier: Apache-2.0
"""
Pydantic models for OpenAI-compatible TTS API.
"""

import time
from typing import Optional

from pydantic import BaseModel, Field


# =============================================================================
# Models List
# =============================================================================


class ModelInfo(BaseModel):
    """Information about an available model."""

    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "qwen3-tts-server"
    description: Optional[str] = None


class ModelsResponse(BaseModel):
    """Response for listing models."""

    object: str = "list"
    data: list[ModelInfo]


# =============================================================================
# Audio (TTS)
# =============================================================================


class AudioSpeechRequest(BaseModel):
    """Request for text-to-speech."""

    model: str = "qwen3-tts"
    input: str
    voice: str = "voice_clone"
    speed: float = 1.0
    response_format: str = "wav"
