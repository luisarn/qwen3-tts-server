# SPDX-License-Identifier: Apache-2.0
"""
Tests for Qwen3-TTS support.

Note: Some tests require mlx-audio to be installed.
"""

import numpy as np
import pytest


class TestTTSEngine:
    """Tests for Text-to-Speech engine."""

    def test_init_default_model(self):
        """Test TTS engine initialization with default Qwen3-TTS model."""
        from qwen3_tts_server.audio.tts import TTSEngine, DEFAULT_TTS_MODEL

        engine = TTSEngine()
        assert engine.model_name == DEFAULT_TTS_MODEL
        assert engine._loaded is False

    def test_init_custom_model(self):
        """Test TTS engine initialization with custom model."""
        from qwen3_tts_server.audio.tts import TTSEngine

        engine = TTSEngine("mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16")
        assert engine.model_name == "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16"

    def test_audio_output(self):
        """Test AudioOutput dataclass."""
        from qwen3_tts_server.audio.tts import AudioOutput

        audio = np.zeros(24000, dtype=np.float32)
        output = AudioOutput(
            audio=audio,
            sample_rate=24000,
            duration=1.0,
        )
        assert output.sample_rate == 24000
        assert output.duration == 1.0
        assert len(output.audio) == 24000


class TestAPIModels:
    """Tests for TTS API models."""

    def test_speech_request(self):
        """Test AudioSpeechRequest model."""
        from qwen3_tts_server.api.models import AudioSpeechRequest

        req = AudioSpeechRequest(
            model="qwen3-tts",
            input="Hello world",
            voice="voice_clone",
            speed=1.2,
        )
        assert req.model == "qwen3-tts"
        assert req.input == "Hello world"
        assert req.voice == "voice_clone"
        assert req.speed == 1.2

    def test_model_info(self):
        """Test ModelInfo model."""
        from qwen3_tts_server.api.models import ModelInfo

        info = ModelInfo(id="qwen3-tts")
        assert info.id == "qwen3-tts"
        assert info.object == "model"

    def test_models_response(self):
        """Test ModelsResponse model."""
        from qwen3_tts_server.api.models import ModelInfo, ModelsResponse

        models = [ModelInfo(id="qwen3-tts")]
        resp = ModelsResponse(data=models)
        assert len(resp.data) == 1
        assert resp.object == "list"


class TestAudioImports:
    """Test that audio modules can be imported."""

    def test_import_audio_module(self):
        """Test importing main audio module."""
        from qwen3_tts_server.audio import TTSEngine, clone_voice, generate_speech

        assert TTSEngine is not None
        assert generate_speech is not None
        assert clone_voice is not None

    def test_import_api_models(self):
        """Test importing API models."""
        from qwen3_tts_server.api import AudioSpeechRequest, ModelInfo, ModelsResponse

        assert AudioSpeechRequest is not None
        assert ModelInfo is not None
        assert ModelsResponse is not None


# Integration tests (require mlx-audio installed)
@pytest.mark.skip(reason="Requires mlx-audio and models downloaded")
class TestAudioIntegration:
    """Integration tests for TTS (require models)."""

    def test_qwen3_tts(self):
        """Test Qwen3-TTS generation."""
        from qwen3_tts_server.audio import generate_speech

        audio = generate_speech(
            "Hello world",
            ref_audio="/path/to/reference.wav",
        )
        assert audio.audio is not None
        assert audio.sample_rate > 0

    def test_qwen3_tts_voice_clone(self):
        """Test Qwen3-TTS voice cloning."""
        from qwen3_tts_server.audio import clone_voice

        audio = clone_voice(
            "Hello world",
            ref_audio="/path/to/reference.wav",
        )
        assert audio.audio is not None
        assert audio.sample_rate > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
