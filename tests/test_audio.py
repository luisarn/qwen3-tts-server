# SPDX-License-Identifier: Apache-2.0
"""
Tests for TTS support.

Note: Some tests require mlx-audio to be installed.
"""

import numpy as np
import pytest


class TestTTSEngine:
    """Tests for Text-to-Speech engine."""

    def test_init_kokoro(self):
        """Test TTS engine initialization with Kokoro."""
        from qwen3_tts_server.audio.tts import TTSEngine

        engine = TTSEngine("mlx-community/Kokoro-82M-bf16")
        assert engine.model_name == "mlx-community/Kokoro-82M-bf16"
        assert engine._model_family == "kokoro"
        assert engine._loaded is False

    def test_init_chatterbox(self):
        """Test TTS engine initialization with Chatterbox."""
        from qwen3_tts_server.audio.tts import TTSEngine

        engine = TTSEngine("mlx-community/chatterbox-turbo-fp16")
        assert engine._model_family == "chatterbox"

    def test_init_vibevoice(self):
        """Test TTS engine initialization with VibeVoice."""
        from qwen3_tts_server.audio.tts import TTSEngine

        engine = TTSEngine("mlx-community/VibeVoice-Realtime-0.5B-4bit")
        assert engine._model_family == "vibevoice"

    def test_init_voxcpm(self):
        """Test TTS engine initialization with VoxCPM."""
        from qwen3_tts_server.audio.tts import TTSEngine

        engine = TTSEngine("mlx-community/VoxCPM1.5")
        assert engine._model_family == "voxcpm"

    def test_init_qwen3_tts(self):
        """Test TTS engine initialization with Qwen3-TTS."""
        from qwen3_tts_server.audio.tts import TTSEngine

        engine = TTSEngine("mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16")
        assert engine._model_family == "qwen3_tts"

    def test_available_voices(self):
        """Test voice lists."""
        from qwen3_tts_server.audio.tts import CHATTERBOX_VOICES, KOKORO_VOICES

        assert "af_heart" in KOKORO_VOICES
        assert len(KOKORO_VOICES) > 5
        assert "default" in CHATTERBOX_VOICES

    def test_get_voices(self):
        """Test get_voices method."""
        from qwen3_tts_server.audio.tts import TTSEngine

        kokoro = TTSEngine("mlx-community/Kokoro-82M-bf16")
        voices = kokoro.get_voices()
        assert "af_heart" in voices

        qwen3 = TTSEngine("mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16")
        qwen3_voices = qwen3.get_voices()
        assert "voice_clone" in qwen3_voices

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
            model_name="mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16",
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
