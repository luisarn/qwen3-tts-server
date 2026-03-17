# SPDX-License-Identifier: Apache-2.0
"""
Text-to-Speech (TTS) engine using mlx-audio.

Supports Qwen3-TTS voice cloning model.
"""

import io
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np

logger = logging.getLogger(__name__)

# Suppress the tokenizer regex warning for Qwen3-TTS models
# The mlx-audio library handles this internally
import warnings
warnings.filterwarnings(
    "ignore",
    message=".*incorrect regex pattern.*fix_mistral_regex.*",
    category=UserWarning,
)

# Default Qwen3-TTS model
DEFAULT_TTS_MODEL = "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16"


@dataclass
class AudioOutput:
    """Output from TTS generation."""

    audio: np.ndarray
    sample_rate: int
    duration: float


class TTSEngine:
    """
    Text-to-Speech engine for Qwen3-TTS voice cloning.

    Usage:
        engine = TTSEngine()
        engine.load()
        audio = engine.generate("Hello world!", ref_audio="reference_voice.wav")
        engine.save(audio, "output.wav")
    """

    def __init__(
        self,
        model_name: str = DEFAULT_TTS_MODEL,
    ):
        """
        Initialize TTS engine.

        Args:
            model_name: HuggingFace model name (default: Qwen3-TTS)
        """
        self.model_name = model_name
        self.model = None
        self._loaded = False

    def load(self) -> None:
        """Load the TTS model."""
        if self._loaded:
            return

        try:
            from mlx_audio.tts.generate import load_model

            self.model = load_model(self.model_name)
            self._loaded = True
            logger.info(f"TTS model loaded: {self.model_name}")
        except ImportError as e:
            logger.error(f"mlx-audio not installed: {e}")
            raise ImportError(
                "mlx-audio is required for TTS. Install with: pip install mlx-audio"
            ) from e

    def generate(
        self,
        text: str,
        ref_audio: Union[str, np.ndarray, None] = None,
        speed: float = 1.0,
    ) -> AudioOutput:
        """
        Generate speech from text.

        Args:
            text: Text to synthesize
            ref_audio: Reference audio for voice cloning (required)
            speed: Speech speed (0.5 to 2.0)

        Returns:
            AudioOutput with audio data and metadata

        Raises:
            RuntimeError: If ref_audio is not provided
        """
        if not self._loaded:
            self.load()

        if ref_audio is None:
            raise RuntimeError(
                "Qwen3-TTS requires reference audio. "
                "Provide ref_audio in request or start server with --default-ref-audio"
            )

        try:
            import mlx.core as mx

            audio_chunks = []
            sample_rate = 24000

            for result in self.model.generate(
                text=text,
                ref_audio=ref_audio,
                speed=speed,
            ):
                audio_data = result.audio
                if hasattr(result, "sample_rate"):
                    sample_rate = result.sample_rate

                # Convert mlx array to numpy
                if isinstance(audio_data, mx.array):
                    audio_np = np.array(audio_data.tolist(), dtype=np.float32)
                elif hasattr(audio_data, "tolist"):
                    audio_np = np.array(audio_data.tolist(), dtype=np.float32)
                else:
                    audio_np = np.array(audio_data, dtype=np.float32)

                audio_chunks.append(audio_np)

            if not audio_chunks:
                raise RuntimeError(
                    "No audio generated. This may be due to model load failure "
                    "or incompatible input parameters."
                )

            # Concatenate all chunks
            full_audio = (
                np.concatenate(audio_chunks)
                if len(audio_chunks) > 1
                else audio_chunks[0]
            )
            duration = len(full_audio) / sample_rate

            return AudioOutput(
                audio=full_audio,
                sample_rate=sample_rate,
                duration=duration,
            )
        except Exception as e:
            logger.error(f"TTS generation failed: {e}")
            raise

    def save(
        self,
        audio: AudioOutput,
        path: Union[str, Path],
    ) -> None:
        """
        Save audio to file.

        Args:
            audio: AudioOutput to save
            path: Output file path
        """
        try:
            from mlx_audio.tts import save_audio

            save_audio(audio.audio, str(path), sample_rate=audio.sample_rate)
            logger.info(f"Audio saved to {path}")
        except ImportError:
            # Fallback to scipy
            import scipy.io.wavfile as wav

            # Ensure audio is in correct format
            audio_int16 = (audio.audio * 32767).astype(np.int16)
            wav.write(str(path), audio.sample_rate, audio_int16)
            logger.info(f"Audio saved to {path} (scipy fallback)")

    def to_bytes(
        self,
        audio: AudioOutput,
        format: str = "wav",
    ) -> bytes:
        """
        Convert audio to bytes.

        Args:
            audio: AudioOutput to convert
            format: Output format (wav)

        Returns:
            Audio data as bytes
        """
        import scipy.io.wavfile as wav

        buffer = io.BytesIO()
        audio_int16 = (audio.audio * 32767).astype(np.int16)
        wav.write(buffer, audio.sample_rate, audio_int16)
        return buffer.getvalue()

    def unload(self) -> None:
        """Unload model to free memory."""
        self.model = None
        self._loaded = False
        logger.info("TTS model unloaded")


def generate_speech(
    text: str,
    model_name: str = DEFAULT_TTS_MODEL,
    ref_audio: Union[str, np.ndarray, None] = None,
    speed: float = 1.0,
) -> AudioOutput:
    """
    Convenience function to generate speech without managing engine.

    Args:
        text: Text to synthesize
        model_name: Model to use
        ref_audio: Reference audio for voice cloning (required for Qwen3-TTS)
        speed: Speech speed

    Returns:
        AudioOutput
    """
    engine = TTSEngine(model_name)
    engine.load()
    return engine.generate(text, ref_audio=ref_audio, speed=speed)


def clone_voice(
    text: str,
    ref_audio: Union[str, np.ndarray],
    model_name: str = DEFAULT_TTS_MODEL,
    speed: float = 1.0,
) -> AudioOutput:
    """
    Convenience function for voice cloning TTS.

    Args:
        text: Text to synthesize
        ref_audio: Reference audio file path or numpy array to clone voice from
        model_name: Voice cloning model to use (default: Qwen3-TTS)
        speed: Speech speed

    Returns:
        AudioOutput with cloned voice

    Example:
        >>> audio = clone_voice("Hello world!", "my_voice.wav")
        >>> engine = TTSEngine()
        >>> engine.save(audio, "output.wav")
    """
    engine = TTSEngine(model_name)
    engine.load()
    return engine.generate(text, ref_audio=ref_audio, speed=speed)
