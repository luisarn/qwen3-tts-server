#!/usr/bin/env python3
"""
Generate speech with local Qwen3-TTS model and play it automatically.

Usage:
    python generate_and_play_local.py "Hello world!"
"""

import sys
import time
from pathlib import Path

def main():
    text = sys.argv[1] if len(sys.argv) > 1 else "Hello! This is a test of Qwen3-TTS voice cloning."
    output_file = "/tmp/qwen3_tts_sample.wav"
    model_path = "~/Qwen3-TTS-12Hz-0.6B-Base-bf16"
    
    # Expand home directory
    model_path = Path(model_path).expanduser()
    
    print("=" * 60)
    print("Qwen3-TTS Voice Generation")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"Text: \"{text}\"")
    print()
    
    try:
        from qwen3_tts_server.audio import TTSEngine
        
        # Initialize with local model
        print("Loading model...")
        engine = TTSEngine(str(model_path))
        engine.load()
        print("✓ Model loaded!")
        print()
        
        # Reference audio - use bundled default
        ref_audio = Path(__file__).parent / "data" / "reference_audio" / "default_voice.wav"
        if not ref_audio.exists():
            print("✗ Reference audio not found!")
            sys.exit(1)
        
        print(f"Reference audio: {ref_audio}")
        print("Generating speech...")
        
        # Generate
        start = time.time()
        audio = engine.generate(text=text, ref_audio=str(ref_audio))
        elapsed = time.time() - start
        
        print(f"✓ Generated in {elapsed:.2f}s")
        print(f"  Duration: {audio.duration:.2f}s")
        print(f"  Sample rate: {audio.sample_rate}Hz")
        
        # Save
        engine.save(audio, output_file)
        print(f"✓ Saved: {output_file}")
        print()
        
        # Play
        import subprocess
        import platform
        
        if platform.system() == 'Darwin':
            print("Playing audio...")
            subprocess.run(['afplay', output_file])
            print("✓ Playback complete!")
        else:
            print(f"To play: ffplay {output_file}")
        
        # Cleanup
        engine.unload()
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        print("Make sure mlx-audio is installed: pip install mlx-audio")
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
