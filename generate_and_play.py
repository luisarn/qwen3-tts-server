#!/usr/bin/env python3
"""
Generate speech with Qwen3-TTS and play it automatically.

Usage:
    python generate_and_play.py "Hello world!"
    python generate_and_play.py "Hello world!" --output myaudio.wav --no-play
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

def wait_for_server(url="http://localhost:8000/health", timeout=300):
    """Wait for server to be ready, with model download progress."""
    import urllib.request
    import urllib.error
    
    print("Waiting for server to start...")
    print("Note: First run downloads the Qwen3-TTS model (~1.2GB)")
    print()
    
    start_time = time.time()
    dots = 0
    
    while time.time() - start_time < timeout:
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.status == 200:
                    print("\r✓ Server is ready!                    ")
                    return True
        except:
            pass
        
        # Show progress
        dots = (dots + 1) % 4
        print(f"\r  Loading model{'.' * dots}{' ' * (3-dots)}", end='', flush=True)
        time.sleep(1)
    
    print("\r✗ Server failed to start                ")
    return False

def generate_speech(text, output_file, server_url="http://localhost:8000"):
    """Generate speech using the server API."""
    import urllib.request
    import urllib.parse
    
    # Build form data manually
    boundary = '----FormBoundary' + str(int(time.time()))
    
    body = (
        f'------{boundary}\r\n'
        f'Content-Disposition: form-data; name="input"\r\n\r\n'
        f'{text}\r\n'
        f'------{boundary}\r\n'
        f'Content-Disposition: form-data; name="model"\r\n\r\n'
        f'qwen3-tts\r\n'
        f'------{boundary}--\r\n'
    ).encode('utf-8')
    
    req = urllib.request.Request(
        f'{server_url}/v1/audio/speech',
        data=body,
        headers={
            'Content-Type': f'multipart/form-data; boundary=----{boundary}'
        },
        method='POST'
    )
    
    print(f"Generating speech...")
    print(f"  Text: \"{text}\"")
    
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            audio_data = resp.read()
            with open(output_file, 'wb') as f:
                f.write(audio_data)
            return len(audio_data)
    except urllib.error.HTTPError as e:
        print(f"✗ HTTP Error {e.code}: {e.read().decode()}")
        return 0
    except Exception as e:
        print(f"✗ Error: {e}")
        return 0

def play_audio(file_path):
    """Play audio file using macOS afplay."""
    import platform
    
    if platform.system() != 'Darwin':
        print(f"⚠️  Auto-play only supported on macOS")
        print(f"   To play manually: ffplay {file_path}")
        return False
    
    print(f"Playing {file_path}...")
    try:
        subprocess.run(['afplay', file_path], check=True)
        return True
    except subprocess.CalledProcessError:
        print("✗ Failed to play audio")
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Generate speech with Qwen3-TTS and play it",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate and play
  python generate_and_play.py "Hello world!"

  # Generate without playing
  python generate_and_play.py "Hello world!" --no-play

  # Custom output file
  python generate_and_play.py "Hello world!" --output hello.wav
        """
    )
    parser.add_argument('text', help='Text to synthesize')
    parser.add_argument('--output', '-o', default='/tmp/qwen3_tts_output.wav',
                        help='Output audio file (default: /tmp/qwen3_tts_output.wav)')
    parser.add_argument('--no-play', action='store_true',
                        help='Generate only, do not play')
    parser.add_argument('--server', default='http://localhost:8000',
                        help='Server URL (default: http://localhost:8000)')
    parser.add_argument('--port', type=int, default=8000,
                        help='Server port (default: 8000)')
    
    args = parser.parse_args()
    
    # Check if server is running
    import urllib.request
    server_running = False
    try:
        with urllib.request.urlopen(f'{args.server}/health', timeout=2) as resp:
            server_running = resp.status == 200
    except:
        pass
    
    server_process = None
    
    if not server_running:
        print("Starting Qwen3-TTS server...")
        print()
        
        # Start server
        server_process = subprocess.Popen(
            [sys.executable, '-m', 'qwen3_tts_server.server', 
             '--port', str(args.port)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        
        # Wait for server
        if not wait_for_server(f'{args.server}/health'):
            print("Failed to start server")
            if server_process:
                server_process.terminate()
            sys.exit(1)
    else:
        print("✓ Using existing server")
    
    # Generate speech
    print()
    audio_size = generate_speech(args.text, args.output, args.server)
    
    if audio_size > 0:
        print(f"✓ Generated: {args.output} ({audio_size} bytes)")
        
        if not args.no_play:
            print()
            play_audio(args.output)
    else:
        print("✗ Failed to generate audio")
    
    # Stop server if we started it
    if server_process:
        print("\nStopping server...")
        server_process.terminate()
        server_process.wait(timeout=10)
        print("✓ Server stopped")

if __name__ == '__main__':
    main()
