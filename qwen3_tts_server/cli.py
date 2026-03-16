#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
CLI for Qwen3-TTS server.

Commands:
    qwen3-tts-server serve --port 8000 --default-ref-audio /path/to/voice.wav

Usage:
    qwen3-tts-server serve --port 8000
    qwen3-tts-server serve --port 8000 --default-ref-audio /path/to/voice.wav
"""

import argparse
import sys


def serve_command(args):
    """Start the OpenAI-compatible TTS server."""
    import os

    import uvicorn

    from . import server

    # Configure server globals
    server._api_key = args.api_key

    # Validate default reference audio if provided
    if args.default_ref_audio:
        if not os.path.exists(args.default_ref_audio):
            print(f"Error: Default reference audio file not found: {args.default_ref_audio}")
            sys.exit(1)
        server._default_ref_audio = args.default_ref_audio

    # Print configuration
    print("=" * 60)
    print("Qwen3-TTS Server Configuration")
    print("=" * 60)
    if args.api_key:
        print("  Authentication: ENABLED (API key required)")
    else:
        print("  Authentication: DISABLED - Use --api-key to enable")
    if args.default_ref_audio:
        print(f"  Default reference audio: {args.default_ref_audio}")
    print("=" * 60)

    print(f"\nStarting server at http://{args.host}:{args.port}")
    print("\nEndpoints:")
    print(f"  - POST http://{args.host}:{args.port}/v1/audio/speech")
    print(f"  - GET  http://{args.host}:{args.port}/v1/audio/voices")
    print(f"  - GET  http://{args.host}:{args.port}/v1/models")
    print(f"  - GET  http://{args.host}:{args.port}/health")

    uvicorn.run(server.app, host=args.host, port=args.port, log_level="info")


def main():
    parser = argparse.ArgumentParser(
        description="Qwen3-TTS: OpenAI-compatible TTS server for Apple Silicon",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  qwen3-tts-server serve --port 8000
  qwen3-tts-server serve --port 8000 --default-ref-audio /path/to/voice.wav
        """,
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Serve command
    serve_parser = subparsers.add_parser("serve", help="Start OpenAI-compatible TTS server")
    serve_parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="Host to bind (default: 0.0.0.0)"
    )
    serve_parser.add_argument(
        "--port", type=int, default=8000, help="Port to bind (default: 8000)"
    )
    serve_parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key for authentication (if not set, no auth required)",
    )
    serve_parser.add_argument(
        "--default-ref-audio",
        type=str,
        default=None,
        help="Default reference audio file for TTS voice cloning (used when no ref_audio is uploaded)",
    )

    args = parser.parse_args()

    if args.command == "serve":
        serve_command(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
