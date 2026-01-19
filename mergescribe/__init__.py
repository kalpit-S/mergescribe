"""
MergeScribe - Voice-to-text with multi-mic redundancy and LLM correction.

This package provides:
- Multi-microphone audio capture with pre-roll buffers
- Parallel transcription via multiple providers (local + cloud)
- Early consensus detection (skip slow providers)
- Silence-based chunking for low-latency long dictation
- LLM correction with app context awareness
- Two-strike correction caching

Main entry point: python -m mergescribe
"""

__version__ = "2.0.0"
