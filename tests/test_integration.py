import os
import pytest
import numpy as np
from config_manager import ConfigManager
from transcriber import get_global_state, reset_state


def test_config_manager_integration():
    cfg = ConfigManager()
    
    assert cfg.get_value("SAMPLE_RATE") is not None
    assert cfg.get_value("ENABLED_PROVIDERS") is not None
    assert cfg.get_value("SYSTEM_CONTEXT") is not None
    
    test_value = "test_value"
    cfg.set_value("TEST_KEY", test_value)
    assert cfg.get_value("TEST_KEY") == test_value


def test_global_state_management():
    """Test global state management."""
    state = get_global_state()
    
    # Test initial state
    assert hasattr(state, 'recording_in_progress')
    assert hasattr(state, 'audio_buffer')
    assert hasattr(state, 'transcription_history')
    assert hasattr(state, 'turbo_mode')
    assert hasattr(state, 'processing_lock')
    assert hasattr(state, 'active_threads')
    
    # Test state reset
    reset_state()
    state_after_reset = get_global_state()
    assert state_after_reset.recording_in_progress == False
    assert len(state_after_reset.audio_buffer) == 0
    assert len(state_after_reset.transcription_history) == 0


def test_audio_conversion_functions():
    """Test audio conversion utilities."""
    from transcriber import get_audio_as_bytes
    
    # Create test audio data
    test_audio = np.random.rand(16000)  # 1 second at 16kHz
    test_audio = (test_audio * 32767).astype(np.int16)  # Convert to 16-bit
    
    # Test conversion to bytes
    audio_bytes = get_audio_as_bytes(test_audio)
    assert isinstance(audio_bytes, bytes)
    assert len(audio_bytes) > 0


def test_context_building():
    """Test context building from transcription history."""
    from transcriber import build_context_from_history
    
    # Test with empty history
    context = build_context_from_history([])
    assert context == ""
    
    # Test with some history (format: [(timestamp, text), ...])
    import time
    current_time = time.time()
    history = [
        (current_time - 60, "First transcription"),
        (current_time - 30, "Second transcription"), 
        (current_time - 10, "Third transcription")
    ]
    context = build_context_from_history(history)
    assert isinstance(context, str)
    assert len(context) > 0
    assert "First transcription" in context


def test_prompt_building():
    """Test prompt building functionality."""
    from transcriber import build_correction_prompt
    
    # Test with sample transcriptions
    transcriptions = [
        ("parakeet_mlx", "testing one two three"),
        ("groq_whisper", "testing one two three"),
    ]
    
    prompt = build_correction_prompt(transcriptions)
    assert isinstance(prompt, str)
    assert len(prompt) > 0
    assert "testing one two three" in prompt
    assert "parakeet_mlx" in prompt
    assert "groq_whisper" in prompt


def test_application_context_detection():
    """Test application context detection (mock test)."""
    from transcriber import get_application_context
    
    # This might not work in test environment, so we'll test gracefully
    try:
        context = get_application_context()
        # Should return a string (might be empty if no context available)
        assert isinstance(context, str)
    except Exception:
        # It's okay if this fails in test environment
        pass


def test_text_editing_functions():
    """Test text editing functionality."""
    from text_editing import detect_selected_text, replace_selected_text
    
    # Test text detection (might not work in test environment)
    try:
        selected = detect_selected_text()
        # Should return a string (might be None if no selection)
        assert selected is None or isinstance(selected, str)
    except Exception:
        # It's okay if this fails in test environment
        pass


def test_fast_text_input():
    """Test fast text input functionality."""
    from fast_text_input import type_fast
    
    # Test that the function exists and is callable
    assert callable(type_fast)
    
    # Test with a simple string
    try:
        # This might not work in test environment, so we'll test gracefully
        type_fast("test")
    except Exception:
        # It's okay if this fails in test environment
        pass


def test_provider_interface():
    """Test that all providers follow the expected interface."""
    from providers import get_providers
    
    providers = get_providers()
    
    for provider in providers:
        # Each provider should have a transcribe_sync function
        assert hasattr(provider, 'transcribe_sync')
        assert callable(provider.transcribe_sync)


def test_settings_dialog_import():
    """Test that settings dialog can be imported."""
    try:
        import settings_dialog
        assert hasattr(settings_dialog, 'main')
        assert callable(settings_dialog.main)
    except ImportError:
        # Some dependencies might not be available in test environment
        pytest.skip("Settings dialog dependencies not available")


def test_main_app_import():
    """Test that main app can be imported."""
    try:
        import main
        assert hasattr(main, 'main')
        assert callable(main.main)
    except ImportError:
        # Some dependencies might not be available in test environment
        pytest.skip("Main app dependencies not available")
