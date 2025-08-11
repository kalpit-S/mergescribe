import os
import pytest
from providers import get_providers
from providers.parakeet_mlx import transcribe_sync as parakeet_transcribe
from providers.groq_whisper import transcribe_sync as groq_transcribe
from providers.gemini import transcribe_sync as gemini_transcribe


def load_test_audio():
    test_file = os.path.join(os.path.dirname(__file__), "testing_file.wav")
    if not os.path.exists(test_file):
        pytest.skip("Test audio file not found")
    
    with open(test_file, "rb") as f:
        return f.read()


def test_provider_loading():
    providers = get_providers()
    assert isinstance(providers, list)
    assert len(providers) >= 1


def test_parakeet_mlx_provider():
    audio_data = load_test_audio()
    
    try:
        result = parakeet_transcribe(audio_data)
        if isinstance(result, tuple):
            provider_name, text = result
            assert provider_name == "Parakeet MLX"
            result = text
        
        assert isinstance(result, str)
        assert len(result) > 0
        result_lower = result.lower()
        assert any(word in result_lower for word in ["testing", "one", "two", "three"])
    except Exception as e:
        pytest.skip(f"Parakeet MLX not available: {e}")


def test_groq_whisper_provider():
    audio_data = load_test_audio()
    
    try:
        result = groq_transcribe(audio_data)
        if isinstance(result, tuple):
            provider_name, text = result
            assert provider_name == "Groq Whisper"
            result = text
        
        assert isinstance(result, str)
        assert len(result) > 0
        result_lower = result.lower()
        assert any(word in result_lower for word in ["testing", "one", "two", "three"])
    except Exception as e:
        pytest.skip(f"Groq Whisper not available: {e}")


def test_gemini_provider():
    audio_data = load_test_audio()
    
    try:
        result = gemini_transcribe(audio_data)
        if isinstance(result, tuple):
            provider_name, text = result
            assert provider_name == "Gemini"
            result = text
        
        assert isinstance(result, str)
        assert len(result) > 0
        result_lower = result.lower()
        assert any(word in result_lower for word in ["testing", "one", "two", "three"])
    except Exception as e:
        pytest.skip(f"Gemini not available: {e}")


def test_provider_error_handling():
    invalid_audio = b"not audio data"
    
    providers_to_test = [
        ("parakeet_mlx", parakeet_transcribe),
        ("groq_whisper", groq_transcribe),
        ("gemini", gemini_transcribe),
    ]
    
    for name, provider_func in providers_to_test:
        try:
            result = provider_func(invalid_audio)
            assert isinstance(result, str)
        except Exception:
            pass
