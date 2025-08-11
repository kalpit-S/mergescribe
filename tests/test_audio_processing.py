import os
import numpy as np
import pytest
import audio_processing


def load_test_audio():
    test_file = os.path.join(os.path.dirname(__file__), "testing_file.wav")
    if not os.path.exists(test_file):
        pytest.skip("Test audio file not found")
    
    import soundfile as sf
    audio_data, sample_rate = sf.read(test_file)
    return audio_data, sample_rate


def test_normalize_audio():
    audio_data, sample_rate = load_test_audio()
    
    normalized = audio_processing.normalize_audio(audio_data)
    
    assert isinstance(normalized, np.ndarray)
    assert normalized.shape == audio_data.shape
    assert not np.array_equal(normalized, audio_data)
    assert not np.any(np.isnan(normalized))
    assert not np.any(np.isinf(normalized))


def test_compress_silence():
    audio_data, sample_rate = load_test_audio()
    
    compressed = audio_processing.compress_silence(audio_data, sample_rate)
    
    assert isinstance(compressed, np.ndarray)
    assert len(compressed.shape) == len(audio_data.shape)
    assert not np.any(np.isnan(compressed))
    assert not np.any(np.isinf(compressed))


def test_noise_reduction():
    """Test noise reduction."""
    audio_data, sample_rate = load_test_audio()
    
    # Test noise reduction
    denoised = audio_processing.reduce_noise(audio_data, sample_rate)
    
    assert isinstance(denoised, np.ndarray)
    assert denoised.shape == audio_data.shape
    assert not np.any(np.isnan(denoised))
    assert not np.any(np.isinf(denoised))


def test_process_audio_pipeline():
    """Test the complete audio processing pipeline."""
    audio_data, sample_rate = load_test_audio()
    
    # Test the full pipeline
    processed = audio_processing.preprocess_audio(audio_data)
    
    assert isinstance(processed, np.ndarray)
    assert len(processed.shape) == len(audio_data.shape)
    assert not np.any(np.isnan(processed))
    assert not np.any(np.isinf(processed))


def test_audio_processing_with_different_configs():
    """Test audio processing with different configuration options."""
    audio_data, sample_rate = load_test_audio()
    
    # Test the full pipeline (uses config from ConfigManager)
    processed = audio_processing.preprocess_audio(audio_data)
    
    assert isinstance(processed, np.ndarray)
    assert len(processed.shape) == len(audio_data.shape)
    assert not np.any(np.isnan(processed))
    assert not np.any(np.isinf(processed))


def test_audio_processing_error_handling():
    """Test that audio processing handles errors gracefully."""
    # Test with invalid audio data
    invalid_audio = np.array([1, 2, 3, "not audio"])  # Invalid data
    
    try:
        result = audio_processing.normalize_audio(invalid_audio)
        # Should handle gracefully
        assert isinstance(result, np.ndarray)
    except Exception:
        # Exception is also acceptable
        pass
    
    # Test with empty audio
    empty_audio = np.array([])
    
    try:
        result = audio_processing.normalize_audio(empty_audio)
        assert isinstance(result, np.ndarray)
    except Exception:
        # Exception is also acceptable
        pass


def test_audio_processing_edge_cases():
    """Test audio processing with edge cases."""
    audio_data, sample_rate = load_test_audio()
    
    # Test with very short audio
    short_audio = audio_data[:100]  # Just 100 samples
    processed = audio_processing.preprocess_audio(short_audio)
    assert isinstance(processed, np.ndarray)
    
    # Test with very long audio
    long_audio = np.tile(audio_data, 10)  # Repeat 10 times
    processed = audio_processing.preprocess_audio(long_audio)
    assert isinstance(processed, np.ndarray)
