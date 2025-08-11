"""Audio preprocessing utilities."""
import time

import numpy as np

from config_manager import ConfigManager


def normalize_audio(audio_data):
    """Peak normalization to improve audio levels."""
    
    try:
        max_sample = np.max(np.abs(audio_data))
        if max_sample > 0.01:  # Only normalize if there's significant audio
            # Normalize to -3dB below full scale
            target_peak = 0.7079  # -3dB in linear scale
            normalized_audio = audio_data * (target_peak / max_sample)
            return normalized_audio
    except Exception as e:
        print(f"Normalization error: {e}")
    
    return audio_data

def reduce_noise(audio_data, sample_rate: int):
    """
    Apply simple noise reduction.
    Simplified implementation that's more reliable and faster.
    
    Args:
        audio_data: Numpy array of audio samples
        
    Returns:
        Noise-reduced audio data
    """
    try:
        if len(audio_data) < 1000:  # Need at least ~60ms of audio
            return audio_data
            
        rms = np.sqrt(np.mean(audio_data**2))
        if rms < 0.001:
            return audio_data
        
        # Apply a simple high-pass filter
        from scipy.signal import butter, filtfilt
        
        nyquist_freq = sample_rate / 2
        cutoff_freq = 80  # Hz - removes low-frequency noise
        normalized_cutoff = cutoff_freq / nyquist_freq
        
        b, a = butter(N=2, Wn=normalized_cutoff, btype='high')
        
        filtered_audio = filtfilt(b, a, audio_data)
        
        return filtered_audio
                
    except Exception as e:
        print(f"Noise reduction error: {e}")
        return audio_data  # Return original on any error

def compress_silence(audio_data, sample_rate: int, max_silence_sec: float = 2.0, threshold: float = 0.01):
    """Compress long internal silences by capping silence durations."""
    try:
        max_silence_samples = int(sample_rate * max_silence_sec)
        
        abs_audio = np.abs(audio_data)
        
        is_sound = abs_audio > threshold
        
        if np.all(is_sound) or np.all(~is_sound):
            return audio_data
        
        transitions = np.where(np.diff(is_sound.astype(int)))[0]
        if len(transitions) < 2:
            return audio_data
            
        if is_sound[0]:
            transitions = np.insert(transitions, 0, 0)
        if is_sound[-1]:
            transitions = np.append(transitions, len(audio_data) - 1)
            
        processed_segments = []
        for i in range(0, len(transitions), 2):
            if i + 1 >= len(transitions):
                processed_segments.append(audio_data[transitions[i]:]) 
                break
                
            start, end = transitions[i], transitions[i+1]
            segment = audio_data[start:end+1]
            
            if is_sound[start]:
                processed_segments.append(segment)
            else:
                silence_length = end - start + 1
                if silence_length > max_silence_samples:
                    compressed_length = max_silence_samples
                    silence_start = start + (silence_length - compressed_length) // 2
                    silence_end = silence_start + compressed_length
                    processed_segments.append(audio_data[silence_start:silence_end])
                else:
                    processed_segments.append(segment)
                    
        if processed_segments:
            return np.concatenate(processed_segments)
        return audio_data
        
    except Exception as e:
        print(f"Silence compression error: {e}")
        return audio_data

def preprocess_audio(audio_data):
    """Run the preprocessing pipeline."""
    if audio_data is None or len(audio_data) == 0:
        return audio_data
        
    try:
        cfg = ConfigManager()
        sr = cfg.get_value("SAMPLE_RATE") or 16000
        enable_norm = bool(cfg.get_value("ENABLE_AUDIO_NORMALIZATION"))
        enable_silence = bool(cfg.get_value("ENABLE_SILENCE_TRIMMING"))
        enable_noise = bool(cfg.get_value("ENABLE_NOISE_REDUCTION"))
        noise_min_sec = cfg.get_value("NOISE_REDUCTION_MIN_SECONDS") or 0
        debug_mode = bool(cfg.get_value("DEBUG_MODE"))

        time.time()
        recording_length_sec = len(audio_data) / sr
        processed_audio = np.copy(audio_data)

        # Step 1: Normalize audio levels
        if enable_norm:
            norm_start = time.time()
            processed_audio = normalize_audio(processed_audio)
            if debug_mode:
                print(f"|- Normalization: {time.time() - norm_start:.3f}s")
            
        # Step 2: Compress long internal silences
        if enable_silence:
            silence_start = time.time()
            original_length = len(processed_audio)
            processed_audio = compress_silence(processed_audio, sample_rate=sr)
            silence_time = time.time() - silence_start
            if debug_mode:
                print(f"|- Silence compression: {silence_time:.3f}s")
            
            if len(processed_audio) < original_length * 0.95:  # More than 5% reduction
                reduction = 100 * (1 - len(processed_audio) / original_length)
                print(f"   └─ Reduced audio length by {reduction:.1f}%")
        
        # Step 3: Conditional noise reduction
        if enable_noise and noise_min_sec > 0 and recording_length_sec > noise_min_sec:
            noise_start = time.time()
            processed_audio = reduce_noise(processed_audio, sample_rate=sr)
            if debug_mode:
                print(f"|- Noise reduction: {time.time() - noise_start:.3f}s")
        elif enable_noise and noise_min_sec > 0:
            if debug_mode:
                print(f"|- Noise reduction: Skipped (recording shorter than {noise_min_sec}s threshold)")
            
        return processed_audio
        
    except Exception as e:
        print(f"Audio preprocessing error: {e}")
        return audio_data
