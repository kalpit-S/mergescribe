import threading
import gc
from io import BytesIO

import mlx.core as mx
import numpy as np
import soundfile as sf
from parakeet_mlx import from_pretrained
from parakeet_mlx.audio import get_logmel

from config_manager import ConfigManager

_model = None
_parakeet_lock = threading.Lock()

def _get_model():
    """Get or initialize the Parakeet model (singleton)."""
    global _model
    if _model is None:
        _model = from_pretrained("mlx-community/parakeet-tdt-0.6b-v2")
        try:
            sr = ConfigManager().get_value("SAMPLE_RATE") or 16000
            dummy_audio = mx.array(np.zeros(sr // 10, dtype=np.float32))
            mel = get_logmel(dummy_audio, _model.preprocessor_config)
            _model.generate(mel)
        except Exception:
            pass
    return _model

def transcribe_sync(audio_bytes, language=None, prompt=None, temperature=0.0):
    """Transcribe audio using Parakeet MLX (synchronous)."""
    try:
        with _parakeet_lock:
            model = _get_model()
            
            audio_file = BytesIO(audio_bytes)
            audio_data, sr = sf.read(audio_file)
            
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)
            
            target_sr = model.preprocessor_config.sample_rate
            if sr != target_sr:
                from scipy import signal
                audio_data = signal.resample(audio_data, int(len(audio_data) * target_sr / sr))
            
            audio_data = audio_data.astype(np.float32)
            audio_mx = mx.array(audio_data)
            
            mel = get_logmel(audio_mx, model.preprocessor_config)
            
            alignments = model.generate(mel)
            
            transcribed_text = "".join([seg.text for seg in alignments])
            
            # Clean up memory to prevent accumulation
            del audio_mx
            del mel
            del alignments
            
            if hasattr(mx, "metal"):
                mx.metal.clear_cache()
            
            return ("Parakeet MLX", transcribed_text)
    except Exception as e:
        print(f"Parakeet MLX error: {e}")
        return ("Parakeet MLX", "")