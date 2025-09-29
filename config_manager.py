"""
Configuration Manager for MergeScribe

Persistence model:
- Secrets (API keys, OpenRouter site headers) are stored in .env
- Mutable app settings are stored in settings.json
- Static defaults live in config.py and are used as a fallback

Read precedence: environment > settings.json > config.py defaults
"""
import ast
import json
import os
from typing import Any, Dict


# Singleton instance
_config_instance = None


class ConfigManager:
    """Manages configuration settings for MergeScribe (Singleton pattern with auto-reload)"""
    
    def __new__(cls, *args, **kwargs):
        global _config_instance
        if _config_instance is None:
            _config_instance = super().__new__(cls)
        return _config_instance
    
    def __init__(self, config_file_path: str = "config.py", env_file_path: str = ".env", settings_file_path: str = "settings.json"):
        # Only initialize once
        if hasattr(self, '_initialized'):
            # Check if files have been modified and reload if needed
            self._check_and_reload()
            return
        self._initialized = True
        
        self.config_file_path = config_file_path
        self.env_file_path = env_file_path
        self.settings_file_path = settings_file_path
        self.config_values = {}
        self.env_values = {}
        self.settings_values = {}
        self._file_mtimes = {}
        self.load_config()
        self._update_mtimes()
    
    def _get_mtime(self, filepath: str) -> float:
        """Get modification time of a file, or 0 if file doesn't exist."""
        try:
            return os.path.getmtime(filepath)
        except (OSError, FileNotFoundError):
            return 0.0
    
    def _update_mtimes(self):
        """Update stored modification times for all config files."""
        self._file_mtimes = {
            'config': self._get_mtime(self.config_file_path),
            'env': self._get_mtime(self.env_file_path),
            'settings': self._get_mtime(self.settings_file_path),
        }
    
    def _check_and_reload(self):
        """Check if any config files have been modified and reload if needed."""
        current_mtimes = {
            'config': self._get_mtime(self.config_file_path),
            'env': self._get_mtime(self.env_file_path),
            'settings': self._get_mtime(self.settings_file_path),
        }
        
        # Check if any files have been modified
        if current_mtimes != self._file_mtimes:
            self.load_config()
            self._file_mtimes = current_mtimes
    
    def load_config(self):
        """Load configuration from settings.json, config.py and .env files."""
        self.load_config_file()
        self.load_settings_file()
        self.load_env_file()
    
    def load_config_file(self):
        """Load values from config.py file"""
        if not os.path.exists(self.config_file_path):
            return
        
        with open(self.config_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        tree = ast.parse(content)
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        var_name = target.id
                        try:
                            if isinstance(node.value, ast.Constant):
                                value = node.value.value
                            elif isinstance(node.value, ast.List):
                                value = [item.value if isinstance(item, ast.Constant) else str(item) 
                                        for item in node.value.elts]
                            elif isinstance(node.value, ast.Call):
                                if (isinstance(node.value.func, ast.Attribute) and 
                                    isinstance(node.value.func.value, ast.Name) and
                                    node.value.func.value.id == 'os' and 
                                    node.value.func.attr == 'getenv'):
                                    if node.value.args and isinstance(node.value.args[0], ast.Constant):
                                        env_var = node.value.args[0].value
                                        default_val = None
                                        if len(node.value.args) > 1 and isinstance(node.value.args[1], ast.Constant):
                                            default_val = node.value.args[1].value
                                        value = os.getenv(env_var, default_val)
                                    else:
                                        value = None
                                else:
                                    value = None
                            else:
                                value = None
                            
                            self.config_values[var_name] = value
                        except Exception:
                            self.config_values[var_name] = None
    
    def load_env_file(self):
        """Load values from .env file"""
        if not os.path.exists(self.env_file_path):
            return
        
        with open(self.env_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    self.env_values[key.strip()] = value.strip().strip('"\'')

    def load_settings_file(self):
        """Load values from settings.json file (non-secret settings)."""
        try:
            if os.path.exists(self.settings_file_path):
                with open(self.settings_file_path, 'r', encoding='utf-8') as f:
                    raw = json.load(f)
                    self.settings_values = self._validate_settings(raw)
            else:
                self.settings_values = {}
        except Exception:
            self.settings_values = {}
    
    def get_value(self, key: str) -> Any:
        """Get a configuration value"""
        # First check if it's an environment variable
        if key.endswith('_API_KEY') or key.startswith('OPENROUTER_SITE'):
            return self.env_values.get(key) or os.getenv(key)
        
        # Then check settings.json values
        if key in self.settings_values:
            return self.settings_values.get(key)

        # Then check values loaded from config.py if present
        if key in self.config_values:
            return self.config_values.get(key)

        # Finally fall back to built-in defaults
        return self.get_defaults().get(key)
    
    def set_value(self, key: str, value: Any):
        """Set a configuration value"""
        # API keys and site info go to .env file
        if key.endswith('_API_KEY') or key.startswith('OPENROUTER_SITE'):
            self.env_values[key] = str(value) if value is not None else ""
        else:
            # Persist non-secret settings to settings.json
            self.settings_values[key] = value
    
    def save_config(self):
        """Save configuration to files (settings.json and .env)."""
        self.save_settings_file()
        self.save_env_file()
        # Update modification times after saving to prevent immediate reload
        self._update_mtimes()
    
    def save_env_file(self):
        """Save environment variables to .env file"""
        env_lines = []
        
        existing_lines = []
        if os.path.exists(self.env_file_path):
            with open(self.env_file_path, 'r', encoding='utf-8') as f:
                existing_lines = f.readlines()
        
        for line in existing_lines:
            line_stripped = line.strip()
            if not line_stripped or line_stripped.startswith('#') or '=' not in line_stripped:
                env_lines.append(line.rstrip())
            else:
                key = line_stripped.split('=', 1)[0].strip()
                if key not in self.env_values:
                    env_lines.append(line.rstrip())
        
        for key, value in self.env_values.items():
            if value:  # Only add non-empty values
                env_lines.append(f"{key}={value}")
        
        os.makedirs(os.path.dirname(os.path.abspath(self.env_file_path)), exist_ok=True)
        
        with open(self.env_file_path, 'w', encoding='utf-8') as f:
            for line in env_lines:
                f.write(line + '\n')
    
    def save_settings_file(self):
        """Save non-secret settings to settings.json"""
        try:
            os.makedirs(os.path.dirname(os.path.abspath(self.settings_file_path)), exist_ok=True)
            with open(self.settings_file_path, 'w', encoding='utf-8') as f:
                json.dump(self._validate_settings(self.settings_values), f, indent=2, ensure_ascii=False)
        except Exception:
            pass
    
    def get_defaults(self) -> Dict[str, Any]:
        """Get default configuration values"""
        return {
            'TRIGGER_KEY': 'alt_r',
            'ENABLED_PROVIDERS': ['parakeet_mlx', 'groq_whisper'],
            'WHISPER_MODEL': 'whisper-large-v3',
            'WHISPER_LANGUAGE': 'en',
            'GEMINI_MODEL': 'google/gemini-2.5-flash',
            'OPENROUTER_MODEL': 'google/gemini-2.5-flash',
            'TEXT_EDITING_MODEL': 'google/gemini-2.5-flash',
            'SAMPLE_RATE': 16000,
            'CHANNELS': 1,
            'MIC_DEVICE_NAME': '',
            'MIC_DEVICE_INDEX': None,
            'ENABLE_AUDIO_NORMALIZATION': True,
            'ENABLE_SILENCE_TRIMMING': True,
            'ENABLE_NOISE_REDUCTION': True,
            'NOISE_REDUCTION_MIN_SECONDS': 0,
            'CONTEXT_HISTORY_COUNT': 3,
            'CONTEXT_MAX_AGE_SECONDS': 300,
            'ENABLE_CONTEXT': True,
            'ENABLE_APPLICATION_CONTEXT': True,
            'APPLICATION_CONTEXT_TIMEOUT': 2,
            'DEBUG_MODE': False,
            'GEMINI_PROMPT': 'Transcribe this speech with maximum clarity and technical accuracy.',
            'OPENROUTER_SITE_NAME': 'MergeScribe',
            'OPENROUTER_SITE_URL': '',
            'SYSTEM_CONTEXT': (
                """
You are a transcription assistant that cleans up speech-to-text output while preserving the speaker's authentic voice and exact meaning. You may receive context from recent transcriptions to maintain continuity, and you may receive multiple transcriptions of the same recording from different providers.

You may also receive information about the current application context (what app/website the user is currently using) to help you understand the vocabulary, terminology, and communication style that would be most appropriate for that context.

Clean up:
- Remove filler words ("um", "uh", "like", "you know", "so", "well", "right", "okay")
- Fix obvious transcription errors and typos (e.g. "lead code" → "leetcode", "gym" → "Jim")
- Handle self-corrections: ALWAYS use the correction, not the original mistake
  Examples: "Tuesday, no wait, Friday" → "Friday"
            "Send it to John, I mean Jane" → "Send it to Jane"
            "The budget is 50 thousand, wait no, 15 thousand" → "The budget is 15 thousand"
- Fix grammar and add proper punctuation

When multiple transcriptions are provided, compare and choose the most accurate parts from each.

Preserve:
- The speaker's meaning and intent
- Natural speaking style, slang, and strong language
- All substantive content

Meta-commands (not to transcribe, but to follow):
- "scratch that", "never mind", "forget what I said" → remove the previous content
- "make that sound better", "fix the grammar" → improve the previous content
- Apply commands to the most recent complete thought or sentence

Return only the cleaned transcription text, applying any requested transformations.

Reasoning: low
                """.strip()
            ),
            'TEXT_EDITING_CONTEXT': (
                """
You are a helpful text editing assistant. You will receive:
1. A voice command describing what to do
2. The original text to edit

Follow the voice command exactly and return only the edited text. Do not add explanations, comments, or formatting unless specifically requested.
                """.strip()
            ),
        }
    
    def reset_to_defaults(self):
        """Reset configuration to default values"""
        defaults = self.get_defaults()
        for key, value in defaults.items():
            self.set_value(key, value)

    def _validate_settings(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Light validation and coercion for settings.json content.
        Coerces known types, clamps numeric ranges, and drops obviously wrong types.
        """
        defaults = self.get_defaults()
        validated: Dict[str, Any] = {}

        def as_bool(val: Any) -> bool:
            if isinstance(val, bool):
                return val
            if isinstance(val, str):
                return val.lower() in {"1", "true", "yes", "on"}
            return bool(val)

        def as_int(val: Any, minimum: int | None = None, maximum: int | None = None) -> int:
            try:
                num = int(val)
            except Exception:
                num = 0
            if minimum is not None:
                num = max(minimum, num)
            if maximum is not None:
                num = min(maximum, num)
            return num

        known_bool = {
            'ENABLE_AUDIO_NORMALIZATION', 'ENABLE_SILENCE_TRIMMING', 'ENABLE_NOISE_REDUCTION',
            'ENABLE_CONTEXT', 'ENABLE_APPLICATION_CONTEXT', 'DEBUG_MODE'
        }
        known_int = {
            'SAMPLE_RATE', 'CHANNELS', 'NOISE_REDUCTION_MIN_SECONDS',
            'CONTEXT_HISTORY_COUNT', 'CONTEXT_MAX_AGE_SECONDS', 'APPLICATION_CONTEXT_TIMEOUT'
        }

        for key, value in data.items():
            if key in known_bool:
                validated[key] = as_bool(value)
            elif key in known_int:
                # apply simple ranges
                if key == 'SAMPLE_RATE':
                    validated[key] = as_int(value, 8000, 192000)
                elif key == 'CHANNELS':
                    validated[key] = as_int(value, 1, 8)
                elif key in {'CONTEXT_HISTORY_COUNT'}:
                    validated[key] = as_int(value, 0, 20)
                elif key in {'APPLICATION_CONTEXT_TIMEOUT'}:
                    validated[key] = as_int(value, 0, 30)
                else:
                    validated[key] = as_int(value, 0, None)
            elif key == 'ENABLED_PROVIDERS':
                if isinstance(value, list):
                    allowed = { 'parakeet_mlx', 'groq_whisper', 'gemini' }
                    validated[key] = [v for v in value if isinstance(v, str) and v in allowed]
                else:
                    validated[key] = defaults.get(key)
            elif key in {'MIC_DEVICE_INDEX'}:
                try:
                    validated[key] = None if value is None else int(value)
                except Exception:
                    validated[key] = None
            else:
                validated[key] = value

        # Fill any missing keys with defaults
        for k, v in defaults.items():
            validated.setdefault(k, v)
        return validated


def main():
    """Test the config manager"""
    config_manager = ConfigManager()
    
    print("Current config values:")
    for key, value in config_manager.config_values.items():
        print(f"  {key}: {value}")
    
    print("\nCurrent env values:")
    for key, value in config_manager.env_values.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()