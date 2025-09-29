
import json
import time

from config_manager import ConfigManager
import config_manager


def make_cfg(tmp_path):
    settings = tmp_path / "settings.json"
    env = tmp_path / ".env"
    cfg = ConfigManager(
        config_file_path=str(tmp_path / "no_config.py"),
        env_file_path=str(env),
        settings_file_path=str(settings),
    )
    return cfg


def test_defaults_present(tmp_path):
    cfg = make_cfg(tmp_path)
    sr = cfg.get_value("SAMPLE_RATE")
    assert isinstance(sr, int) and sr > 0
    assert isinstance(cfg.get_value("ENABLED_PROVIDERS"), list)
    assert isinstance(cfg.get_value("SYSTEM_CONTEXT"), str)


def test_validation_coercion(tmp_path):
    cfg = make_cfg(tmp_path)
    cfg.settings_values = {
        "SAMPLE_RATE": "44100",
        "ENABLE_CONTEXT": "true",
        "CHANNELS": "2",
        "APPLICATION_CONTEXT_TIMEOUT": "3",
        "ENABLED_PROVIDERS": ["parakeet_mlx", "invalid", 123],
    }
    # Save then reload (triggers validation) in temp file
    cfg.save_settings_file()
    cfg.load_settings_file()

    assert cfg.get_value("SAMPLE_RATE") == 44100
    assert cfg.get_value("ENABLE_CONTEXT") is True
    assert cfg.get_value("CHANNELS") == 2
    assert cfg.get_value("APPLICATION_CONTEXT_TIMEOUT") == 3
    assert cfg.get_value("ENABLED_PROVIDERS") == ["parakeet_mlx"]


def test_hot_reload(tmp_path):
    """Test that ConfigManager singleton automatically reloads when files change."""
    # Reset singleton for clean test
    config_manager._config_instance = None
    
    settings = tmp_path / "settings.json"
    env = tmp_path / ".env"
    
    # Create initial config with DEBUG_MODE=False
    with open(settings, 'w') as f:
        json.dump({"DEBUG_MODE": False, "SAMPLE_RATE": 16000}, f)
    
    # Create first instance
    cfg1 = ConfigManager(
        config_file_path=str(tmp_path / "no_config.py"),
        env_file_path=str(env),
        settings_file_path=str(settings),
    )
    
    assert cfg1.get_value("DEBUG_MODE") is False
    assert cfg1.get_value("SAMPLE_RATE") == 16000
    
    # Modify the settings file (with a small delay to ensure mtime changes)
    time.sleep(0.01)
    with open(settings, 'w') as f:
        json.dump({"DEBUG_MODE": True, "SAMPLE_RATE": 48000}, f)
    
    # Create second instance - should be same object but with reloaded values
    cfg2 = ConfigManager(
        config_file_path=str(tmp_path / "no_config.py"),
        env_file_path=str(env),
        settings_file_path=str(settings),
    )
    
    # Verify it's the same singleton instance
    assert cfg1 is cfg2, "ConfigManager should be a singleton"
    
    # Verify values were reloaded
    assert cfg2.get_value("DEBUG_MODE") is True, "DEBUG_MODE should be reloaded to True"
    assert cfg2.get_value("SAMPLE_RATE") == 48000, "SAMPLE_RATE should be reloaded to 48000"
    
    # Verify the first instance also sees the new values (since they're the same object)
    assert cfg1.get_value("DEBUG_MODE") is True
    assert cfg1.get_value("SAMPLE_RATE") == 48000
    
    # Cleanup singleton for other tests
    config_manager._config_instance = None


