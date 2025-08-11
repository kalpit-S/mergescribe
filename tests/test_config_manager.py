
from config_manager import ConfigManager


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


