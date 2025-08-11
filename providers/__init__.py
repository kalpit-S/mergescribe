from importlib import import_module

from config_manager import ConfigManager


def get_providers():
    providers = []
    enabled = ConfigManager().get_value("ENABLED_PROVIDERS") or []
    for name in enabled:
        try:
            module = import_module(f".{name}", package="providers")
            providers.append(module)
        except ImportError as e:
            print(f"Failed to load provider {name}: {e}")
    return providers
