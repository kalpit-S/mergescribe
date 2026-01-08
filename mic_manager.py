"""Smart microphone manager that finds devices by name, not index.

Solves the problem of device indices changing when:
- Docking/undocking from a hub
- AirPods or other Bluetooth devices connect/disconnect
- Sleep/wake cycles
- USB devices are plugged/unplugged
"""

from dataclasses import dataclass
from typing import Optional

import sounddevice as sd

from config_manager import ConfigManager


@dataclass
class MicDevice:
    """Represents a microphone device."""
    index: int
    name: str
    max_input_channels: int
    default_samplerate: float
    is_default: bool = False


def list_input_devices() -> list[MicDevice]:
    """List all available input (microphone) devices."""
    try:
        devices = sd.query_devices()
    except Exception:
        return []
    
    if isinstance(devices, dict):
        devices = [devices]
    
    # Get default input device index
    try:
        default_input_idx = sd.default.device[0]
        if isinstance(default_input_idx, str):
            default_input_idx = int(default_input_idx)
    except Exception:
        default_input_idx = None
    
    result = []
    for idx, dev in enumerate(devices):
        try:
            max_channels = dev.get("max_input_channels", 0)
            if max_channels > 0:
                result.append(MicDevice(
                    index=idx,
                    name=dev.get("name", f"Device {idx}"),
                    max_input_channels=max_channels,
                    default_samplerate=dev.get("default_samplerate", 44100),
                    is_default=(idx == default_input_idx),
                ))
        except Exception:
            continue
    
    return result


def _normalize_name(name: str) -> str:
    """Normalize device name for comparison (lowercase, strip whitespace)."""
    return name.lower().strip()


def _extract_clean_name(stored_name: str) -> str:
    """Extract clean device name from stored format like '[4] HyperX SoloCast'."""
    name = stored_name.strip()
    # Remove index prefix like "[4] " if present
    if name.startswith("[") and "] " in name:
        name = name.split("] ", 1)[1]
    return name


def find_device_by_name(preferred_name: str, devices: Optional[list[MicDevice]] = None) -> Optional[MicDevice]:
    """
    Find a device by name, using fuzzy matching.
    
    Matching priority:
    1. Exact match (case-insensitive)
    2. Preferred name is contained in device name
    3. Device name is contained in preferred name
    
    Returns None if no match found.
    """
    if not preferred_name:
        return None
    
    if devices is None:
        devices = list_input_devices()
    
    if not devices:
        return None
    
    clean_preferred = _extract_clean_name(preferred_name)
    normalized_preferred = _normalize_name(clean_preferred)
    
    # First pass: exact match
    for dev in devices:
        if _normalize_name(dev.name) == normalized_preferred:
            return dev
    
    # Second pass: preferred name contained in device name
    for dev in devices:
        if normalized_preferred in _normalize_name(dev.name):
            return dev
    
    # Third pass: device name contained in preferred name
    for dev in devices:
        if _normalize_name(dev.name) in normalized_preferred:
            return dev
    
    return None


def get_preferred_device() -> tuple[Optional[int], Optional[str], bool]:
    """
    Get the preferred microphone device, searching by name.
    
    Returns:
        tuple of (device_index, device_name, was_found_by_name)
        - device_index: The current index of the device, or None for system default
        - device_name: The device name, or None if using system default
        - was_found_by_name: True if the device was found by name search
    """
    cfg = ConfigManager()
    stored_name = cfg.get_value("MIC_DEVICE_NAME")
    stored_index = cfg.get_value("MIC_DEVICE_INDEX")
    
    # If no preference set, use system default
    if not stored_name and stored_index is None:
        return None, None, False
    
    devices = list_input_devices()
    
    # Try to find by name first (this is the key improvement!)
    if stored_name:
        device = find_device_by_name(stored_name, devices)
        if device:
            # Check if index changed
            if stored_index is not None and device.index != stored_index:
                clean_name = _extract_clean_name(stored_name)
                print(f"🔄 Microphone '{clean_name}' moved from index {stored_index} to {device.index}")
                # Update stored index for next time
                _update_stored_index(device.index)
            return device.index, device.name, True
    
    # Name search failed - try stored index as fallback
    if stored_index is not None:
        for dev in devices:
            if dev.index == stored_index:
                # Found at stored index but name doesn't match - warn user
                clean_name = _extract_clean_name(stored_name) if stored_name else "Unknown"
                print(f"⚠️  Device at index {stored_index} is now '{dev.name}' (expected '{clean_name}')")
                return None, None, False  # Fall back to default instead of wrong device
    
    return None, None, False


def _update_stored_index(new_index: int) -> None:
    """Update the stored device index (when device moves to new index)."""
    try:
        cfg = ConfigManager()
        cfg.set_value("MIC_DEVICE_INDEX", new_index)
        cfg.save_config()
    except Exception:
        pass  # Non-critical


def get_device_display_label(device: MicDevice) -> str:
    """Get a display label for a device."""
    default_marker = " (default)" if device.is_default else ""
    return f"{device.name}{default_marker}"


def save_preferred_device(device: Optional[MicDevice]) -> None:
    """Save the user's preferred microphone device."""
    cfg = ConfigManager()
    
    if device is None:
        # Clear preference - use system default
        cfg.set_value("MIC_DEVICE_NAME", "")
        cfg.set_value("MIC_DEVICE_INDEX", None)
    else:
        # Store clean name (without index) and current index
        cfg.set_value("MIC_DEVICE_NAME", device.name)
        cfg.set_value("MIC_DEVICE_INDEX", device.index)
    
    cfg.save_config()


def check_preferred_device_status() -> tuple[str, bool]:
    """
    Check if the preferred device is currently available.
    
    Returns:
        tuple of (status_message, is_available)
    """
    cfg = ConfigManager()
    stored_name = cfg.get_value("MIC_DEVICE_NAME")
    
    if not stored_name:
        return "Using system default microphone", True
    
    clean_name = _extract_clean_name(stored_name)
    devices = list_input_devices()
    device = find_device_by_name(stored_name, devices)
    
    if device:
        return f"✓ '{clean_name}' ready", True
    else:
        available_names = [d.name for d in devices[:3]]  # Show first 3 available
        available_str = ", ".join(available_names) if available_names else "none found"
        return f"✗ '{clean_name}' not found. Available: {available_str}", False

