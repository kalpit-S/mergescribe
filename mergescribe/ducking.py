"""
Other audio turned down while dictating - when it plays out loud.

Speakers leak into the microphone and headphones don't, so while the key is
held the system output is lowered if it is going to speakers, and put back on
release. It is put back only if nobody changed it meanwhile: a user who
reaches for the volume mid-dictation keeps the setting they chose.

Core Audio is called through ctypes, not PyObjC, whose wrapper can't hand
back property data; this also works in any Python on macOS.
"""

from __future__ import annotations

import ctypes
import struct
from typing import Optional, Tuple

DUCK_TO = 0.25   # of the volume it was at

# Transports that play into the room. Built-in output depends on its data
# source: the MacBook's speakers, not its headphone jack. Bluetooth and USB
# are left alone - usually headphones or a headset, and a guess either way.
_ROOM_TRANSPORTS = {"hdmi", "dprt", "airp"}


def plays_out_loud(transport: str, source: str) -> bool:
    if transport == "bltn":
        return source == "ispk"
    return transport in _ROOM_TRANSPORTS


def _code(text: str) -> int:
    return struct.unpack(">I", text.encode("latin-1"))[0]


def _text(code: int) -> str:
    return struct.pack(">I", code & 0xFFFFFFFF).decode("latin-1")


class _Address(ctypes.Structure):
    _fields_ = [("selector", ctypes.c_uint32), ("scope", ctypes.c_uint32), ("element", ctypes.c_uint32)]


class CoreAudioOutput:
    """The default output device: what it is, and its volume."""

    def __init__(self):
        lib = ctypes.cdll.LoadLibrary("/System/Library/Frameworks/CoreAudio.framework/CoreAudio")
        lib.AudioObjectGetPropertyData.argtypes = [
            ctypes.c_uint32, ctypes.POINTER(_Address), ctypes.c_uint32, ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint32), ctypes.c_void_p]
        lib.AudioObjectSetPropertyData.argtypes = [
            ctypes.c_uint32, ctypes.POINTER(_Address), ctypes.c_uint32, ctypes.c_void_p,
            ctypes.c_uint32, ctypes.c_void_p]
        self._lib = lib

    def _get(self, device: int, selector: str, scope: str, ctype):
        out, size = ctype(), ctypes.c_uint32(ctypes.sizeof(ctype))
        address = _Address(_code(selector), _code(scope), 0)
        err = self._lib.AudioObjectGetPropertyData(device, ctypes.byref(address), 0, None,
                                                   ctypes.byref(size), ctypes.byref(out))
        return None if err else out.value

    def device(self) -> Optional[int]:
        return self._get(1, "dOut", "glob", ctypes.c_uint32)    # 1: the system object

    def kind(self, device: int) -> Tuple[str, str]:
        transport = self._get(device, "tran", "glob", ctypes.c_uint32)
        source = self._get(device, "ssrc", "outp", ctypes.c_uint32)
        return (_text(transport) if transport is not None else "",
                _text(source) if source is not None else "")

    def volume(self, device: int) -> Optional[float]:
        return self._get(device, "vmvc", "outp", ctypes.c_float)

    def set_volume(self, device: int, value: float) -> None:
        level = ctypes.c_float(max(0.0, min(1.0, value)))
        address = _Address(_code("vmvc"), _code("outp"), 0)
        self._lib.AudioObjectSetPropertyData(device, ctypes.byref(address), 0, None,
                                             ctypes.sizeof(level), ctypes.byref(level))


class Ducker:
    """duck() when recording starts, restore() when it stops. Never raises."""

    def __init__(self, output=None):
        if output is None:
            try:
                output = CoreAudioOutput()
            except OSError:
                output = None
        self._output = output
        self._saved: Optional[Tuple[int, float, float]] = None   # device, before, quiet

    def duck(self) -> None:
        if self._output is None or self._saved is not None:
            return
        try:
            device = self._output.device()
            if device is None or not plays_out_loud(*self._output.kind(device)):
                return
            before = self._output.volume(device)
            if not before:
                return
            self._output.set_volume(device, before * DUCK_TO)
            # Read back what the device settled on: some step their volume.
            quiet = self._output.volume(device)
            self._saved = (device, before, quiet if quiet is not None else before * DUCK_TO)
        except Exception as e:
            print(f"[Audio] Couldn't lower other audio: {e}")

    def restore(self) -> None:
        saved, self._saved = self._saved, None
        if saved is None or self._output is None:
            return
        device, before, quiet = saved
        try:
            now = self._output.volume(device)
            if now is not None and abs(now - quiet) < 0.01:
                self._output.set_volume(device, before)
        except Exception as e:
            print(f"[Audio] Couldn't restore other audio: {e}")
