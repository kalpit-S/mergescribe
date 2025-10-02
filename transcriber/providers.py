import concurrent.futures
import time
from typing import Dict, Iterable, List, Tuple

from config_manager import ConfigManager
from providers import get_providers

from .audio import get_audio_as_bytes
from .state import is_debug

Transcription = Tuple[str, str]
ProviderTimings = Dict[str, float]


def collect_transcriptions(audio_data, state) -> Tuple[List[Transcription], ProviderTimings]:
    """Run transcription across all configured providers."""
    providers = get_providers()

    print("🎤 Transcribing...")

    sample_rate = state.current_sample_rate or (ConfigManager().get_value("SAMPLE_RATE") or 16000)
    audio_bytes = get_audio_as_bytes(audio_data, sample_rate=sample_rate)
    transcriptions: List[Transcription] = []
    provider_timings: ProviderTimings = {}

    recording_length_sec = len(audio_data) / sample_rate
    provider_timeout = max(3, min(120, recording_length_sec))

    if is_debug():
        print(
            f"[DEBUG] Provider timeout set to {provider_timeout}s "
            f"for {recording_length_sec:.1f}s recording"
        )

    max_workers = max(1, len(providers))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_provider = {
            executor.submit(provider.transcribe_sync, audio_bytes): provider.__name__
            for provider in providers
        }

        provider_start_times = {
            future: time.time()
            for future in future_to_provider
        }

        if state.turbo_mode:
            transcriptions, provider_timings = _handle_turbo_mode(
                future_to_provider, provider_start_times, provider_timeout
            )
        else:
            transcriptions, provider_timings = _handle_normal_mode(
                future_to_provider, provider_start_times, provider_timeout
            )

    return transcriptions, provider_timings


def compare_transcriptions(transcriptions: Iterable[Transcription]) -> None:
    """Print full transcriptions from all providers for debugging."""
    transcriptions = list(transcriptions)
    if len(transcriptions) < 2:
        return

    print("\n📊 Transcription Comparison:")
    print("─" * 60)

    for name, text in transcriptions:
        print(f"  {name}: {text}")

    print("─" * 60)


def _handle_turbo_mode(future_to_provider, provider_start_times, timeout):
    if is_debug():
        print("[DEBUG] Using turbo mode - waiting for first result")

    transcriptions = []
    provider_timings = {}

    done, not_done = concurrent.futures.wait(
        future_to_provider,
        return_when=concurrent.futures.FIRST_COMPLETED,
        timeout=timeout,
    )

    for future in done:
        try:
            provider_name = future_to_provider[future]
            provider_timings[provider_name] = time.time() - provider_start_times[future]

            name, text = future.result()
            if text:
                if is_debug():
                    print(f"\n[Raw Transcription from {name}]\n{text}\n")
                transcriptions.append((name, text))
                break
        except Exception as exc:  # noqa: BLE001 - log and continue gathering
            print(f"Provider error: {exc}")

    if not transcriptions and not_done:
        if is_debug():
            print("[DEBUG] First result was empty, waiting for remaining providers...")

        additional_done, still_not_done = concurrent.futures.wait(not_done, timeout=timeout)

        for future in additional_done:
            try:
                provider_name = future_to_provider[future]
                provider_timings[provider_name] = time.time() - provider_start_times[future]

                name, text = future.result()
                if text:
                    if is_debug():
                        print(f"\n[Raw Transcription from {name}]\n{text}\n")
                    transcriptions.append((name, text))
                    break
            except Exception as exc:  # noqa: BLE001 - log and continue
                print(f"Provider error: {exc}")

        not_done = still_not_done

    for future in not_done:
        future.cancel()

    return transcriptions, provider_timings


def _handle_normal_mode(future_to_provider, provider_start_times, timeout):
    transcriptions = []
    provider_timings = {}

    try:
        for future in concurrent.futures.as_completed(future_to_provider, timeout=timeout):
            provider_name = future_to_provider[future]
            provider_timings[provider_name] = time.time() - provider_start_times[future]

            try:
                name, text = future.result()
                if text:
                    if is_debug():
                        print(f"\n[Raw Transcription from {name}]\n{text}\n")
                    transcriptions.append((name, text))
            except Exception as exc:  # noqa: BLE001 - log and continue collecting
                print(f"{provider_name} error: {exc}")
    except concurrent.futures.TimeoutError:
        print("[WARNING] Provider transcription timed out. Processing partial results if any.")
        transcriptions, provider_timings = _handle_timeout(
            future_to_provider, provider_start_times
        )

    return transcriptions, provider_timings


def _handle_timeout(future_to_provider, provider_start_times):
    transcriptions = []
    provider_timings = {}

    for future in future_to_provider:
        if future.done() and not future.cancelled():
            provider_name = future_to_provider[future]
            provider_timings[provider_name] = time.time() - provider_start_times[future]
            try:
                name, text = future.result()
                if text:
                    if is_debug():
                        print(f"\n[Raw Transcription from {name}]\n{text}\n")
                    transcriptions.append((name, text))
            except Exception as exc:  # noqa: BLE001 - log and continue
                print(f"{provider_name} error after timeout: {exc}")

    for future in future_to_provider:
        if not future.done():
            future.cancel()

    if is_debug():
        print("[DEBUG] Remaining futures cancelled after timeout")

    return transcriptions, provider_timings
