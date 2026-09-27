"""
LLM correction for transcription results.

Sends transcription results to an LLM (via OpenRouter) for correction and
formatting. On failure the call is retried once; if it still fails the
caller falls back to the raw transcript.
"""

import json
import threading
import time
from typing import Callable, List, Optional, Tuple

import requests

from .types import TranscriptionResult, AppContext, ConfigSnapshot, LLMCorrectionResult


# Persistent session for connection reuse (saves ~70-100ms per request)
_openrouter_session = requests.Session()

OPENROUTER_MODEL_DEFAULT = "openai/gpt-6-luna"

# Abort a stream that stops producing content while the connection stays open.
# Observed once in ~15 calls: TTFT 0.71s then a 19s mid-stream stall, which
# dribbles text into the user's document with no way to tell it's hung.
_STREAM_STALL_SECONDS = 8.0

# Correction is a rewrite, and thinking only adds latency (a Gemini that thinks
# by default measured 11.9s against 1.6s with it off), so reasoning is asked to
# be off unless the user chose otherwise. Some models refuse "none" outright -
# Gemini 3.5 and newer, Grok 4.7, GLM 5.3: "reasoning is mandatory" - and those
# get "minimal" instead. Which ones is learned from the refusal, not kept in a
# list that goes stale with every release.
_NEEDS_SOME_REASONING: set = set()


def _config_str(config: ConfigSnapshot, key: str, default: str = "") -> str:
    """Read a string config field defensively for tests/partial config objects."""
    value = getattr(config, key, default)
    if isinstance(value, str):
        return value
    return default


def _config_str_list(config: ConfigSnapshot, key: str) -> List[str]:
    """Read a list of string config values defensively."""
    value = getattr(config, key, [])
    if isinstance(value, list):
        return [item for item in value if isinstance(item, str) and item.strip()]
    return []


def _config_bool(config: ConfigSnapshot, key: str, default: bool) -> bool:
    """Read a boolean config field defensively."""
    value = getattr(config, key, default)
    if isinstance(value, bool):
        return value
    return default


# Default system prompt for correction
DEFAULT_SYSTEM_CONTEXT = """Turn this speech-to-text transcript into the message the speaker would have typed if they had written it themselves.

Typed text drops the scaffolding of speech: filler sounds, filler words that carry no meaning ("like", "you know", "I mean", "kind of", "basically"), stutters, repeats, and false starts. When the speaker corrects themselves, keep only the version they settled on. When the speaker calls something off with "scratch that" or "never mind", drop it along with what it cancels; anything they say after still stands. Reply with exactly [nothing] only if nothing at all is left.

Everything else is theirs. Keep their words, slang, tone, and phrasing, and never add words they didn't say. Fix punctuation, grammar, and clear mishearings; when several transcriptions of the same audio are given, use them together to work out what was said. Recognizers often drop or swap short words like "not", "never" and "no" ("no reason" heard as "a reason"), so if any transcription has a negation the others lack, keep it: losing one reverses the meaning. Unfamiliar names are usually real products, people, or jargon, so keep them. When unsure, keep what was said.

Never use an em dash or en dash. Where the speaker breaks off, restarts or changes course, end the sentence there or use a comma, as a person typing would.

Return only the cleaned text, as one continuous line with no line breaks, markdown, or commentary."""


DEFAULT_EDITING_PROMPT = "You are a text editing assistant. Apply the user's requested change precisely and return only the edited text."


def _finish_without_streaming(shown: str, prompt: str, system_prompt: str,
                              config: ConfigSnapshot, on_delta: Optional[Callable[[str], None]]) -> str:
    """
    Recover a correction whose stream broke after typing began: ask again
    without streaming, and type only the rest if the new answer continues what
    is on screen. Otherwise raise, so the caller can hand over the whole text.
    """
    print("[LLM] Stream broke after typing began; finishing it without streaming")
    full, complete = _stream_openrouter(prompt, system_prompt, config)
    if complete and full.startswith(shown):
        if on_delta is not None and len(full) > len(shown):
            on_delta(full[len(shown):])
        return full
    raise CorrectionInterrupted(shown, full if complete else "")


def build_system_prompt(config: ConfigSnapshot, custom_instructions: str = "") -> str:
    """Assemble the system prompt: base + user preferences + learned vocabulary."""
    configured = _config_str(config, "system_prompt")
    system_prompt = configured if configured else DEFAULT_SYSTEM_CONTEXT

    if custom_instructions:
        system_prompt += f"\n\nUser preferences:\n{custom_instructions}"

    if _config_bool(config, "learn_vocabulary", True):
        from .vocabulary import learned_corrections, supplied_prompt, supplied_terms, vocabulary_prompt
        for evidence in (vocabulary_prompt(learned_corrections()), supplied_prompt(supplied_terms())):
            if evidence:
                system_prompt += f"\n\n{evidence}"

    return system_prompt


def correct_with_llm(
    results: List[TranscriptionResult],
    context: Optional[AppContext],
    config: ConfigSnapshot,
    on_delta: Optional[Callable[[str], None]] = None,
    on_metadata: Optional[Callable[[LLMCorrectionResult], None]] = None,
    custom_instructions: str = "",
    on_generation_metadata: Optional[Callable[[str, dict], None]] = None,
) -> str:
    """
    Call the correction LLM. Returns "" if the call fails after a retry;
    the caller is responsible for falling back to the raw transcript.

    Args:
        results: All transcription results from providers/mics
        context: Active application context (for prompt customization)
        config: Configuration snapshot
        on_delta: Optional callback for streaming tokens
        on_metadata: Optional callback to receive LLM result metadata
        custom_instructions: User's custom instructions
        on_generation_metadata: Called off-thread with (generation_id, usage)
            once OpenRouter reports cost/provider details
    """
    results = [r for r in results if r.text.strip()]
    if not results:
        return ""

    if not config.openrouter_api_key:
        print("[LLM] No OpenRouter API key configured")
        return ""

    prompt = _build_prompt(results, context)
    system_prompt = build_system_prompt(config, custom_instructions)

    total_words = max(len(r.text.split()) for r in results)
    est_tokens = (len(prompt) + len(system_prompt)) // 4
    model = _config_str(config, "openrouter_correction_model", OPENROUTER_MODEL_DEFAULT)

    start = time.perf_counter()
    first_token_at: List[float] = []

    shown: List[str] = []   # every token handed to on_delta, i.e. on screen

    def timing_delta(token: str) -> None:
        if not first_token_at:
            first_token_at.append(time.perf_counter())
        if on_delta is not None:
            shown.append(token)
            on_delta(token)

    result = ""
    for attempt in range(2):
        metadata: dict = {}
        result, complete = _stream_openrouter(
            prompt, system_prompt, config, timing_delta,
            metadata_out=metadata, on_generation_metadata=on_generation_metadata,
        )
        if complete and result:
            break
        if shown:
            # Part of it is already typed: streaming again would type it twice.
            result = _finish_without_streaming("".join(shown), prompt, system_prompt, config, on_delta)
            break
        result = ""
        if attempt == 0:
            print("[LLM] Retrying OpenRouter")

    if not result:
        print("[LLM] Correction failed")
        return ""

    elapsed = (time.perf_counter() - start) * 1000
    ttft_s = (first_token_at[0] - start) if first_token_at else elapsed / 1000
    # Say what actually served it, not what was asked for: ":nitro" and the
    # provider preferences mean the two can differ.
    served = metadata.get("resolved_model") or model
    provider = metadata.get("backend_provider")
    print(
        f"[LLM] {served}{f' @{provider}' if provider else ''}"
        f"{f' (asked {model})' if served != model else ''} | "
        f"in: {total_words} words, ~{est_tokens} tok prompt | "
        f"TTFT {ttft_s:.2f}s | total {elapsed/1000:.2f}s"
    )

    if on_metadata:
        on_metadata(LLMCorrectionResult(
            text=result,
            provider="openrouter",
            model=model,
            input_tokens_est=est_tokens,
            latency_ms=elapsed,
            streamed=on_delta is not None,
            generation_id=metadata.get("generation_id"),
            backend_provider=metadata.get("backend_provider"),
            resolved_model=metadata.get("resolved_model"),
            provider_order=metadata.get("provider_order", []),
            allow_fallbacks=metadata.get("allow_fallbacks"),
            reasoning_effort=metadata.get("reasoning_effort", ""),
            usage=metadata.get("usage", {}),
        ))

    return result


def _build_prompt(
    results: List[TranscriptionResult],
    context: Optional[AppContext],
) -> str:
    """
    Build the LLM prompt from transcription results.

    Earlier dictations are deliberately not included: the model regurgitated
    them instead of transcribing the current one - 14 of 42 captured
    corrections leaked, worst case 187 words of a previous dictation for a
    4-word utterance. A short input can't outweigh fluent nearby text.
    """

    def listing(group: List[TranscriptionResult]) -> List[str]:
        """One line per distinct reading; identical ones only cost tokens."""
        seen: set = set()
        lines = []
        for r in group:
            normalized = " ".join(r.text.lower().split())
            if normalized and normalized not in seen:
                seen.add(normalized)
                lines.append(f"[{r.provider}/{r.mic}]: {r.text}")
        return lines

    # A long dictation is transcribed in consecutive parts. Listing them flat
    # would read as alternatives of one piece of audio, and deduplicating
    # across them would drop a sentence the speaker really said twice.
    parts = sorted({r.chunk for r in results})
    if len(parts) > 1:
        blocks = [f"Part {n}:\n" + "\n".join(listing([r for r in results if r.chunk == part]))
                  for n, part in enumerate(parts, 1)]
        transcription_text = ("The audio came in consecutive parts; each lists what the recognizers "
                              "heard of that part.\n" + "\n".join(blocks))
    else:
        transcription_text = "\n".join(listing(results))

    context_parts = []

    if context:
        context_parts.append(f"Active application: {context.app_name}")
        if context.window_title:
            context_parts.append(f"Window: {context.window_title}")


    context_text = "\n".join(context_parts) if context_parts else ""

    parts = []
    if context_text:
        parts.append(context_text)
    parts.append(f"Transcriptions:\n{transcription_text}")

    return "\n\n".join(parts)


class CorrectionInterrupted(Exception):
    """
    The correction's stream broke after some of it had already been typed.

    shown is what reached the screen; complete is the whole correction if a
    second attempt produced one that doesn't continue what was shown, else "".
    """

    def __init__(self, shown: str, complete: str):
        super().__init__("correction stream interrupted after typing began")
        self.shown = shown
        self.complete = complete


def _call_openrouter(
    prompt: str,
    system_prompt: str,
    config: ConfigSnapshot,
    on_delta: Optional[Callable[[str], None]] = None,
    timeout: int = 15,
    metadata_out: Optional[dict] = None,
    on_generation_metadata: Optional[Callable[[str, dict], None]] = None,
) -> str:
    """The whole reply, or "" if it didn't arrive complete."""
    text, complete = _stream_openrouter(prompt, system_prompt, config, on_delta, timeout,
                                        metadata_out, on_generation_metadata)
    return text if complete else ""


def _stream_openrouter(
    prompt: str,
    system_prompt: str,
    config: ConfigSnapshot,
    on_delta: Optional[Callable[[str], None]] = None,
    timeout: int = 15,
    metadata_out: Optional[dict] = None,
    on_generation_metadata: Optional[Callable[[str, dict], None]] = None,
) -> Tuple[str, bool]:
    """
    Call OpenRouter's chat completions API with streaming.

    Returns (text, complete). complete is False when the stream broke,
    stalled or reported an error partway - text is then only what arrived.
    """
    if not config.openrouter_api_key:
        return "", False

    model = _config_str(config, "openrouter_correction_model", OPENROUTER_MODEL_DEFAULT)
    provider_order = _config_str_list(config, "openrouter_correction_provider_order")
    allow_fallbacks = _config_bool(config, "openrouter_correction_allow_fallbacks", True)
    # ":nitro" and friends are routing hints on the same underlying model.
    base_model = model.split(":", 1)[0]
    reasoning_effort = (_config_str(config, "openrouter_correction_reasoning_effort").strip().lower()
                        or "none")
    if reasoning_effort == "none" and base_model in _NEEDS_SOME_REASONING:
        reasoning_effort = "minimal"

    if metadata_out is not None:
        metadata_out.update({
            "provider_order": provider_order,
            "allow_fallbacks": allow_fallbacks if provider_order else None,
            "reasoning_effort": reasoning_effort,
        })

    headers = {
        "Authorization": f"Bearer {config.openrouter_api_key}",
        "Content-Type": "application/json",
    }

    data: dict = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.5,
        "max_tokens": 2000,
        "stream": True,
    }

    if provider_order:
        data["provider"] = {
            "order": provider_order,
            "allow_fallbacks": allow_fallbacks,
        }

    data["reasoning"] = {"effort": reasoning_effort}

    collected_chunks: List[str] = []
    generation_id: Optional[str] = None

    try:
        response = _openrouter_session.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=timeout,
            stream=True,
        )

        if response.status_code != 200:
            try:
                detail = response.text[:300]
            except Exception:
                detail = ""
            if response.status_code == 400 and reasoning_effort == "none" and "reason" in detail.lower():
                _NEEDS_SOME_REASONING.add(base_model)
                print(f"[LLM] {base_model} won't switch reasoning off; using minimal")
                return _stream_openrouter(prompt, system_prompt, config, on_delta, timeout,
                                          metadata_out, on_generation_metadata)
            print(f"[LLM] OpenRouter API error: {response.status_code} {detail[:120]}")
            return "", False

        last_content_at = time.time()
        complete = True
        for line in response.iter_lines():
            if time.time() - last_content_at > _STREAM_STALL_SECONDS:
                print(f"[LLM] Stream stalled >{_STREAM_STALL_SECONDS:.0f}s")
                complete = False
                break

            if not line:
                continue

            line_text = line.decode("utf-8").strip()

            if not line_text or line_text.startswith(":"):
                continue

            if not line_text.startswith("data: "):
                continue

            payload = line_text[6:]
            if payload == "[DONE]":
                break

            try:
                parsed = json.loads(payload)
            except json.JSONDecodeError:
                continue

            if parsed.get("id"):
                generation_id = parsed["id"]

            # OpenRouter names the model and provider it routed to on each
            # chunk. Without this the log could only repeat what we asked for.
            if metadata_out is not None:
                if parsed.get("model"):
                    metadata_out["resolved_model"] = parsed["model"]
                if parsed.get("provider"):
                    metadata_out["backend_provider"] = parsed["provider"]

            if "error" in parsed:
                print(f"[LLM] OpenRouter stream error: {parsed['error']}")
                complete = False
                break

            if parsed.get("usage") and metadata_out is not None:
                metadata_out["usage"] = parsed["usage"]

            try:
                choice = parsed.get("choices", [{}])[0]
                delta = choice.get("delta", {})
                content = delta.get("content")
                if content:
                    last_content_at = time.time()
                    collected_chunks.append(content)
                    if on_delta is not None:
                        on_delta(content)
            except Exception:
                continue

        if generation_id and metadata_out is not None:
            metadata_out["generation_id"] = generation_id
            if not metadata_out.get("backend_provider") and provider_order and not allow_fallbacks:
                metadata_out["backend_provider"] = provider_order[0]
            metadata_out.setdefault("resolved_model", model)
            # Usage/cost lives behind a second HTTP request. Fetching it inline
            # delayed session completion by up to ~4.5s after the text was
            # already typed, which showed up as a "busy" rejection on the next
            # hotkey press, so it now runs in the background.
            if on_generation_metadata is not None:
                _fetch_generation_metadata_async(
                    config.openrouter_api_key, generation_id, on_generation_metadata,
                )

        return "".join(collected_chunks), complete

    except Exception as e:
        print(f"[LLM] OpenRouter error: {e}")
        return "".join(collected_chunks), False


def _fetch_generation_metadata_async(
    api_key: str, generation_id: str, on_result: Callable[[str, dict], None]
) -> None:
    """Fetch usage/cost metadata in the background; never blocks the session."""
    def work() -> None:
        try:
            data = _fetch_openrouter_generation_metadata(api_key, generation_id)
            if data:
                on_result(generation_id, data)
        except Exception as e:
            print(f"[LLM] Generation metadata fetch failed: {e}")

    threading.Thread(target=work, daemon=True).start()


def _fetch_openrouter_generation_metadata(api_key: str, generation_id: str) -> dict:
    """Fetch OpenRouter usage/provider metadata for a completed generation."""
    for attempt in range(2):
        try:
            response = _openrouter_session.get(
                "https://openrouter.ai/api/v1/generation",
                headers={"Authorization": f"Bearer {api_key}"},
                params={"id": generation_id},
                timeout=2,
            )
            if response.status_code != 200:
                if attempt == 0:
                    time.sleep(0.25)
                    continue
                return {}

            payload = response.json().get("data", {})
            usage = {
                key: payload.get(key)
                for key in (
                    "tokens_prompt",
                    "tokens_completion",
                    "native_tokens_prompt",
                    "native_tokens_completion",
                    "native_tokens_cached",
                    "native_tokens_reasoning",
                    "num_media_prompt",
                    "num_media_completion",
                    "total_cost",
                    "upstream_inference_cost",
                    "latency",
                    "generation_time",
                    "finish_reason",
                    "native_finish_reason",
                    "router",
                    "is_byok",
                    "request_id",
                )
                if payload.get(key) is not None
            }
            return {
                "backend_provider": payload.get("provider_name"),
                "resolved_model": payload.get("model"),
                "usage": usage,
            }
        except Exception:
            if attempt == 0:
                time.sleep(0.25)
                continue
            return {}
    return {}


def edit_text_with_llm(
    selected_text: str,
    voice_command: str,
    config: ConfigSnapshot,
) -> str:
    """
    Apply a voice command to edit selected text via LLM.

    Returns the original text unchanged if the call fails.
    """
    prompt = (
        f"TASK: {voice_command}\n\n"
        f"ORIGINAL TEXT:\n{selected_text}\n\n"
        "INSTRUCTIONS: Apply the task to the original text above. Return ONLY the edited text, "
        "nothing else. No explanations, no formatting, no extra content."
    )

    system_prompt = config.editing_prompt if config.editing_prompt else DEFAULT_EDITING_PROMPT

    result = _call_openrouter(prompt, system_prompt, config)
    if not result:
        result = _call_openrouter(prompt, system_prompt, config)

    return result if result else selected_text
