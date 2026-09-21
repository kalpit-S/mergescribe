"""
LLM correction for transcription results.

Sends transcription results to an LLM (via OpenRouter) for correction and
formatting. On failure the call is retried once; if it still fails the
caller falls back to the raw transcript.
"""

import json
import threading
import time
from typing import Callable, List, Optional

import requests

from .types import TranscriptionResult, AppContext, ConfigSnapshot, LLMCorrectionResult


# Persistent session for connection reuse (saves ~70-100ms per request)
_openrouter_session = requests.Session()

OPENROUTER_MODEL_DEFAULT = "google/gemini-3.1-flash-lite"

# Abort a stream that stops producing content while the connection stays open.
# Observed once in ~15 calls: TTFT 0.71s then a 19s mid-stream stall, which
# dribbles text into the user's document with no way to tell it's hung.
_STREAM_STALL_SECONDS = 8.0

# Correction models whose reasoning/thinking should be disabled outright.
# Gemini 3.5 and newer (3.5/3.6/3.7) reject reasoning:none with a 400
# ("Reasoning is mandatory for this endpoint") and default to a large thinking
# budget if the field is omitted — measured 11.9s vs 1.6s on 3.7-flash. Those
# need openrouter_correction_reasoning_effort = "minimal" in settings instead.
OPENROUTER_NO_REASONING_MODELS = {
    "x-ai/grok-4.3",
    "google/gemini-3.1-flash-lite",
    "google/gemini-3.1-flash-lite-preview",
    "openai/gpt-5.4-mini",
    "openai/gpt-5.6-luna",
    "openai/gpt-5.6-terra",
    "z-ai/glm-5.2",
    "anthropic/claude-opus-4.8-fast",
    "minimax/minimax-m2.7:nitro",
}


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

Return only the cleaned text, as one continuous line with no line breaks, em dashes, markdown, or commentary."""


# Appended to the system prompt when an on-screen field inventory is provided
ROUTING_PROTOCOL = """

Output routing: an inventory of on-screen text input fields is provided.
Decide where this dictation belongs based on its content and each field's app, window, label, and contents.
Your reply MUST start with a first line of exactly "TARGET: <field-id>" (e.g. "TARGET: f3"), or "TARGET: focused" for the field the user is currently in ("Active application" above tells you where that is).
From the second line onward, output only the corrected transcript.
If the user explicitly says where to put it ("put this in Slack", "send to the terminal"), obey that — and don't transcribe the routing instruction itself.
Otherwise prefer "TARGET: focused" unless the content clearly belongs in a specific other field.
"focused" is always valid even when the focused window has no entry in the inventory — if the dictation plausibly continues whatever the Active application/Window shows, stay focused rather than moving it somewhere merely plausible.
The entry marked FOCUSED is what the user is looking at; entries marked as background windows are probably hidden behind it, so text sent there lands somewhere the user cannot see and has to be undone. Treat those as a last resort: pick one only when the user names it, or when the content plainly cannot belong in the focused field. Topical similarity alone is not enough."""


DEFAULT_EDITING_PROMPT = "You are a text editing assistant. Apply the user's requested change precisely and return only the edited text."


def build_system_prompt(
    config: ConfigSnapshot,
    field_targets: Optional[list] = None,
    custom_instructions: str = "",
) -> str:
    """Assemble the system prompt: base + routing protocol + user preferences."""
    configured = _config_str(config, "system_prompt")
    system_prompt = configured if configured else DEFAULT_SYSTEM_CONTEXT

    if field_targets:
        system_prompt += ROUTING_PROTOCOL
        routing_instructions = _config_str(config, "routing_instructions").strip()
        if routing_instructions:
            system_prompt += f"\nUser routing preferences (follow these):\n{routing_instructions}"

    if custom_instructions:
        system_prompt += f"\n\nUser preferences:\n{custom_instructions}"

    if _config_bool(config, "learn_vocabulary", True):
        from .vocabulary import learned_corrections, vocabulary_prompt
        learned = vocabulary_prompt(learned_corrections())
        if learned:
            system_prompt += f"\n\n{learned}"

    return system_prompt


def correct_with_llm(
    results: List[TranscriptionResult],
    context: Optional[AppContext],
    config: ConfigSnapshot,
    on_delta: Optional[Callable[[str], None]] = None,
    history_context: str = "",
    on_metadata: Optional[Callable[[LLMCorrectionResult], None]] = None,
    custom_instructions: str = "",
    field_targets: Optional[list] = None,
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
        history_context: Recent transcriptions for continuity
        on_metadata: Optional callback to receive LLM result metadata
        custom_instructions: User's custom instructions
        field_targets: On-screen text fields for output routing
        on_generation_metadata: Called off-thread with (generation_id, usage)
            once OpenRouter reports cost/provider details
    """
    results = [r for r in results if r.text.strip()]
    if not results:
        return ""

    if not config.openrouter_api_key:
        print("[LLM] No OpenRouter API key configured")
        return ""

    prompt = _build_prompt(results, context, history_context, field_targets)
    system_prompt = build_system_prompt(config, field_targets, custom_instructions)

    total_words = max(len(r.text.split()) for r in results)
    est_tokens = (len(prompt) + len(system_prompt)) // 4
    model = _config_str(config, "openrouter_correction_model", OPENROUTER_MODEL_DEFAULT)

    start = time.perf_counter()
    first_token_at: List[float] = []

    def timing_delta(token: str) -> None:
        if not first_token_at:
            first_token_at.append(time.perf_counter())
        if on_delta is not None:
            on_delta(token)

    metadata: dict = {}
    result = _call_openrouter(
        prompt, system_prompt, config, timing_delta,
        metadata_out=metadata, on_generation_metadata=on_generation_metadata,
    )

    # One retry with a fresh attempt (transient errors, stream hiccups)
    if not result:
        print("[LLM] Retrying OpenRouter")
        metadata = {}
        result = _call_openrouter(
            prompt, system_prompt, config, timing_delta,
            metadata_out=metadata, on_generation_metadata=on_generation_metadata,
        )

    if not result:
        print("[LLM] Correction failed")
        return ""

    elapsed = (time.perf_counter() - start) * 1000
    ttft_s = (first_token_at[0] - start) if first_token_at else elapsed / 1000
    print(
        f"[LLM] {model} | in: {total_words} words, ~{est_tokens} tok prompt | "
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
    history_context: str = "",
    field_targets: Optional[list] = None,
) -> str:
    """Build the LLM prompt from transcription results."""

    # Deduplicate results by normalized text to save tokens
    seen_normalized: set = set()
    unique_results: List[TranscriptionResult] = []
    for r in results:
        normalized = " ".join(r.text.lower().split())
        if normalized and normalized not in seen_normalized:
            seen_normalized.add(normalized)
            unique_results.append(r)

    transcriptions = []
    for r in unique_results:
        transcriptions.append(f"[{r.provider}/{r.mic}]: {r.text}")

    transcription_text = "\n".join(transcriptions)

    context_parts = []

    if context:
        context_parts.append(f"Active application: {context.app_name}")
        if context.window_title:
            context_parts.append(f"Window: {context.window_title}")

    if history_context:
        context_parts.append(
            "### PRIOR DICTATIONS (already delivered; context only)\n"
            "Never transcribe, continue, or repeat any of this. It exists only to\n"
            "disambiguate pronouns and names in the new audio below.\n"
            f"{history_context}\n### END PRIOR DICTATIONS"
        )

    if field_targets:
        from .fields import describe_fields_for_prompt
        context_parts.append(
            f"Available text input fields:\n{describe_fields_for_prompt(field_targets)}"
        )

    context_text = "\n".join(context_parts) if context_parts else ""

    parts = []
    if context_text:
        parts.append(context_text)
    parts.append(f"Transcriptions:\n{transcription_text}")

    return "\n\n".join(parts)


def _call_openrouter(
    prompt: str,
    system_prompt: str,
    config: ConfigSnapshot,
    on_delta: Optional[Callable[[str], None]] = None,
    timeout: int = 15,
    metadata_out: Optional[dict] = None,
    on_generation_metadata: Optional[Callable[[str, dict], None]] = None,
) -> str:
    """Call OpenRouter's chat completions API with streaming."""
    if not config.openrouter_api_key:
        return ""

    model = _config_str(config, "openrouter_correction_model", OPENROUTER_MODEL_DEFAULT)
    provider_order = _config_str_list(config, "openrouter_correction_provider_order")
    allow_fallbacks = _config_bool(config, "openrouter_correction_allow_fallbacks", True)
    reasoning_effort = _config_str(config, "openrouter_correction_reasoning_effort").strip().lower()
    # ":nitro" and friends are routing hints on the same underlying model, so
    # the base id decides whether reasoning has to be switched off.
    if not reasoning_effort and model.split(":", 1)[0] in OPENROUTER_NO_REASONING_MODELS:
        reasoning_effort = "none"

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

    if reasoning_effort:
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
            print(f"[LLM] OpenRouter API error: {response.status_code}")
            return ""

        last_content_at = time.time()
        for line in response.iter_lines():
            if time.time() - last_content_at > _STREAM_STALL_SECONDS:
                print(f"[LLM] Stream stalled >{_STREAM_STALL_SECONDS:.0f}s, using what arrived")
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

            if "error" in parsed:
                print(f"[LLM] OpenRouter stream error: {parsed['error']}")
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

        return "".join(collected_chunks)

    except Exception as e:
        print(f"[LLM] OpenRouter error: {e}")
        return ""


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
