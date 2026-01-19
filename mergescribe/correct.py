"""
LLM correction for transcription results.

Sends transcription results to an LLM for correction and formatting.
Uses automatic routing based on available API keys and input length.
"""

import json
import threading
import time
from typing import Callable, List, Optional

import requests

from .types import TranscriptionResult, AppContext, ConfigSnapshot, LLMCorrectionResult
from .router import CorrectionRouter, GROQ_MODEL, GEMINI_MODEL, OPENROUTER_MODEL


# Persistent sessions for connection reuse (saves ~70-100ms per request)
_openrouter_session = requests.Session()
_gemini_session = requests.Session()

# Groq client with thread-safe initialization
_groq_client = None
_groq_client_key = None
_groq_lock = threading.Lock()


def _get_groq_client(api_key: str):
    """Lazy-load Groq client with API key. Thread-safe, recreates on key change."""
    global _groq_client, _groq_client_key

    # Fast path: client exists with same key
    if _groq_client is not None and _groq_client_key == api_key:
        return _groq_client

    with _groq_lock:
        # Double-check after acquiring lock
        if _groq_client is not None and _groq_client_key == api_key:
            return _groq_client

        try:
            from groq import Groq
            _groq_client = Groq(api_key=api_key)
            _groq_client_key = api_key
        except ImportError:
            print("[LLM] Groq package not installed")
            return None

    return _groq_client


# Default system prompt for correction
DEFAULT_SYSTEM_CONTEXT = """You are a transcription assistant that cleans up speech-to-text output while preserving the speaker's authentic voice and exact meaning.

Clean up:
- Remove pure filler sounds: "um", "uh", "er", "ah", "hmm"
- Fix obvious transcription errors and typos (e.g. "lead code" → "leetcode")
- Handle self-corrections: use the correction, not the mistake
  Examples: "Tuesday, no wait, Friday" → "Friday"
            "Send it to John, I mean Jane" → "Send it to Jane"
- Fix grammar and add proper punctuation

BE CONSERVATIVE - when in doubt, preserve the original words:
- Keep "I mean" at the start of sentences (it's intentional emphasis)
- Keep tag questions like "right?" or "you know?" at the end
- Keep "like" unless it's clearly a filler (e.g., "it was, like, so big")
- Keep words that are being discussed or quoted

When multiple transcriptions are provided, compare and choose the most accurate parts from each.

Preserve:
- The speaker's meaning and intent
- Natural speaking style, slang, and strong language
- All substantive content

Meta-commands (follow these, don't transcribe them):
- "scratch that", "never mind", "forget what I said" → remove the previous content

Formatting:
- Model names use digits, not words: "GPT 5.2", "Gemini 2.5 Pro", "Claude 3.5"

Return only the cleaned transcription text."""


def correct_with_llm(
    results: List[TranscriptionResult],
    context: Optional[AppContext],
    config: ConfigSnapshot,
    on_delta: Optional[Callable[[str], None]] = None,
    history_context: str = "",
    on_metadata: Optional[Callable[[LLMCorrectionResult], None]] = None,
    custom_instructions: str = "",
) -> str:
    """
    Call LLM to correct/merge transcriptions.

    Automatically routes to the best available provider based on:
    - Which API keys are configured
    - Input length (short → fast provider, long → smart provider)

    Args:
        results: All transcription results from providers/mics
        context: Active application context (for prompt customization)
        config: Configuration snapshot
        on_delta: Optional callback for streaming tokens
        history_context: Recent transcriptions for continuity
        on_metadata: Optional callback to receive LLM result metadata
        custom_instructions: User's custom instructions (e.g., "When in Twitter, use lowercase")

    Returns:
        Corrected text
    """
    if not results:
        return ""

    # Build prompt
    prompt = _build_prompt(results, context, history_context)

    # Build system prompt - use custom if configured, otherwise default
    system_prompt = config.system_prompt if config.system_prompt else DEFAULT_SYSTEM_CONTEXT
    if custom_instructions:
        system_prompt += f"\n\nUser preferences:\n{custom_instructions}"

    # Count words for routing decision (use longest single transcription, not sum)
    total_words = max(len(r.text.split()) for r in results) if results else 0

    # Create router and select provider
    router = CorrectionRouter(config)
    provider = router.select_provider(total_words)

    if not provider:
        print("[LLM] No LLM API keys configured")
        return ""

    # Estimate tokens for logging
    est_tokens = len(prompt) // 4

    start = time.perf_counter()
    result = None

    # Try selected provider
    result = _call_provider(provider.name, prompt, system_prompt, config, on_delta)

    # If failed, try fallback
    if not result:
        router.record_failure(provider.name)
        fallback = router.get_fallback(exclude=provider.name)
        if fallback:
            print(f"[LLM] Falling back to {fallback.name}")
            result = _call_provider(fallback.name, prompt, system_prompt, config, on_delta)
            if result:
                router.record_success(fallback.name)
                provider = fallback  # Update for metadata
            else:
                router.record_failure(fallback.name)
    else:
        router.record_success(provider.name)

    if not result:
        print("[LLM] All providers failed")
        return ""

    elapsed = (time.perf_counter() - start) * 1000
    print(f"[LLM] {provider.name} ({total_words} words) -> {elapsed/1000:.2f}s")

    if on_metadata:
        on_metadata(LLMCorrectionResult(
            text=result,
            provider=provider.name,
            model=provider.model,
            input_tokens_est=est_tokens,
            latency_ms=elapsed,
            streamed=on_delta is not None,
        ))

    return result


def _call_provider(
    provider_name: str,
    prompt: str,
    system_prompt: str,
    config: ConfigSnapshot,
    on_delta: Optional[Callable[[str], None]] = None,
) -> str:
    """Call the specified provider."""
    if provider_name == "groq":
        return _call_groq(prompt, system_prompt, config, on_delta)
    elif provider_name == "gemini":
        return _call_gemini_direct(prompt, system_prompt, config, on_delta)
    elif provider_name == "openrouter":
        return _call_openrouter(prompt, system_prompt, config, on_delta)
    else:
        print(f"[LLM] Unknown provider: {provider_name}")
        return ""


def _build_prompt(
    results: List[TranscriptionResult],
    context: Optional[AppContext],
    history_context: str = "",
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

    # Format transcriptions
    transcriptions = []
    for r in unique_results:
        transcriptions.append(f"[{r.provider}/{r.mic}]: {r.text}")

    transcription_text = "\n".join(transcriptions)

    # Context sections
    context_parts = []

    if context:
        context_parts.append(f"Active application: {context.app_name}")
        if context.window_title:
            context_parts.append(f"Window: {context.window_title}")

    if history_context:
        context_parts.append(f"Previous context (for reference only, do not include in output): {history_context}")

    context_text = "\n".join(context_parts) if context_parts else ""

    # Build prompt
    rigor = context.rigor_level if context else "normal"

    if rigor == "high":
        style_note = "Style: formal (strict grammar)"
    elif rigor == "low":
        style_note = "Style: casual (preserve natural speech)"
    else:
        style_note = ""

    parts = []
    if context_text:
        parts.append(context_text)
    if style_note:
        parts.append(style_note)
    parts.append(f"Transcriptions:\n{transcription_text}")

    return "\n\n".join(parts)


def _call_groq(
    prompt: str,
    system_prompt: str,
    config: ConfigSnapshot,
    on_delta: Optional[Callable[[str], None]] = None,
    timeout: int = 15,
) -> str:
    """Call Groq's GPT-OSS model for fast correction."""
    client = _get_groq_client(config.groq_api_key)
    if not client:
        return ""

    try:
        if on_delta:
            collected = []
            completion = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_completion_tokens=2000,
                stream=True,
            )
            for chunk in completion:
                content = chunk.choices[0].delta.content
                if content:
                    collected.append(content)
                    on_delta(content)
            return "".join(collected)
        else:
            completion = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_completion_tokens=2000,
            )
            return completion.choices[0].message.content or ""

    except Exception as e:
        print(f"[LLM] Groq error: {e}")
        return ""


def _call_gemini_direct(
    prompt: str,
    system_prompt: str,
    config: ConfigSnapshot,
    on_delta: Optional[Callable[[str], None]] = None,
    timeout: int = 15,
) -> str:
    """Call Gemini API directly for correction."""
    if not config.gemini_api_key:
        return ""

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={config.gemini_api_key}"

    data = {
        "contents": [
            {"role": "user", "parts": [{"text": f"{system_prompt}\n\n{prompt}"}]}
        ],
        "generationConfig": {
            "temperature": 0.3,
            "maxOutputTokens": 2000,
            "thinkingConfig": {"thinkingBudget": 0},
        },
    }

    try:
        response = _gemini_session.post(url, json=data, timeout=timeout)

        if response.status_code != 200:
            print(f"[LLM] Gemini API error: {response.status_code}")
            return ""

        result = response.json()
        text = result.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")

        if on_delta and text:
            on_delta(text)

        return text

    except Exception as e:
        print(f"[LLM] Gemini error: {e}")
        return ""


def _call_openrouter(
    prompt: str,
    system_prompt: str,
    config: ConfigSnapshot,
    on_delta: Optional[Callable[[str], None]] = None,
    timeout: int = 15,
) -> str:
    """Call OpenRouter API with streaming support."""
    if not config.openrouter_api_key:
        return ""

    headers = {
        "Authorization": f"Bearer {config.openrouter_api_key}",
        "Content-Type": "application/json",
    }

    data = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.3,
        "max_tokens": 2000,
        "stream": True,
    }

    collected_chunks: List[str] = []

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

        for line in response.iter_lines():
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

            if "error" in parsed:
                print(f"[LLM] OpenRouter stream error: {parsed['error']}")
                break

            try:
                choice = parsed.get("choices", [{}])[0]
                delta = choice.get("delta", {})
                content = delta.get("content")
                if content:
                    collected_chunks.append(content)
                    if on_delta is not None:
                        on_delta(content)
            except Exception:
                continue

        return "".join(collected_chunks)

    except Exception as e:
        print(f"[LLM] OpenRouter error: {e}")
        return ""


DEFAULT_EDITING_PROMPT = "You are a text editing assistant. Apply the user's requested change precisely and return only the edited text."


def edit_text_with_llm(
    selected_text: str,
    voice_command: str,
    config: ConfigSnapshot,
) -> str:
    """
    Apply a voice command to edit selected text via LLM.

    Uses the same routing logic as correction.
    """
    # Create a simple prompt
    prompt = (
        f"TASK: {voice_command}\n\n"
        f"ORIGINAL TEXT:\n{selected_text}\n\n"
        "INSTRUCTIONS: Apply the task to the original text above. Return ONLY the edited text, "
        "nothing else. No explanations, no formatting, no extra content."
    )

    # Use custom editing prompt if configured
    system_prompt = config.editing_prompt if config.editing_prompt else DEFAULT_EDITING_PROMPT

    # Use router to select provider
    router = CorrectionRouter(config)
    # Text editing is usually short, but use priority for quality
    provider = router.select_provider(word_count=50)  # Treat as "long" for quality

    if not provider:
        print("[LLM] No LLM API keys configured")
        return selected_text

    result = _call_provider(provider.name, prompt, system_prompt, config)

    if not result:
        # Try fallback
        fallback = router.get_fallback(exclude=provider.name)
        if fallback:
            result = _call_provider(fallback.name, prompt, system_prompt, config)

    return result if result else selected_text
