"""Tests for scripts/provider_value.py: what counts as a word only one recognizer heard."""

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "provider_value", Path(__file__).resolve().parent.parent / "scripts" / "provider_value.py")
pv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pv)


def test_a_word_only_one_recognizer_heard_counts():
    assert pv.heard_alone(pv.words("ask Claude to fix it"), [pv.words("ask cloud to fix it")]) == {"claude"}


def test_spelling_variants_are_not_disagreements():
    assert pv.heard_alone(pv.words("send the e-mail setup"), [pv.words("send the email set up")]) == set()
    assert pv.heard_alone(pv.words("okay, 3 things"), [pv.words("OK, three things")]) == set()


def test_filler_words_are_ignored():
    assert pv.heard_alone(pv.words("um so uh yes"), [pv.words("so yes")]) == set()


def test_stream_names_match_what_the_provider_reports():
    from mergescribe.providers.openrouter_stt import OpenRouterSTTProvider

    for model in ("assemblyai/universal-3-5-pro", "fish-audio/transcribe-1-pro", "microsoft/mai-transcribe-2"):
        assert pv.stream_name(model) == OpenRouterSTTProvider(api_key="k", model=model).name
