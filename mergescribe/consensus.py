"""
Consensus checking for transcription results.

Uses normalized text comparison to handle punctuation differences.
"Hello world." and "Hello world" and "Hello, world" all match.
"""

import re
from collections import Counter
from typing import Optional, List

from .types import TranscriptionResult, ConfigSnapshot


# Filler words that should trigger LLM correction
FILLER_WORDS = {"um", "uh", "uhm", "umm", "hmm", "hm", "er", "ah", "like", "you know", "i mean", "sort of", "kind of"}


def _contains_filler(text: str) -> bool:
    """Check if text contains filler words."""
    words = text.lower().split()
    # Check single-word fillers
    for word in words:
        if word in FILLER_WORDS:
            return True
    # Check multi-word fillers
    text_lower = text.lower()
    for filler in FILLER_WORDS:
        if " " in filler and filler in text_lower:
            return True
    return False


def normalize_for_matching(text: str) -> str:
    """
    Strip punctuation and normalize whitespace for comparison.

    Examples:
        "Hello world." -> "hello world"
        "Hello, world" -> "hello world"
        "Hello   world" -> "hello world"
    """
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = ' '.join(text.lower().split())  # Normalize whitespace
    return text


def check_consensus(
    results: List[TranscriptionResult],
    config: ConfigSnapshot
) -> Optional[str]:
    """
    Check if enough providers agree on the transcription.

    Uses normalized comparison to handle punctuation differences.
    Returns the original text (with punctuation) from the first matching result.

    Args:
        results: List of transcription results from different providers/mics
        config: Configuration snapshot with consensus thresholds

    Returns:
        Agreed-upon text if consensus reached, None otherwise
    """
    if not results:
        return None

    # Normalize for comparison
    normalized = [(r, normalize_for_matching(r.text)) for r in results]

    # Filter out empty results
    normalized = [(r, norm) for r, norm in normalized if norm]
    if not normalized:
        return None

    # Count occurrences
    counts = Counter(norm for _, norm in normalized)
    winner_norm, count = counts.most_common(1)[0]

    # Check thresholds
    if count >= config.consensus_threshold:
        word_count = len(winner_norm.split())
        if word_count <= config.consensus_max_words:
            # Check for filler words - route to LLM if found
            if _contains_filler(winner_norm):
                return None

            # Return original text (with punctuation) from first match
            for result, norm in normalized:
                if norm == winner_norm:
                    return result.text

    return None
