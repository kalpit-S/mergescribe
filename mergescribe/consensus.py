"""
Consensus: enough distinct recognizers heard the same words.

It settles what was said, so a chunk stops waiting for slower recognizers.
It does not settle whether the text needs tidying - "um, ship it" can be
agreed on too - so agreed text still goes to the judge and the correction
model like any other.

Compared after normalising: "Hello world." and "Hello, world" match.
"""

import re
from collections import Counter
from typing import Optional, List

from .types import TranscriptionResult, ConfigSnapshot


def normalize_for_matching(text: str) -> str:
    """
    Strip punctuation and normalize whitespace for comparison.

    Examples:
        "Hello world." -> "hello world"
        "Hello, world" -> "hello world"
        "Hello   world" -> "hello world"
    """
    # Apostrophes join a word ("don't"); any other mark separates words, so
    # "re-sign" stays two words and doesn't match "resign".
    text = re.sub(r"[^\w\s]", " ", text.replace("'", "").replace("\u2019", ""))
    return " ".join(text.lower().split())


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
    winner_norm, _ = counts.most_common(1)[0]

    # Count distinct PROVIDERS, not raw matching results. The same model
    # transcribing two mics agrees with itself on its own systematic errors:
    # two mics rule out acoustic noise, not model bias. Proper nouns and jargon
    # (a surname heard as a common word) are pure model prior, so both mics return
    # the identical mistake. Cross-provider agreement is evidence; cross-mic
    # agreement is not.
    agreeing_providers = {r.provider for r, norm in normalized if norm == winner_norm}

    if len(agreeing_providers) < config.consensus_threshold:
        return None
    # The original text (with punctuation) from the first match
    return next(result.text for result, norm in normalized if norm == winner_norm)
