"""
Deciding whether a transcript needs correcting at all.

Roughly a quarter of dictations come back already saying exactly what the
speaker meant, and waiting ~1.1s for a correction model to hand them back
unchanged is pure latency. Consensus catches some of those — but only when two
recognizers agree word for word on a short phrase with no filler in a
hard-coded list.

This asks a classifier instead. TypeSafe's Jev answers typed questions about a
state with calibrated probabilities, so the questions are the narrow, factual
kind and the decision to skip correction is made here, in code, from the numbers.

The questions are not a list of things that happen to go wrong. They are the
correction step's own jobs (see DEFAULT_SYSTEM_CONTEXT): drop filler, drop
stutters and false starts, fix mishearings, fix punctuation and grammar. A
transcript can be typed as it stands exactly when none of those jobs applies.

It runs *alongside* the correction request, never before it: the judge answers
in ~250ms and the correction model's first token lands around 750ms, so a
"clean" verdict wins the race and types immediately, while anything else costs
nothing because the correction was already in flight.

Measured on 600 logged single-chunk dictations corrected by the current model,
labelled by what that correction actually produced (scripts/judge_eval.py
re-runs it; about a quarter needed no correction at all). The first design
asked filler, restart and misheard about "the best entry": it skipped 9.7% of
dictations, but 14 of its 58 skips typed a word the correction would have fixed
(a misheard verb, a misheard noun). Jev scores each question on
its own, so "the best entry" in one question need not be the one another
question picked. Asking every job of every transcript, with a best pick that
can answer "none of them", skips 3.2% with 3 wrong of 19. A single holistic
question ("could this be sent as is?") and a graded "how much editing?" score
both did worse at every threshold. Adding the destination app and the
speaker's learned vocabulary to the state made it more conservative without
making it more accurate, so the state stays lean. The speaker's written
preferences still count, through a question of their own (see PREFERENCES).
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .types import AppContext, ConfigSnapshot, TranscriptionResult

DECISIONS_URL = "https://openrouter.ai/api/alpha/decisions"
# Follows the latest Jev, to improve as it does. LIMIT was fitted on
# jev-1.13-20260917; scripts/judge_eval.py shows whether a new version still
# sits well under it.
JUDGE_MODEL_DEFAULT = "~typesafe/jev-latest"

# The correction step's jobs, one snap judgment each, asked of every transcript.
# A question that bundles them ("could this be sent as is?") measured worse.
JOBS = {
    "filler": ("contains filler: sounds like um or uh, or words that carry no meaning such as "
               "like, you know, I mean, kind of, basically.",
               "at least one filler", "every word carries meaning"),
    "restart": ("contains a stutter, a repeated word or phrase, a false start, a self-correction, "
                "or something the speaker called off (scratch that, never mind).",
                "present", "the speech runs straight through"),
    "misheard": ("has a word the speech recognizer probably got wrong: one that does not fit the "
                 "sentence, or that the other transcripts hear differently in a way that changes "
                 "the meaning.",
                 "a likely recognition error", "every word is probably right"),
    "punct": ("needs punctuation, capitalization or grammar fixes that a careful writer would "
              "make before sending it.",
              "needs fixes", "reads correctly as written"),
}

# Asked only when the speaker has preferences (see preferences()), which typing a
# transcript as it stands would otherwise skip ("all lowercase in Slack"). They go in this
# question rather than the shared state: in the state they made every other
# question more cautious (skips fell from 8.0% to 4.2% of 400 dictations, with
# no fewer wrong ones), while here the other scores move by about 0.01. With a
# rule to write in lowercase in Slack and X, 12 of 30 logged dictations there
# were skipped before and none after; elsewhere skips held at 7.8% against 8.0%.
PREFERENCES = ("typed as it stands into {destination}, would break one of the speaker's "
               "preferences below that applies there.\n\nSpeaker's preferences:\n{preferences}",
               "breaks a preference", "follows every preference that applies here")

# One threshold for every question, including "none of them" on the best pick.
# Fitted as a single number so there is little to overfit; the sweep is in
# scripts/judge_eval.py. It stays well below 0.5 on purpose: a borderline
# dictation repeated came back 0.58 and then 0.41, while confident answers
# repeat exactly, so a cut in the middle would decide by coin flip.
LIMIT = 0.3


def _timeout_seconds(config: ConfigSnapshot, default: float = 0.9) -> float:
    """The judge's budget in seconds, tolerating a config that holds junk."""
    try:
        return float(getattr(config, "judge_timeout_ms", default * 1000)) / 1000.0
    except (TypeError, ValueError):
        return default


@dataclass
class Verdict:
    """What the judge concluded, and everything worth logging about it."""

    clean: bool
    text: str
    scores: Dict[str, float] = field(default_factory=dict)
    model: str = ""
    latency_ms: int = 0


def _candidates(results: List[TranscriptionResult]) -> List[str]:
    """Distinct non-empty transcripts, longest first so ties favour detail."""
    seen, out = set(), []
    for text in sorted((r.text.strip() for r in results), key=len, reverse=True):
        if text and text.lower() not in seen:
            seen.add(text.lower())
            out.append(text)
    return out


def preferences(config: ConfigSnapshot) -> str:
    """
    What the speaker asked of the correction step beyond JOBS: their written
    instructions, and their own prompt when it replaces the default. JOBS are
    the default prompt's jobs; a replaced prompt ("keep it formal", "answer in
    Spanish") asks for more, and a transcript typed as it stands can break that.
    Jev reads any whole prompt as work still to do, so a replaced one all but
    stops skips: the default prompt passed in as if replaced let 2 of 150
    dictations through against 8, and a Spanish one none. Safe, if slower.
    """
    from .correct import DEFAULT_SYSTEM_CONTEXT

    def text(name: str) -> str:
        value = getattr(config, name, "")
        return value.strip() if isinstance(value, str) else ""

    prompt = text("system_prompt")
    replaced = prompt if prompt != DEFAULT_SYSTEM_CONTEXT.strip() else ""
    return "\n\n".join(part for part in (replaced, text("custom_instructions")) if part)


def destination(context: Optional[AppContext]) -> str:
    """Where the text is going: the app, and its window title when there is one."""
    if not context or not context.app_name:
        return "an unknown app"
    title = context.window_title
    return f"{context.app_name} (window: {title})" if title else context.app_name


def questions(count: int, preferences: str = "", context: Optional[AppContext] = None) -> Dict[str, dict]:
    """Every question for `count` candidate transcripts, all answered in one pass."""
    out = {"best": {
        "type": "choice",
        "instructions": "Which entry of `transcripts` is most likely word for word what the speaker said?",
        "criteria": {**{str(i): f"transcript at index {i}" for i in range(count)},
                     "none": "none of them: each has at least one word wrong or missing"},
    }}
    jobs = dict(JOBS)
    if preferences:
        claim, yes, no = PREFERENCES
        jobs["prefs"] = (claim.format(destination=destination(context), preferences=preferences), yes, no)
    for i in range(count):
        for job, (claim, yes, no) in jobs.items():
            out[f"{job}_{i}"] = {"type": "noul", "instructions": f"Transcript at index {i} {claim}",
                                 "criteria": {"true": yes, "false": no}}
    return out


def decide(answers: dict, count: int):
    """
    (clean, index of the transcript to type, that transcript's scores).

    The pick is the likeliest transcript, not "none"; it is clean only when
    "none" is unlikely and none of the correction step's jobs applies to it.
    """
    probabilities = answers["best"].get("probabilities") or {}
    indices = [str(i) for i in range(count)]
    if any(i in probabilities for i in indices):
        pick = int(max(indices, key=lambda i: float(probabilities.get(i, 0.0))))
    else:
        choice = str(answers["best"].get("choice"))
        pick = int(choice) if choice in indices else 0
    scores = {"none": float(probabilities.get("none", 0.0))}
    scores.update({job: float(answers[f"{job}_{pick}"]["noul"])
                   for job in (*JOBS, "prefs") if f"{job}_{pick}" in answers})
    return all(v <= LIMIT for v in scores.values()), pick, scores


def judge_transcripts(results: List[TranscriptionResult], config: ConfigSnapshot,
                      context: Optional[AppContext] = None) -> Optional[Verdict]:
    """
    Ask whether these transcripts can be typed as they are.

    Returns None when the judge is off, unreachable, or too slow to matter —
    every one of which just means the correction model's answer is used, so the
    caller has nothing to handle.
    """
    # Strictly True: a config stub whose attributes are truthy objects must not
    # switch a network call on by accident.
    if getattr(config, "judge_enabled", False) is not True:
        return None
    if not isinstance(getattr(config, "openrouter_api_key", None), str) or not config.openrouter_api_key:
        return None
    candidates = _candidates(results)
    if not candidates:
        return None

    body = json.dumps({
        "model": getattr(config, "judge_model", "") or JUDGE_MODEL_DEFAULT,
        "state": {"transcripts": candidates},
        "questions": questions(len(candidates), preferences(config), context),
    }).encode()
    request = urllib.request.Request(DECISIONS_URL, body, {
        "Authorization": f"Bearer {config.openrouter_api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/kalpit-S/mergescribe",
        "X-Title": "MergeScribe",
    })

    started = time.monotonic()
    timeout = max(0.1, _timeout_seconds(config))
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.load(response)
    except (urllib.error.URLError, TimeoutError, ValueError, OSError) as e:
        print(f"[Judge] unavailable ({e}); using the correction model")
        return None

    try:
        clean, pick, scores = decide(payload["answers"], len(candidates))
    except (KeyError, ValueError, TypeError, AttributeError) as e:
        print(f"[Judge] unexpected reply ({e}); using the correction model")
        return None

    return Verdict(
        clean=clean,
        text=candidates[pick],
        scores=scores,
        model=str(payload.get("model", "")),
        latency_ms=round((time.monotonic() - started) * 1000),
    )
