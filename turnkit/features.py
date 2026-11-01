"""Turn-taking and participation features for a single conversation."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

BACKCHANNELS = frozenset(
    {"yeah", "yes", "yep", "right", "okay", "ok", "sure", "mm", "mhm", "mm-hmm", "uh-huh",
     "totally", "definitely", "exactly", "true", "wow", "oh"}
)
I_PRONOUNS = frozenset({"i", "me", "my", "mine", "myself"})
WE_PRONOUNS = frozenset({"we", "us", "our", "ours", "ourselves"})
_WORD = re.compile(r"[a-z']+(?:-[a-z']+)*")


def _words(text: str) -> list[str]:
    return _WORD.findall(str(text).lower())


def gini(values) -> float:
    """Inequality of a set of non-negative amounts (0 = perfectly equal)."""
    x = np.asarray(list(values), dtype=float)
    if x.size < 2 or x.sum() == 0:
        return 0.0
    return float(np.abs(np.subtract.outer(x, x)).sum() / (2 * x.size * x.sum()))


def is_backchannel(text: str, backchannels=BACKCHANNELS, max_words: int = 2) -> bool:
    """A short turn made only of backchannel tokens, e.g. "yeah", "mm-hmm right"."""
    w = _words(text)
    return 0 < len(w) <= max_words and all(t in backchannels for t in w)


def transitions(turns: pd.DataFrame) -> pd.DataFrame:
    """Speaker changes between consecutive turns.

    ``gap`` is the incoming turn's start minus the outgoing turn's end: positive
    values are silences, negative values are overlaps.
    """
    t = turns.reset_index(drop=True)
    if len(t) < 2:
        return pd.DataFrame(columns=["to_index", "from_speaker", "to_speaker", "time", "gap"])
    prev, nxt = t.iloc[:-1].reset_index(drop=True), t.iloc[1:].reset_index(drop=True)
    switch = prev["speaker"].values != nxt["speaker"].values
    return pd.DataFrame({
        "to_index": np.arange(1, len(t))[switch],
        "from_speaker": prev["speaker"][switch].values,
        "to_speaker": nxt["speaker"][switch].values,
        "time": nxt["start"][switch].values,
        "gap": (nxt["start"] - prev["end"])[switch].values,
    })


def conversation_features(turns: pd.DataFrame, backchannels=BACKCHANNELS) -> dict:
    """Conversation-level turn-taking features.

    Parameters
    ----------
    turns : output of :func:`turnkit.build_turns`.

    Returns
    -------
    dict with timing features (turn length, switch rate, gaps, overlaps),
    participation balance, and simple lexical features when text is present.
    """
    if turns.empty:
        return {"n_turns": 0}
    t = turns.reset_index(drop=True)
    total_time = float(t["end"].max() - t["start"].min())
    speaking = t.groupby("speaker")["duration"].sum()
    tr = transitions(t)
    has_text = t["text"].str.len().gt(0).any()

    feats: dict = {
        "n_speakers": int(t["speaker"].nunique()),
        "n_turns": int(len(t)),
        "duration_sec": total_time,
        "mean_turn_duration": float(t["duration"].mean()),
        "median_turn_duration": float(t["duration"].median()),
        "n_switches": int(len(tr)),
        "switches_per_min": float(len(tr) / (total_time / 60)) if total_time > 0 else np.nan,
        "floor_share_gini": gini(speaking.values),
        "speaking_time_ratio": float(speaking.sum() / total_time) if total_time > 0 else np.nan,
        "mean_gap": float(tr.loc[tr["gap"] > 0, "gap"].mean()) if (tr["gap"] > 0).any() else np.nan,
        "median_switch_gap": float(tr["gap"].median()) if len(tr) else np.nan,
        "overlap_rate": float((tr["gap"] < 0).mean()) if len(tr) else np.nan,
    }
    if has_text:
        words = t["text"].map(_words)
        n_words = words.map(len)
        bc = t["text"].map(lambda s: is_backchannel(s, backchannels))
        # An interruption is an overlapping switch whose incoming turn is not a backchannel.
        interrupts = (tr["gap"].values < 0) & ~bc.values[tr["to_index"].astype(int).values] if len(tr) else np.array([])
        all_words = [w for ws in words for w in ws]
        feats.update({
            "word_gini": gini(n_words.groupby(t["speaker"]).sum().values),
            "backchannel_rate": float(bc.mean()),
            "question_rate": float(t["text"].str.contains(r"\?").mean()),
            "interruption_rate": float(interrupts.mean()) if len(tr) else np.nan,
            "type_token_ratio": len(set(all_words)) / len(all_words) if all_words else np.nan,
            "i_rate": sum(w in I_PRONOUNS for w in all_words) / max(1, len(all_words)),
            "we_rate": sum(w in WE_PRONOUNS for w in all_words) / max(1, len(all_words)),
        })
    return feats


def speaker_features(turns: pd.DataFrame, backchannels=BACKCHANNELS) -> pd.DataFrame:
    """Per-speaker features: floor share, turn length, and (with text) word use."""
    if turns.empty:
        return pd.DataFrame()
    t = turns.reset_index(drop=True)
    total_speaking = t["duration"].sum()
    rows = []
    for spk, g in t.groupby("speaker"):
        row = {
            "speaker": spk,
            "n_turns": int(len(g)),
            "speaking_time": float(g["duration"].sum()),
            "floor_share": float(g["duration"].sum() / total_speaking) if total_speaking else np.nan,
            "mean_turn_duration": float(g["duration"].mean()),
        }
        if g["text"].str.len().gt(0).any():
            words = [w for s in g["text"] for w in _words(s)]
            row.update({
                "n_words": len(words),
                "backchannel_rate": float(g["text"].map(lambda s: is_backchannel(s, backchannels)).mean()),
                "question_rate": float(g["text"].str.contains(r"\?").mean()),
                "type_token_ratio": len(set(words)) / len(words) if words else np.nan,
            })
        rows.append(row)
    return pd.DataFrame(rows)


def windowed_features(turns: pd.DataFrame, n_windows: int = 5, **kwargs) -> pd.DataFrame:
    """Conversation features in ``n_windows`` equal-time windows.

    Useful for asking whether turn-taking changes over a conversation. Turns
    are assigned to the window containing their start time.
    """
    if turns.empty:
        return pd.DataFrame()
    t = turns.reset_index(drop=True)
    t0, t1 = t["start"].min(), t["end"].max()
    edges = np.linspace(t0, t1, n_windows + 1)
    win = np.clip(np.searchsorted(edges, t["start"], side="right") - 1, 0, n_windows - 1)
    rows = []
    for k in range(n_windows):
        g = t[win == k]
        f = conversation_features(g, **kwargs) if len(g) else {"n_turns": 0}
        f.update({"window": k, "window_start": float(edges[k]), "window_end": float(edges[k + 1])})
        rows.append(f)
    return pd.DataFrame(rows)
