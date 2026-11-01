"""Load diarized speech segments and merge them into speaking turns."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd

REQUIRED = ["speaker", "start", "end"]


def load_segments(path: str | Path) -> pd.DataFrame:
    """Read diarized segments from a CSV/TSV table or a WhisperX-style JSON file.

    Tables need ``speaker``, ``start`` and ``end`` columns (seconds) and may
    include ``text``. JSON files may be a list of segments or a dict with a
    ``segments`` list, as written by WhisperX after speaker assignment.
    """
    path = Path(path)
    if path.suffix.lower() == ".json":
        raw = json.loads(path.read_text(encoding="utf-8"))
        rows = raw["segments"] if isinstance(raw, dict) else raw
        df = pd.DataFrame(rows)
    else:
        sep = "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","
        df = pd.read_csv(path, sep=sep)
    return segments_from_frame(df)


def segments_from_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalise a segment table."""
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"segments are missing required columns: {missing}")
    out = df.copy()
    out = out.dropna(subset=["speaker", "start", "end"])
    out["speaker"] = out["speaker"].astype(str)
    out["start"] = out["start"].astype(float)
    out["end"] = out["end"].astype(float)
    if "text" not in out.columns:
        out["text"] = ""
    out["text"] = out["text"].fillna("").astype(str).str.strip()
    bad = out["end"] < out["start"]
    if bad.any():
        raise ValueError(f"{int(bad.sum())} segments end before they start")
    return out.sort_values(["start", "end"]).reset_index(drop=True)[["speaker", "start", "end", "text"]]


def build_turns(
    segments: pd.DataFrame | Iterable[dict],
    merge_gap: float = 0.30,
    min_duration: float = 0.25,
) -> pd.DataFrame:
    """Merge consecutive same-speaker segments into turns.

    Segments from the same speaker separated by less than ``merge_gap`` seconds
    are joined into one turn; turns shorter than ``min_duration`` seconds are
    dropped. The defaults (300 ms, 250 ms) follow common practice for
    conversational turn segmentation.

    Returns one row per turn with ``speaker``, ``start``, ``end``, ``duration``
    and ``text``.
    """
    seg = segments if isinstance(segments, pd.DataFrame) else pd.DataFrame(list(segments))
    seg = segments_from_frame(seg)
    turns: list[dict] = []
    for row in seg.itertuples(index=False):
        if turns and turns[-1]["speaker"] == row.speaker and row.start - turns[-1]["end"] < merge_gap:
            last = turns[-1]
            last["end"] = max(last["end"], row.end)
            last["text"] = (last["text"] + " " + row.text).strip()
        else:
            turns.append({"speaker": row.speaker, "start": row.start, "end": row.end, "text": row.text})
    out = pd.DataFrame(turns, columns=["speaker", "start", "end", "text"])
    if out.empty:
        out["duration"] = []
        return out
    out["duration"] = out["end"] - out["start"]
    out = out[out["duration"] >= min_duration].reset_index(drop=True)
    return out[["speaker", "start", "end", "duration", "text"]]
