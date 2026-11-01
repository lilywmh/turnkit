"""Command line: turnkit FILE [FILE ...] --out features.csv"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .features import conversation_features, speaker_features, windowed_features
from .turns import build_turns, load_segments


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="turnkit", description="Turn-taking features from diarized conversations.")
    p.add_argument("files", nargs="+", help="CSV/TSV or WhisperX JSON files, one conversation each")
    p.add_argument("--out", default="turnkit_features.csv", help="output CSV (default: %(default)s)")
    p.add_argument("--level", choices=["conversation", "speaker", "window"], default="conversation")
    p.add_argument("--windows", type=int, default=5, help="number of windows for --level window")
    p.add_argument("--merge-gap", type=float, default=0.30, help="seconds; merge same-speaker segments closer than this")
    p.add_argument("--min-duration", type=float, default=0.25, help="seconds; drop turns shorter than this")
    a = p.parse_args(argv)

    frames = []
    for f in a.files:
        turns = build_turns(load_segments(f), merge_gap=a.merge_gap, min_duration=a.min_duration)
        if a.level == "conversation":
            df = pd.DataFrame([conversation_features(turns)])
        elif a.level == "speaker":
            df = speaker_features(turns)
        else:
            df = windowed_features(turns, n_windows=a.windows)
        df.insert(0, "conversation", Path(f).stem)
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(a.out, index=False)
    print(f"Wrote {len(out)} rows to {a.out}")


if __name__ == "__main__":
    main()
