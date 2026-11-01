# turnkit

**Turn-taking features from diarized conversations.**

`turnkit` turns speaker-labeled timestamps (for example, WhisperX output) into
interpretable measures of how a conversation is structured: how long people
hold the floor, how often they switch, how they share speaking time, how much
they overlap, and how this changes over the course of the conversation.

It grew out of my work on what makes conversations between strangers feel
connected, where the pace of turn exchange turned out to be one of the most
stable behavioral correlates of how partners rated each other. The code here is
a clean, data-free version of the feature extraction I use in that research.

## Install

```bash
pip install git+https://github.com/lilywmh/turnkit.git
```

Requires Python 3.9+ with NumPy and pandas.

## Quickstart

```python
import turnkit as tk

segments = tk.load_segments("conversation.json")   # WhisperX JSON, or CSV/TSV
turns = tk.build_turns(segments)                     # merge segments into turns

tk.conversation_features(turns)   # one dict per conversation
tk.speaker_features(turns)        # one row per speaker
tk.windowed_features(turns, 5)    # the same features in 5 time windows
```

From the command line, one row per file:

```bash
turnkit data/*.json --out features.csv
turnkit data/*.json --level speaker --out speakers.csv
turnkit data/*.json --level window --windows 5 --out dynamics.csv
```

Try it on the included example:

```bash
python examples/quickstart.py
```

## Input

Any table with one row per speech segment:

| column    | meaning                          |
|-----------|----------------------------------|
| `speaker` | speaker label                    |
| `start`   | start time in seconds            |
| `end`     | end time in seconds              |
| `text`    | transcript (optional)            |

WhisperX JSON files with a `segments` list are read directly. Without `text`,
you still get every timing feature; lexical features are skipped.

## How turns are built

Consecutive segments from the same speaker are merged when the silence between
them is shorter than `merge_gap` (default 0.30 s), and turns shorter than
`min_duration` (default 0.25 s) are dropped. Both are arguments to
`build_turns` and flags on the command line.

## Features

**Timing** (always available)

| feature | definition |
|---|---|
| `n_turns`, `n_speakers`, `duration_sec` | counts and span of the conversation |
| `mean_turn_duration`, `median_turn_duration` | seconds per turn |
| `n_switches`, `switches_per_min` | changes of speaker between consecutive turns |
| `floor_share_gini` | inequality of speaking time across speakers (0 = equal) |
| `speaking_time_ratio` | share of the conversation with someone speaking |
| `mean_gap` | mean silence at speaker switches |
| `median_switch_gap` | median gap at switches (negative = overlap) |
| `overlap_rate` | share of switches where the next speaker starts early |

**Text** (when a `text` column is present)

| feature | definition |
|---|---|
| `interruption_rate` | overlapping switches whose incoming turn is not a backchannel |
| `backchannel_rate` | share of turns that are only "yeah", "mm-hmm", "right", etc. |
| `question_rate` | share of turns containing a question |
| `word_gini` | inequality of words spoken across speakers |
| `type_token_ratio` | lexical diversity |
| `i_rate`, `we_rate` | first-person singular and plural pronoun rates |

The backchannel word list is a parameter (`backchannels=`), so you can adapt it
to your data or language.

## Notes

- Feature definitions are simple on purpose, so they are easy to report in a
  methods section. Check them against your own data before relying on them,
  especially diarization quality, which drives every timing measure.
- Lexical features use a basic English tokenizer.

## Tests

```bash
pip install -e ".[test]"
pytest
```

## License

MIT. Author: Meihui (Lily) Wu, University of Southern California.
