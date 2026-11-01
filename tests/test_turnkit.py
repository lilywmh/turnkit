import json

import numpy as np
import pandas as pd
import pytest

import turnkit as tk


def seg(rows):
    return pd.DataFrame(rows, columns=["speaker", "start", "end", "text"])


def test_merge_same_speaker_within_gap():
    s = seg([("A", 0.0, 1.0, "hello"), ("A", 1.2, 2.0, "there"), ("B", 2.5, 3.0, "hi")])
    t = tk.build_turns(s)
    assert list(t["speaker"]) == ["A", "B"]
    assert t.loc[0, "text"] == "hello there"
    assert t.loc[0, "end"] == pytest.approx(2.0)


def test_no_merge_beyond_gap_and_drop_short():
    s = seg([("A", 0.0, 1.0, "a"), ("A", 1.5, 2.0, "b"), ("B", 2.1, 2.2, "x")])
    t = tk.build_turns(s)
    assert list(t["speaker"]) == ["A", "A"]  # 0.5 s gap kept separate; 0.1 s turn dropped


def test_conversation_features_timing():
    s = seg([("A", 0, 2, "so what did you think?"), ("B", 2.5, 4, "i loved it"),
             ("A", 3.8, 6, "me too we should go"), ("B", 6.2, 6.6, "yeah")])
    t = tk.build_turns(s)
    f = tk.conversation_features(t)
    assert f["n_turns"] == 4
    assert f["n_switches"] == 3
    assert f["mean_turn_duration"] == pytest.approx((2 + 1.5 + 2.2 + 0.4) / 4)
    assert f["switches_per_min"] == pytest.approx(3 / (6.6 / 60))
    assert f["overlap_rate"] == pytest.approx(1 / 3)
    assert f["interruption_rate"] == pytest.approx(1 / 3)
    assert f["backchannel_rate"] == pytest.approx(1 / 4)
    assert f["question_rate"] == pytest.approx(1 / 4)


def test_backchannel_overlap_is_not_interruption():
    s = seg([("A", 0, 3, "long story here"), ("B", 2.5, 2.9, "mm-hmm"), ("A", 3.3, 5, "and then")])
    f = tk.conversation_features(tk.build_turns(s))
    assert f["overlap_rate"] > 0
    assert f["interruption_rate"] == 0


def test_gini_and_floor_share():
    assert tk.gini([1, 1]) == 0
    assert tk.gini([1, 0]) == pytest.approx(0.5)
    s = seg([("A", 0, 3, "a"), ("B", 3.5, 4.5, "b")])
    sp = tk.speaker_features(tk.build_turns(s)).set_index("speaker")
    assert sp.loc["A", "floor_share"] == pytest.approx(0.75)


def test_timing_only_input_has_no_text_features():
    s = pd.DataFrame({"speaker": ["A", "B"], "start": [0, 1.5], "end": [1, 2.5]})
    f = tk.conversation_features(tk.build_turns(s))
    assert "backchannel_rate" not in f and f["n_switches"] == 1


def test_windowed_features():
    s = seg([(("A", "B")[i % 2], i * 2.0, i * 2.0 + 1.5, "word") for i in range(10)])
    w = tk.windowed_features(tk.build_turns(s), n_windows=5)
    assert len(w) == 5 and w["n_turns"].sum() == 10


def test_load_whisperx_json(tmp_path):
    p = tmp_path / "conv.json"
    p.write_text(json.dumps({"segments": [
        {"speaker": "SPEAKER_00", "start": 0.0, "end": 1.0, "text": "hi"},
        {"speaker": "SPEAKER_01", "start": 1.4, "end": 2.0, "text": "hey"},
    ]}))
    t = tk.build_turns(tk.load_segments(p))
    assert len(t) == 2


def test_bad_segments_raise():
    with pytest.raises(ValueError):
        tk.segments_from_frame(pd.DataFrame({"speaker": ["A"], "start": [2.0]}))
    with pytest.raises(ValueError):
        tk.segments_from_frame(pd.DataFrame({"speaker": ["A"], "start": [2.0], "end": [1.0]}))
