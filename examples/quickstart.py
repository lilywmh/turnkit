"""Run from the repository root: python examples/quickstart.py"""
from pathlib import Path

import turnkit as tk

here = Path(__file__).parent
segments = tk.load_segments(here / "sample_conversation.csv")
turns = tk.build_turns(segments)

print("Conversation-level features")
for k, v in tk.conversation_features(turns).items():
    print(f"  {k:22s} {v:.3f}" if isinstance(v, float) else f"  {k:22s} {v}")

print("\nPer-speaker features")
print(tk.speaker_features(turns).round(3).to_string(index=False))

print("\nHow turn-taking changes across the conversation (3 windows)")
cols = ["window", "n_turns", "mean_turn_duration", "switches_per_min"]
print(tk.windowed_features(turns, n_windows=3)[cols].round(2).to_string(index=False))
