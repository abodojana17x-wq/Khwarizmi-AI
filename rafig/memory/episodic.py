"""EpisodicMemory — timestamped experiences.

Episodic memories record *what happened*: an interaction, an event,
a decision.  They are the raw material RAFIQ can later consult when
reasoning about similar situations.
"""

from __future__ import annotations

from pathlib import Path

from ._sqlite import SQLiteMemory


class EpisodicMemory(SQLiteMemory):
    _table = "episodes"
    _default_source = "episodic"
