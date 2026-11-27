"""Small, explicitly scoped conversation memory; never a screen-wide transcript."""

from dataclasses import dataclass

from dantetl.domain import Rect
from dantetl.ocr.observation import TextTrack
from dantetl.translation.grouping import TextBlock
from dantetl.translation.limits import MAX_CONTEXT_CHARACTERS, MAX_CONTEXT_TURNS


@dataclass(frozen=True, slots=True)
class DialogueTurn:
    source: str
    english: str


def _same_panel(a: Rect, b: Rect) -> bool:
    height = max(20, min(a.height, b.height))
    return abs(a.x - b.x) <= height and abs(a.y - b.y) <= height


def _continues(
    previous: TextBlock,
    current: TextBlock,
    corrections: tuple[tuple[TextTrack, TextTrack], ...],
) -> bool:
    """Follow exact members, tracker-approved corrections and tracked continuation only.

    Reusing a screen location or similar wording does not establish turn identity.
    A trailing line may grow and new lines may follow; earlier members must survive.
    """
    if len(current.lines) < len(previous.lines):
        return False
    predecessors = dict(corrections)
    for index, old in enumerate(previous.lines):
        new = current.lines[index]
        if new.track_id != old.track_id:
            return False
        if new.text == old.text or predecessors.get(new) == old:
            continue
        if not (
            index == len(previous.lines) - 1
            and new.text_revision > old.text_revision
            and new.text.startswith(old.text)
        ):
            return False
    return True


class DialogueMemory:
    """At most two complete published turns / 512 characters, expiring after 30s.

    Only one unambiguous dialogue panel may own history. Short menu labels never
    qualify. Revisions of a revealing/corrected utterance replace its history entry.
    """

    def __init__(self) -> None:
        self._history: tuple[tuple[int, DialogueTurn], ...] = ()
        self._bounds: Rect | None = None
        self._current: TextBlock | None = None
        self._anchor: TextBlock | None = None
        self._turn = 0
        self._last_seen = float("-inf")
        self._last_published = float("-inf")

    def reset(self) -> None:
        self.__init__()

    def observe(
        self,
        blocks: tuple[TextBlock, ...],
        now: float,
        *,
        corrections: tuple[tuple[TextTrack, TextTrack], ...] = (),
    ) -> None:
        if now - self._last_seen > 3 or (self._history and now - self._last_published > 30):
            self._history = ()
            self._bounds = None
            self._anchor = None
        candidates = tuple(
            b
            for b in blocks
            if b.dialogue
            or (
                self._bounds is not None
                and _same_panel(self._bounds, b.bounds)
                and b.text.endswith(b.language.endings)
                and len(b.text) >= 2
                and not b.text.startswith(b.language.lists.possible_prefix)
            )
        )
        if len(candidates) != 1:
            self._current = None
            if candidates:
                self.reset()  # Ambiguous panels/choices do not share a conversation.
            return
        current = candidates[0]
        if self._bounds is not None and not _same_panel(self._bounds, current.bounds):
            self._history = ()
            self._anchor = None
        if self._anchor is None or not _continues(self._anchor, current, corrections):
            self._turn += 1
        self._anchor = current
        self._current, self._bounds, self._last_seen = current, current.bounds, now

    def context_for(self, block: TextBlock) -> tuple[DialogueTurn, ...]:
        if self._current is None or block.key != self._current.key:
            return ()
        return tuple(t for token, t in self._history if token != self._turn)

    def published(self, block: TextBlock, english: str, now: float) -> None:
        if self._current is None or self._current.key != block.key:
            return
        history = tuple(t for t in self._history if t[0] != self._turn)
        turn = DialogueTurn(block.text, english)
        if len(turn.source) + len(turn.english) > MAX_CONTEXT_CHARACTERS:
            return
        history = (*history, (self._turn, turn))[-MAX_CONTEXT_TURNS:]
        while sum(len(t.source) + len(t.english) for _, t in history) > MAX_CONTEXT_CHARACTERS:
            history = history[1:]
        self._history, self._last_published = history, now
