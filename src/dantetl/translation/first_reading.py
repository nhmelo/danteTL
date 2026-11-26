"""Bound first-reading work to one exact group per live set of lines."""

from dantetl.ocr.tracking import TextActivity
from dantetl.translation.grouping import TextBlock
from dantetl.translation.values import TranslationRequest


class FirstReadings:
    def __init__(self) -> None:
        self._members: dict[int, TranslationRequest | None] = {}

    def update(
        self,
        blocks: tuple[TextBlock, ...],
        ready: tuple[TextBlock, ...],
        activity: tuple[TextActivity, ...],
        blocked: set[int],
        *,
        reviewed: bool,
    ) -> tuple[TextBlock, ...]:
        observations = {a.observation.track_id: a for a in activity}
        confirmed = set(ready)
        previous, self._members = self._members, {}
        candidates = []
        for block in blocks:
            request = TranslationRequest.from_block(block)
            ids = tuple(t.track_id for t in block.lines)
            if block in confirmed or any(
                t.track_id in blocked
                or (a := observations.get(t.track_id)) is None
                or a.observation != t
                or a.pending
                or a.growing
                or (t.track_id in previous and previous[t.track_id] != request)
                for t in block.lines
            ):
                self._members.update(dict.fromkeys(ids))
                continue
            if not all(observations[i].reliable_first_read for i in ids):
                # A withdrawn reading cannot start fresh after a later quiet capture.
                self._members.update(dict.fromkeys(i for i in ids if i in previous))
                continue
            if reviewed or all(i in previous for i in ids):
                self._members.update(dict.fromkeys(ids, request))
                candidates.append(block)
        return tuple(candidates)


def can_preview_cached(block: TextBlock) -> bool:
    text, language = block.text, block.language
    if text.endswith((*language.commas, ":", "：")):
        return False
    if (
        language.starts_item(text)
        or language.section_heading(text)
        or language.is_status_counter(text)
    ):
        return True
    if (
        len(block.lines) > 1
        and ("%" in text or "％" in text)
        and language.explicit_sentence_end(text)
    ):
        return True
    return (
        len(block.lines) == 1
        and not block.dialogue
        and (text in language.standalone_menu_labels or 2 <= len(text) < 10)
        and not language.has_speaker_punctuation(text)
        and not language.quote_start.match(text)
    )
