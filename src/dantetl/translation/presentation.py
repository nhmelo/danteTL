"""Bounded, atomic publication of English for live OCR tracks."""

from dataclasses import dataclass, replace

from dantetl.ocr.observation import TextTrack
from dantetl.ocr.regions import intersects
from dantetl.ocr.visual_values import MenuSample
from dantetl.translation.first_reading import can_preview_cached
from dantetl.translation.grouping import TextBlock
from dantetl.translation.values import TranslatedText, TranslationRequest


@dataclass(frozen=True, slots=True)
class Publication:
    painted: TextBlock
    translation: TranslatedText

    def at_live_bounds(self, live: dict[int, TextTrack]) -> "Publication":
        lines = self.painted.lines
        if all(t.bounds == live[t.track_id].bounds for t in lines):
            return self
        placed = tuple(replace(t, bounds=live[t.track_id].bounds) for t in lines)
        return Publication(replace(self.painted, lines=placed), self.translation)


class EnglishPresentation:
    """Own the painted scene independently of pending OCR revisions and translation work."""

    def __init__(self) -> None:
        self._published: tuple[Publication, ...] = ()
        self._visible: tuple[Publication, ...] = ()
        self._concealed: dict[int, str] = {}
        self._ready: dict[TranslationRequest, TranslatedText] = {}
        self._scope: object = None
        self._menus: dict[MenuSample, Publication] = {}
        self._inputs: (
            tuple[
                tuple[TextBlock, ...],
                tuple[TranslatedText, ...],
                frozenset[TextTrack],
                frozenset[TranslationRequest] | None,
            ]
            | None
        ) = None

    def reset(self) -> None:
        self._published = ()
        self._visible = ()
        self._concealed.clear()
        self._ready.clear()
        self._inputs = None
        self._scope = None
        self._menus.clear()

    def menu_previews(
        self,
        matches: tuple[MenuSample, ...],
        retained: tuple[TextBlock, ...],
    ) -> tuple[Publication, ...]:
        """Paint an exact returning label until normal OCR publication takes over."""
        if not matches:
            return ()
        singles = {
            p.painted.lines[0]: p
            for p in self._visible
            if len(p.painted.lines) == 1 and not p.painted.dialogue
        }
        current = {line for block in retained for line in block.lines}
        previews = []
        for sample in matches:
            line = sample.line
            if (
                line in current
                and line in singles
                and singles[line].translation.request.source == line.text
            ):
                self._menus[sample] = singles[line]
                if len(self._menus) > 16:
                    del self._menus[next(iter(self._menus))]
            stored = self._menus.get(sample)
            if stored is None or any(t.track_id == line.track_id for t in current):
                continue
            if any(
                intersects(line.bounds, p.painted.bounds)
                or p.painted.lines[0].track_id == -line.track_id
                for p in (*self._visible, *previews)
            ):
                continue
            if any(
                intersects(sample.guard, b.bounds)
                and (
                    len(b.lines) != 1
                    or b.dialogue
                    or b.text != line.text
                    or b.bounds != line.bounds
                )
                for b in retained
            ):
                continue
            # The preview has a display-only identity. It never enters tracking,
            # translation, dialogue history or the set of accepted OCR observations.
            preview = replace(stored.painted, lines=(replace(line, track_id=-line.track_id),))
            previews.append(
                Publication(
                    preview,
                    replace(stored.translation, request=TranslationRequest.from_block(preview)),
                )
            )
        return tuple(previews)

    def cached_previews(
        self,
        blocks: tuple[TextBlock, ...],
        hits: tuple[TranslatedText, ...],
        hidden: frozenset[TextTrack],
        menus: tuple[Publication, ...],
    ) -> tuple[Publication, ...]:
        """Temporary paint never enters retained English or dialogue memory."""
        answers = {hit.request: hit for hit in hits}
        occupied = tuple(p.painted.bounds for p in (*self._published, *menus))
        return tuple(
            Publication(block, answer)
            for block in blocks
            if (answer := answers.get(TranslationRequest.from_block(block))) is not None
            and answer.has_words
            and can_preview_cached(block)
            and not any(t in hidden or t.track_id in self._concealed for t in block.lines)
            and not any(intersects(block.bounds, bounds) for bounds in occupied)
        )

    def update(
        self,
        blocks: tuple[TextBlock, ...],
        translations: tuple[TranslatedText, ...],
        *,
        scope: object = None,
        cleared: frozenset[TextTrack] = frozenset(),
        eligible: frozenset[TranslationRequest] | None = None,
    ) -> tuple[Publication, ...]:
        if scope != self._scope:
            self.reset()
            self._scope = scope
        inputs = (blocks[:128], translations[:128], cleared, eligible)
        if inputs == self._inputs:
            return self._visible
        self._inputs = inputs
        results = {t.request: t for t in inputs[1]}
        current: list[TextBlock] = []
        requests: list[TranslationRequest] = []
        for block in inputs[0]:
            if block.language.contains_text(block.text):
                request = TranslationRequest.from_block(block)
                if eligible is None or request in eligible:
                    current.append(block)
                    requests.append(request)
        blocks = tuple(current)
        self._ready = {
            key: value
            for key in requests
            if (value := results.get(key) or self._ready.get(key)) is not None
        }

        live = {t.track_id: t for b in blocks for t in b.lines}
        owner = {t.track_id: i for i, b in enumerate(blocks) for t in b.lines}
        prior = []
        for p in self._published:
            lines = p.painted.lines
            if not all(t.track_id in live for t in lines):
                continue
            # Once removed from the screen, old English cannot fill the wait for a
            # different source that happens to reuse the same track IDs.
            if all(
                t.track_id not in self._concealed
                or self._concealed[t.track_id] == live[t.track_id].text
                for t in lines
            ):
                prior.append(p)

        # A split can need several replies; a merge can replace several painted
        # labels. Switch the connected old/new blocks together. A slow reply in
        # one paragraph cannot hold an independent label elsewhere on screen.
        successors = [{owner[t.track_id] for t in p.painted.lines} for p in prior]
        overlaps: list[list[int]] = [[] for _ in blocks]
        for old, indices in enumerate(successors):
            for index in indices:
                overlaps[index].append(old)
        waiting = {i for i, key in enumerate(requests) if key not in self._ready}
        pending = list(waiting)
        held: set[int] = set()
        while pending:
            for old in overlaps[pending.pop()]:
                if old in held:
                    continue
                held.add(old)
                pending.extend(successors[old] - waiting)
                waiting.update(successors[old])

        painted = [prior[i].at_live_bounds(live) for i in sorted(held)]
        # A reply without words is complete: it releases prior English but paints nothing.
        painted.extend(
            Publication(block, self._ready[key])
            for i, (block, key) in enumerate(zip(blocks, requests, strict=True))
            if i not in waiting and self._ready[key].has_words
        )
        self._published = tuple(sorted(painted, key=lambda p: p.painted.lines[0].track_id))
        cleared_ids = {t.track_id for t in cleared if live.get(t.track_id) == t}
        # Carry dismissed members through regrouping and the wait for a reply.
        # A partial return cannot restore a whole paragraph.
        waiting_for_return = self._concealed.keys() & cleared_ids
        concealed = {key: live[key].text for key in waiting_for_return}
        visible = []
        for publication in self._published:
            members = publication.painted.lines
            if cleared_ids and (
                all(t.track_id in cleared_ids for t in members)
                or any(t.track_id in waiting_for_return for t in members)
            ):
                concealed.update((t.track_id, live[t.track_id].text) for t in members)
            else:
                visible.append(publication)
        self._concealed = concealed
        self._visible = tuple(visible)
        return self._visible
