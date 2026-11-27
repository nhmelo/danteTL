"""Explicit-time replay of visual retention against an owned analyzer."""

from dataclasses import replace

from dantetl.ocr.observation import CandidateReview
from dantetl.ocr.presence import prepare_evidence
from dantetl.ocr.tracking import TextTracker
from dantetl.ocr.visual import VisualAnalyzer


class Replay:
    def __init__(self, first):
        self.visual = VisualAnalyzer()
        self.tracker = TextTracker()
        self.result = first
        self.ids = self.tracker.observe(
            first, review=CandidateReview(pixels=prepare_evidence(first))
        )
        self.tracks = self.tracker.advance(first.frame, first.completed_at + 0.5).confirmed
        self.visual.observe(first.frame, first, self.tracks, self.ids, candidate=first)

    def step(self, incoming):
        state = self.visual.observe(
            incoming.frame,
            self.result,
            self.tracks,
            self.ids,
            self.tracker.snapshot().activity,
            incoming,
        )
        self.ids = self.tracker.observe(
            incoming,
            review=replace(
                state.review, pixels=prepare_evidence(incoming, self.tracker.evidence_bounds())
            ),
            presence=state.presence,
        )
        self.tracks = self.tracker.advance(incoming.frame, incoming.completed_at + 0.5).confirmed
        self.result = incoming
        self.visual.observe(
            incoming.frame,
            incoming,
            self.tracks,
            self.ids,
            self.tracker.snapshot().activity,
            incoming,
        )
        return state
