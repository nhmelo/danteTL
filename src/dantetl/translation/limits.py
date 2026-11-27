"""Text bounds shared by grouping, dialogue context, prompts and every translation backend."""

MAX_SOURCE_CHARACTERS = 512
"""Characters in one request's complete source text."""

MAX_CONTEXT_TURNS = 2
MAX_CONTEXT_CHARACTERS = 512
"""Source plus English characters across the recent turns sent with a request."""

MAX_ENGLISH_CHARACTERS = 4096
"""Visible answer characters accepted from any backend."""
