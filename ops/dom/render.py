"""Checkpoint-encoding rendering and exact token counts for `local:` chat rows (self-traces).

Extracted verbatim from the original track's transcript converter (not part of this release), so that
`selftraces.py` computes `n_tokens` exactly as s02's `Renderer.text_ids` will: the checkpoint's own
encoding/encoding.py renders the OpenAI-format messages (DSML tool calls, <think> reasoning) and the
checkpoint tokenizer counts them.

CPU only. Never imports torch.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ops.lib import env  # noqa: E402

_TOK = None
_E = None


def tok():
    global _TOK
    if _TOK is None:
        from ops.lib.tok import Tok
        _TOK = Tok(env.SRC)
    return _TOK


def enc_mod():
    global _E
    if _E is None:
        if str(env.ENCODING) not in sys.path:
            sys.path.insert(0, str(env.ENCODING))
        import encoding as E
        _E = E
    return _E


def render(messages: list[dict], thinking: bool) -> str:
    """The exact call s02's Renderer.text_ids makes (minus tokenization)."""
    return enc_mod().encode_messages(messages, thinking_mode="thinking" if thinking else "chat")


def n_tokens(messages: list[dict], thinking: bool) -> int:
    return len(tok().encode(render(messages, thinking)))
