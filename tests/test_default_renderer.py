from __future__ import annotations

import pytest

from renderers.default import DefaultRenderer


class _UserRequiredTokenizer:
    eos_token_id = None
    all_special_tokens: list[str] = []

    def apply_chat_template(self, messages, **kwargs):
        if not any(message.get("role") == "user" for message in messages):
            raise ValueError("No user query found in messages.")
        text = "|".join(
            f"{message.get('role', '')}:{message.get('content', '')}"
            for message in messages
        )
        if kwargs.get("add_generation_prompt"):
            text += "|assistant:"
        return list(range(len(text)))

    def decode(self, token_ids, skip_special_tokens=False):
        return "".join(chr(65 + (token_id % 26)) for token_id in token_ids)


class _BrokenUserTokenizer(_UserRequiredTokenizer):
    def apply_chat_template(self, messages, **kwargs):
        raise ValueError("No user query found in messages.")


def test_default_renderer_skips_system_only_prefix():
    renderer = DefaultRenderer(_UserRequiredTokenizer())
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Do the task."},
    ]

    rendered = renderer.render(messages, add_generation_prompt=True)

    assert rendered.token_ids
    assert len(rendered.message_indices) == len(rendered.token_ids)
    assert set(rendered.message_indices) <= {1, -1}


def test_default_renderer_does_not_hide_user_prefix_errors():
    renderer = DefaultRenderer(_BrokenUserTokenizer())
    messages = [{"role": "user", "content": "Do the task."}]

    with pytest.raises(ValueError, match="No user query found"):
        renderer.render(messages)
