import json
import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class FakeLLM:
    """Routes prompts to canned replies by substring; records calls. embed() can be made to fail."""

    def __init__(self, replies: dict[str, object] | None = None, embed_ok: bool = False):
        self.replies = replies or {}
        self.embed_ok = embed_ok
        self.calls: list[str] = []

    def chat(self, model, prompt, *, system="", images=None, json_mode=False, max_tokens=700):
        self.calls.append(prompt)
        for needle, reply in self.replies.items():
            if needle in prompt:
                return reply if isinstance(reply, str) else json.dumps(reply)
        return "{}"

    def embed(self, model, texts):
        if not self.embed_ok:
            raise RuntimeError("this server does not support embeddings")
        return [[float(len(t) % 7), 1.0, 0.0] for t in texts]


@pytest.fixture
def fake_llm():
    return FakeLLM()


@pytest.fixture
def make_image(tmp_path):
    def _make(name="img.png", size=(400, 300), color=(200, 60, 60), mode="RGB"):
        path = tmp_path / name
        Image.new(mode, size, color).save(path)
        return str(path)
    return _make
