from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from focus.environments.browsercomp_plus import (
    BrowserWorkspace,
    RetrievalArchive,
    TraceFind,
    TraceOpen,
    TraceSearch,
    browser_navigation,
)
from focus.harness import InMemoryWorkspace, NullWorkspace, ToolError


class Corpus:
    def __init__(self):
        self.calls = []
        self.docs = {
            "a": {"text": "A museum.\nThe town is Larkhaven.\n"},
            "b": {"text": "An author.\nThe book is Glass Orchard.\n"},
        }

    async def search(self, query, top_k):
        self.calls.append((query, top_k))
        order = ["a", "b"] if query == "museum" else ["b", "a"]
        return [
            SimpleNamespace(docid=docid, rank=i + 1, score=1)
            for i, docid in enumerate(order[:top_k])
        ]

    async def documents(self, docids):
        return {docid: self.docs[docid] for docid in docids if docid in self.docs}


async def test_shared_tools_keep_navigation_separate_across_forks():
    archive = RetrievalArchive(Corpus())
    search, open_, find = TraceSearch(archive), TraceOpen(archive), TraceFind(archive)
    root = BrowserWorkspace()
    await search.call({"query": "museum"}, workspace=root)
    await open_.call({"id": 0}, workspace=root)
    left, right = root.fork(), root.fork()
    await asyncio.gather(
        search.call({"query": "author"}, workspace=left),
        open_.call({"loc": 1}, workspace=right),
    )
    second = json.loads(await open_.call({"id": 0}, workspace=left))
    assert second["docid"] == "b"
    assert second["cursor"] == 1
    found = json.loads(
        await find.call({"pattern": "LARKHAVEN", "cursor": 0}, workspace=left)
    )
    assert found["matches"][0]["line"] == 1
    assert browser_navigation(left)["current"] == 1
    assert browser_navigation(root) == browser_navigation(right)
    assert browser_navigation(root)["pages"] == ["a"]
    assert browser_navigation(left)["pages"] == ["a", "b"]
    # External inspection cannot mutate a saved branch.
    snapshot = browser_navigation(left)
    snapshot["pages"].clear()
    assert browser_navigation(left)["pages"] == ["a", "b"]
    restored = BrowserWorkspace.deserialize(left.serialize())
    assert browser_navigation(restored) == browser_navigation(left)
    assert restored.tools() == ()
    assert "page cursors" in restored.compaction_guidance()


async def test_custom_limits_and_generic_workspace():
    client = Corpus()
    archive = RetrievalArchive(client)
    workspace = InMemoryWorkspace()
    search = TraceSearch(archive, top_k=1, snippet_chars=5)
    result = json.loads(await search.call({"query": "museum"}, workspace=workspace))
    assert client.calls == [("museum", 1)]
    assert len(result["hits"]) == 1
    assert result["hits"][0]["preview"] == "A mus"
    assert result["hits"][0]["id"] == 0
    await search.call({"query": "museum", "topn": 2}, workspace=workspace)
    assert browser_navigation(workspace)["results"] == ["a", "b"]
    open_ = TraceOpen(archive, read_chars=200)
    result = json.loads(
        await open_.call({"id": 0, "num_lines": 1}, workspace=workspace)
    )
    assert len(result["lines"]) == 1
    assert result["next_line"] == 1
    reread = json.loads(
        await open_.call({"loc": result["next_line"]}, workspace=workspace)
    )
    assert reread["lines"][0]["line"] == 1
    assert reread["cursor"] == result["cursor"]


async def test_invalid_arguments_and_missing_documents_preserve_current_page():
    client = Corpus()
    archive = RetrievalArchive(client)
    workspace = BrowserWorkspace()
    open_, find = TraceOpen(archive), TraceFind(archive)
    with pytest.raises(ToolError, match="No page"):
        await open_.call({}, workspace=workspace)
    with pytest.raises(ToolError, match="Open a page"):
        await find.call({"pattern": "museum"}, workspace=workspace)
    await TraceSearch(archive).call({"query": "museum"}, workspace=workspace)
    await open_.call({"id": 0}, workspace=workspace)
    saved = browser_navigation(workspace)
    for args in [{"id": 2}, {"id": True}, {"num_lines": 0}, {"loc": -2}]:
        with pytest.raises(ToolError):
            await open_.call(args, workspace=workspace)
    assert browser_navigation(workspace) == saved
    del client.docs["b"]
    missing = json.loads(await open_.call({"id": 1}, workspace=workspace))
    assert missing["not_found"]
    assert browser_navigation(workspace) == saved


async def test_missing_workspace_fails_before_retrieval():
    client = Corpus()
    search = TraceSearch(RetrievalArchive(client))
    for workspace in [None, NullWorkspace()]:
        with pytest.raises(TypeError, match="BrowserWorkspace"):
            await search.call({"query": "museum"}, workspace=workspace)
    assert client.calls == []


async def test_transport_failure_does_not_replace_search_results():
    class Broken(Corpus):
        async def search(self, query, top_k):
            if query == "broken":
                raise ConnectionError("offline")
            return await super().search(query, top_k)

    archive = RetrievalArchive(Broken())
    search = TraceSearch(archive)
    workspace = BrowserWorkspace()
    await search.call({"query": "museum"}, workspace=workspace)
    saved = browser_navigation(workspace)
    with pytest.raises(ConnectionError, match="offline"):
        await search.call({"query": "broken"}, workspace=workspace)
    assert browser_navigation(workspace) == saved
    assert isinstance(archive.failure, ConnectionError)


@pytest.mark.parametrize("chars", [0, -1, True])
def test_invalid_preview_limits(chars):
    with pytest.raises(ValueError, match="snippet_chars"):
        TraceSearch(RetrievalArchive(), snippet_chars=chars)
