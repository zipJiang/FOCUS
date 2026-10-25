"""TRACE browser tools with result IDs and branch-local page cursors.

Compose these tools with any ToolEnv. Each rollout must receive a BrowserWorkspace
or another writable Workspace; tools share retrieval observations through an archive
while keeping navigation state in the workspace. The direct-document-ID primitives
in browser.py remain available for tools with other navigation interfaces.
"""

from __future__ import annotations

import json

from focus.harness import BaseTool, InMemoryWorkspace, ToolError

from .browser import BrowserFind, BrowserOpen, BrowserSearch, integer

NAVIGATION = "browser/navigation"


class BrowserWorkspace(InMemoryWorkspace):
    """Checkpointed navigation, with no extra model-facing tools."""

    def tools(self):
        return ()

    def guidance(self):
        return ""

    def compaction_guidance(self):
        return (
            "Preserve result IDs, page cursors, line citations, "
            "and unresolved questions."
        )


def browser_navigation(workspace):
    """Read a detached navigation snapshot: results, pages, current, and retrieved.

    ``results`` maps the latest search result IDs to corpus document IDs. ``pages``
    maps persistent page cursors to document IDs; ``current`` selects a page or is
    -1 before the first open. ``retrieved`` is the union of search hits for reporting.
    The workspace owns this state and copies it when a rollout forks.
    """
    if not all(
        callable(getattr(workspace, name, None)) for name in ("exists", "read", "write")
    ):
        raise TypeError(
            "TRACE tools require BrowserWorkspace or a writable key-value workspace"
        )
    if workspace.exists(NAVIGATION):
        return json.loads(workspace.read(NAVIGATION))
    return {"results": [], "pages": [], "current": -1, "retrieved": []}


class TraceSearch(BaseTool):
    """Search the corpus and assign result IDs local to the current branch.

    A new search replaces the result list. Previously opened page cursors remain
    valid. ``top_k`` sets the default result count; ``snippet_chars`` bounds each
    preview. The model can override the count with ``topn`` (1–20).
    """

    name = "browser.search"
    description = "Search the corpus. Open a returned zero-based result id."
    parameters = BrowserSearch.parameters

    def __init__(self, archive, *, top_k=10, snippet_chars=700):
        if type(snippet_chars) is not int or snippet_chars < 1:
            raise ValueError("snippet_chars must be a positive integer")
        self.search = BrowserSearch(archive, top_k=top_k, snippet_chars=snippet_chars)

    async def call(self, params, *, workspace=None, **kwargs):
        nav = browser_navigation(workspace)
        result = json.loads(await self.search.call(params))
        nav["results"] = [hit["docid"] for hit in result["hits"]]
        nav["retrieved"] = list(dict.fromkeys(nav["retrieved"] + nav["results"]))
        for index, hit in enumerate(result["hits"]):
            hit["id"] = index
        workspace.write(NAVIGATION, json.dumps(nav))
        return json.dumps(result, ensure_ascii=False)


class TraceOpen(BaseTool):
    """Open a result ID or reread the current page, recording a stable page cursor.

    ``id=-1`` selects the current page; ``loc=-1`` starts at line zero. Reads default
    to 20 lines and are capped by ``read_chars``. Only successful opens change the
    current page. Shared tool instances keep all navigation in the given workspace.
    """

    name = "browser.open"
    description = (
        "Read a search result. id=-1 rereads the current page; loc=-1 starts at line 0."
    )
    parameters = {
        "type": "object",
        "properties": {
            "id": {"type": "integer", "default": -1},
            "loc": {"type": "integer", "default": -1},
            "num_lines": {"type": "integer", "default": -1},
        },
        "additionalProperties": False,
    }

    def __init__(self, archive, *, read_chars=6000):
        self.open = BrowserOpen(archive, read_chars=read_chars)

    async def call(self, params, *, workspace=None, **kwargs):
        args = self.verify_args(params)
        nav = browser_navigation(workspace)
        index = integer(args, "id", -1, -1, 2**31 - 1)
        if index == -1:
            if nav["current"] == -1:
                raise ToolError("No page is open. Search, then open a result id.")
            docid = nav["pages"][nav["current"]]
        else:
            if index >= len(nav["results"]):
                raise ToolError("id is outside the most recent search results")
            docid = nav["results"][index]
        loc = integer(args, "loc", -1, -1, 2**31 - 1)
        count = integer(args, "num_lines", -1, -1, 60)
        result = json.loads(
            await self.open.call(
                {
                    "docid": docid,
                    "loc": max(0, loc),
                    "num_lines": 20 if count == -1 else count,
                }
            )
        )
        if not result.get("not_found"):
            if docid not in nav["pages"]:
                nav["pages"].append(docid)
            nav["current"] = nav["pages"].index(docid)
            result["cursor"] = nav["current"]
            workspace.write(NAVIGATION, json.dumps(nav))
        return json.dumps(result, ensure_ascii=False)


class TraceFind(BaseTool):
    """Find literal text in a page selected by its branch-local cursor.

    ``cursor=-1`` selects the current page. Matching is case-insensitive and returns
    up to 20 matches with line numbers and snippets. Finding in another page leaves
    the current page unchanged.
    """

    name = "browser.find"
    description = (
        "Find literal text in an opened page; cursor=-1 uses the current page."
    )
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string"},
            "cursor": {"type": "integer", "default": -1},
        },
        "required": ["pattern"],
        "additionalProperties": False,
    }

    def __init__(self, archive):
        self.find = BrowserFind(archive)

    async def call(self, params, *, workspace=None, **kwargs):
        args = self.verify_args(params)
        nav = browser_navigation(workspace)
        cursor = integer(args, "cursor", -1, -1, 2**31 - 1)
        cursor = nav["current"] if cursor == -1 else cursor
        if not 0 <= cursor < len(nav["pages"]):
            raise ToolError("Open a page before using its cursor.")
        result = json.loads(
            await self.find.call(
                {
                    "docid": nav["pages"][cursor],
                    "pattern": args["pattern"],
                }
            )
        )
        result["cursor"] = cursor
        return json.dumps(result, ensure_ascii=False)
