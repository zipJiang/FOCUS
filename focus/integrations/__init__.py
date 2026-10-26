"""Adapters to training frameworks. The engine ends at data; these hand it over.

Every integration asks the user for the same thing -- a
:data:`~focus.integrations.common.CollectorFor` that builds the collector for one
:class:`~focus.integrations.common.Problem` under the trainer's current policy -- and
sends critic records to a :data:`~focus.integrations.common.CriticSink`.
:mod:`focus.integrations.tinker` supplies a training loop you run;
:mod:`focus.integrations.slime` supplies hooks Slime calls.
"""

from focus.integrations.common import (
    CollectorFor,
    CriticSink,
    JsonlDirectory,
    Problem,
    collect_records,
)

__all__ = [
    "Problem",
    "CollectorFor",
    "CriticSink",
    "JsonlDirectory",
    "collect_records",
]
