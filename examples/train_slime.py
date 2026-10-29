"""Slime hooks for the MATH tutorial.

Point Slime's generate and evaluation functions at this module, and its reward,
advantage, and loss-reducer hooks at ``focus.integrations.slime``.
"""
import os

from focus.integrations import JsonlDirectory
from focus.integrations.slime import make_evaluate, make_generate
from math_training import collector_for

# Critic records leave Slime's loop: one JSONL file per collected prompt.
critic_sink = JsonlDirectory(os.environ.get("FOCUS_CRITIC_OUTPUT", "focus-critic-targets"))

# The model is used without tools; Segments reads each reply as a token chunk.
generate = make_generate(collector_for, profile="token_chunk", critic_sink=critic_sink)
evaluate = make_evaluate(collector_for, profile="token_chunk")
