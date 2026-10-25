"""Search passes, tree-aware training preparation, and rollout building blocks."""

from focus.allocation import (
    Allocator,
    BestFirstAllocator,
    FifoAllocator,
    RolloutLedger,
    TokenEntropyAllocator,
    ValueRefinement,
    register_allocator,
)
from focus.codec import ChatCodec, Message, TextCodec
from focus.config import (
    RolloutConfig,
    RolloutSpec,
    SearchPass,
    VinePpoConfig,
)
from focus.execution import AdmissionPool, ExecutionServices, LocalInferenceExecutor
from focus.execution.budget import ResourceBudget
from focus.export import format_edges, format_tree, to_samples
from focus.generation import (
    GenerateResult,
    OpenAIChatPolicy,
    OpenAIResponsesPolicy,
    Policy,
    PolicyFormat,
    PreparedPrompt,
    SamplingParams,
    SGLangPolicy,
    SlimePolicy,
    TinkerPolicy,
    TokenChunk,
    TokenChunkParser,
    TokenContinuationPolicy,
    TokenId,
    VLLMPolicy,
    register_policy,
)
from focus.generation.parsing import (
    HermesToolCallParser,
    NativeToolCallParser,
    QwenXMLToolCallParser,
    ToolTurn,
)
from focus.harness import (
    SUBMIT,
    AgenticCompactor,
    AnyOf,
    BaseTool,
    Compactor,
    Env,
    InMemoryWorkspace,
    NullWorkspace,
    PromptTokens,
    RolloutState,
    RoundBudget,
    Runner,
    StateCounter,
    StepResult,
    SubmitFields,
    TokenScorer,
    ToolEnv,
    ToolError,
    Trigger,
    Turn,
    Workspace,
    build_budget,
    regions,
)
from focus.harness.checkpoint import CheckpointPolicy, EveryDecision, SparseCheckpoints
from focus.loop import Runtime, collect_samples, run_search
from focus.preparation import (
    AdvantageEstimator,
    DirectBranchTdEstimator,
    GrpoEstimator,
    RefinedTdEstimator,
    VinePpoEstimator,
    prepare_samples,
    register_estimator,
)
from focus.preparation.records import ActorSample, CriticSample, PreparedBatch
from focus.protocols import AppendOnly, ContextProtocol, Summary
from focus.registry import (
    build,
    construct_target,
    register,
    registrable,
)
from focus.reward import (
    AsyncRewardModel,
    RewardModel,
    RewardResult,
    VLLMRewardModel,
    register_reward,
)
from focus.reward.config import RewardConfig
from focus.scheduler import (
    AnchorProposal,
    ConcurrencyGating,
    GatingPolicy,
    MixtureProposal,
    ProposalPolicy,
    SchedulerView,
    WidthGating,
)
from focus.scheduler.core.allocation import (
    AllocationLedger,
    AllocationSession,
    BestFirstSession,
    FifoSession,
    RankedSession,
    Reservation,
)
from focus.scheduler.core.entropy import TokenEntropySession
from focus.tiktoken_chat import TiktokenChatTokenizer

# Grouped by the concern a reader has when they reach for the name, and sorted inside
# each group. The groups are the documentation: they are the order the docs introduce
# the library in, which is not the order an alphabetical list would give.
__all__ = [
    "BestFirstAllocator",
    "TokenEntropyAllocator",
    "RolloutLedger",
    "AllocationSession",
    "AllocationLedger",
    "Reservation",
    "FifoSession",
    "BestFirstSession",
    "RankedSession",
    "TokenEntropySession",
    "SearchPass",
    "VinePpoConfig",
    "FifoAllocator",
    "register_allocator",
    "prepare_samples",
    "ExecutionServices",
    "AdmissionPool",
    "LocalInferenceExecutor",
    "ResourceBudget",
    "CheckpointPolicy",
    "EveryDecision",
    "SparseCheckpoints",
    "ContextProtocol",
    "AppendOnly",
    "Summary",
    # The one call, what it is given, and what it hands back.
    "ActorSample",
    "CriticSample",
    "PreparedBatch",
    "Runtime",
    "SchedulerView",
    "format_edges",
    "format_tree",
    "run_search",
    "collect_samples",
    "to_samples",
    # The env you write, and the runner that drives it.
    "BaseTool",
    "Env",
    "RewardConfig",
    "Policy",
    "PolicyFormat",
    "PreparedPrompt",
    "RolloutState",
    "RoundBudget",
    "Runner",
    "SUBMIT",
    "StepResult",
    "SubmitFields",
    "TokenScorer",
    "ToolEnv",
    "ToolError",
    "ToolTurn",
    "Turn",
    "regions",
    # What the agent writes to, and what folds the context when it grows.
    "AgenticCompactor",
    "Compactor",
    "AnyOf",
    "InMemoryWorkspace",
    "NullWorkspace",
    "PromptTokens",
    "StateCounter",
    "Trigger",
    "Workspace",
    "build_budget",
    # Talking to a model: the tool dialect it speaks, the codec, the backends.
    "ChatCodec",
    "GenerateResult",
    "HermesToolCallParser",
    "Message",
    "NativeToolCallParser",
    "OpenAIChatPolicy",
    "OpenAIResponsesPolicy",
    "QwenXMLToolCallParser",
    "SamplingParams",
    "SGLangPolicy",
    "SlimePolicy",
    "TextCodec",
    "TiktokenChatTokenizer",
    "TinkerPolicy",
    "TokenChunk",
    "TokenChunkParser",
    "TokenContinuationPolicy",
    "TokenId",
    "VLLMPolicy",
    "register_policy",
    # Scoring a finished rollout with a reward model.
    "AsyncRewardModel",
    "RewardModel",
    "RewardResult",
    "VLLMRewardModel",
    "register_reward",
    # The shape of the search: how far an expansion rolls, how wide, under which law.
    "AnchorProposal",
    "GatingPolicy",
    "ProposalPolicy",
    "ConcurrencyGating",
    "MixtureProposal",
    "WidthGating",
    # What an edge is worth to the actor loss.
    "AdvantageEstimator",
    "Allocator",
    "RefinedTdEstimator",
    "DirectBranchTdEstimator",
    "GrpoEstimator",
    "VinePpoEstimator",
    "ValueRefinement",
    "register_estimator",
    # Saying all of the above in a config file, and building it from a name.
    "RolloutConfig",
    "RolloutSpec",
    "build",
    "construct_target",
    "register",
    "registrable",
]
