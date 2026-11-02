#!/usr/bin/env bash
# Submit the MATH training recipe from the Slime tutorial to an existing Ray node.
set -euo pipefail

: "${FOCUS_HOOKS:?Set FOCUS_HOOKS to the directory containing the tutorial files}"
: "${FOCUS_MODEL:?Set FOCUS_MODEL to the downloaded model directory}"
: "${FOCUS_DATA:?Set FOCUS_DATA to the prepared MATH JSONL file}"
: "${FOCUS_INITIAL:?Set FOCUS_INITIAL to the converted training checkpoint}"
: "${FOCUS_RUN:?Set FOCUS_RUN to the output directory for this experiment}"

cd /root/slime
source scripts/models/qwen3-4B-Instruct-2507.sh

ray job submit --address http://127.0.0.1:8265 \
  --runtime-env "$FOCUS_HOOKS/slime_runtime.yaml" \
  -- python /root/slime/train.py \
  "${MODEL_ARGS[@]}" \
  --train-backend megatron \
  --actor-num-nodes 1 \
  --actor-num-gpus-per-node 2 \
  --colocate \
  --hf-checkpoint "$FOCUS_MODEL" \
  --ref-load "$FOCUS_INITIAL" \
  --save "$FOCUS_RUN/checkpoints" \
  --save-interval 1 \
  --prompt-data "$FOCUS_DATA" \
  --input-key problem \
  --label-key solution \
  --num-rollout 3 \
  --rollout-batch-size 4 \
  --global-batch-size 4 \
  --n-samples-per-prompt 1 \
  --rollout-max-response-len 4096 \
  --rollout-max-context-len 8192 \
  --rollout-temperature 0.7 \
  --rollout-top-p 1.0 \
  --custom-generate-function-path train_slime.generate \
  --custom-reward-post-process-path focus.integrations.slime.preserve_advantages \
  --custom-advantage-function-path focus.integrations.slime.advantages \
  --custom-pg-loss-reducer-function-path focus.integrations.slime.sum_policy_tokens \
  --use-rollout-logprobs \
  --advantage-estimator grpo \
  --loss-type policy_loss \
  --pg-loss-type ppo \
  --eps-clip 0.2 \
  --eps-clip-high 0.2 \
  --kl-coef 0 \
  --entropy-coef 0 \
  --optimizer adam \
  --lr 1e-6 \
  --lr-decay-style constant \
  --weight-decay 0.1 \
  --adam-beta1 0.9 \
  --adam-beta2 0.98 \
  --tensor-model-parallel-size 2 \
  --pipeline-model-parallel-size 1 \
  --context-parallel-size 1 \
  --sequence-parallel \
  --use-dynamic-batch-size \
  --max-tokens-per-gpu 8192 \
  --log-probs-chunk-size 1024 \
  --recompute-granularity full \
  --recompute-method uniform \
  --recompute-num-layers 1 \
  --seq-length 8192 \
  --max-position-embeddings 262144 \
  --bf16 \
  --rollout-num-gpus-per-engine 2 \
  --sglang-mem-fraction-static 0.7 \
  --attention-dropout 0.0 \
  --hidden-dropout 0.0 \
  --accumulate-allreduce-grads-in-fp32 \
  --attention-softmax-in-fp32 \
  --attention-backend flash
