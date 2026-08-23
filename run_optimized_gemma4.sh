#!/bin/bash
# Optimized Gemma 4 Workload Runner
# Includes Phase 4 Latency Hiding Scheduler and Phase 5 SparseCore Offload Thresholding

export USE_BATCHED_RPA_SEQ_ON_LANE=1
export USE_FUSED_DENSE_MLP_KERNEL=1

# 1. Latency Hiding Scheduler (Phase 4)
# Explicitly overlaps the newly decoupled TP All-Reduces from the MoE and Dense MLP branches
export LIBTPU_INIT_ARGS="--xla_tpu_enable_latency_hiding_scheduler=true \
--xla_tpu_overlap_compute_collective_tc=true \
--xla_tpu_enable_async_collective_fusion=true \
--xla_tpu_enable_async_collective_fusion_multiple_steps=true \
--xla_tpu_enable_all_experimental_scheduler_features=true "

# 2. Advanced SparseCore Coprocessor Offloading (Phase 5)
# Offloads the hoisted All-Reduces to the Ghostfish SparseCore, but strictly 
# thresholds at 1MB to prevent overhead regressions on small scalar collectives.
export LIBTPU_INIT_ARGS="${LIBTPU_INIT_ARGS} --xla_tpu_enable_sc_allreduce=true \
--xla_tpu_sc_allreduce_threshold_bytes=1048576"

echo "Executing optimized Gemma 4 workload with advanced compiler flags..."
# Placeholder for the actual underlying execution script command (e.g., benchmark.sh)
# ./tests/e2e/benchmarking/benchmark.sh
