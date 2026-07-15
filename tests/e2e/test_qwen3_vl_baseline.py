import os
import time

# Set XLA flags before any JAX/vLLM imports
os.environ['LIBTPU_INIT_ARGS'] = (
    '--xla_tpu_enable_latency_hiding_scheduler=true '
    '--xla_tpu_enable_async_collective_fusion=true '
    '--xla_tpu_enable_async_collective_fusion_fuse_all_reduce=true '
    '--xla_tpu_overlap_compute_collective_tc=true '
    '--xla_tpu_dvfs_p_state=7 '
    '--xla_tpu_enable_async_collective_fusion_multiple_steps=true '
    '--xla_tpu_enable_all_experimental_scheduler_features=true '
    '--xla_tpu_enable_scheduler_memory_pressure_tracking=true '
    '--xla_tpu_scheduler_percent_shared_memory_limit=100 '
    '--xla_latency_hiding_scheduler_rerun=5 '
    '--xla_lhs_prioritize_async_depth_over_stall=ENABLED '
    + os.environ.get('LIBTPU_INIT_ARGS', '')
)
os.environ['VLLM_XLA_CHECK_RECOMPILATION'] = '0'
os.environ['SKIP_JAX_PRECOMPILE'] = '1'
os.environ['NEW_MODEL_DESIGN'] = '1'
os.environ['HF_HOME'] = '/mnt/data_1tb/prishajain_google_com/huggingface'
os.environ['VLLM_XLA_CACHE_PATH'] = '/mnt/data_1tb/prishajain_google_com/jax_cache_qwen'

import gc
from dataclasses import asdict
from vllm import LLM, EngineArgs, SamplingParams
from vllm.assets.image import ImageAsset
from vllm.multimodal.image import convert_image_mode

def run_inference():
    model = "Qwen/Qwen3-VL-235B-A22B-Thinking-FP8"
    tp_size = 8
    ep_size = 1
    tensor_parallel_size = tp_size * ep_size
    batch_size = 64
    temperature = 0.0
    max_tokens = 128  # Generate 128 tokens for profiling
    max_model_len = 4096
    gpu_memory_utilization = 0.8  # Reduce to avoid OOM in TP4 EP2
    modality = "image"

    print("Preparing for Qwen3-VL inference...")
    image = convert_image_mode(ImageAsset("cherry_blossom").pil_image, "RGB")
    question = "What is the content of this image? pad pad"

    prompt = ("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
              f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
              f"{question}<|im_end|>\n"
              "<|im_start|>assistant\n")

    from vllm.config import ProfilerConfig
    profiler_config = ProfilerConfig(
        profiler="torch",
        torch_profiler_dir="/mnt/data_1tb/prishajain_google_com/profiles",
    )

    engine_args = EngineArgs(
        model=model,
        max_model_len=max_model_len,
        tensor_parallel_size=tensor_parallel_size,
        enable_expert_parallel=(ep_size > 1),  # Enable expert parallel path in vLLM if EP > 1
        gpu_memory_utilization=gpu_memory_utilization,
        max_num_seqs=batch_size,
        mm_processor_kwargs={
            "size": {
                "longest_edge": 1003520,
                "shortest_edge": 3136
            },
            "fps": 1,
        },
        limit_mm_per_prompt={modality: 1},
        trust_remote_code=True,
        profiler_config=profiler_config,
    )
    engine_args = asdict(engine_args)
    if engine_args.get("additional_config") is None:
        engine_args["additional_config"] = {}

    # Configure mesh sharding strategy
    engine_args["additional_config"]["sharding"] = {
        "sharding_strategy": {
            "tensor_parallelism": tp_size,
            "expert_parallelism": ep_size
        }
    }

    # Enable continue decode optimizations (V3)
    engine_args["additional_config"]["enable_continue_decode"] = True
    engine_args["additional_config"]["max_decode_steps"] = 128
    engine_args["async_scheduling"] = False
    
    engine_args["compilation_config"]["cudagraph_capture_sizes"] = []
    
    pass_config = engine_args["compilation_config"].get("pass_config") or {}
    pass_config = {k: v for k, v in pass_config.items() if v is not None}
    pass_config["enable_sp"] = True
    pass_config["sp_min_token_num"] = 1
    pass_config["fuse_gemm_comms"] = False
    pass_config["fuse_allreduce_rms"] = False
    pass_config["fuse_act_padding"] = False
    engine_args["compilation_config"]["pass_config"] = pass_config

    import pprint
    print("Engine args:")
    pprint.pprint(engine_args)
    print("Initializing LLM engine...")
    start_load = time.time()
    llm = LLM(**engine_args)
    print(f"Model loaded in {time.time() - start_load:.2f} seconds")

    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
    )

    inputs = [{
        "prompt": prompt,
        "multi_modal_data": {
            "image": image
        },
    } for _ in range(batch_size)]

    # Run multiple times to measure warm throughput
    num_runs = 5
    for i in range(num_runs):
        print(f"\nRunning inference {i+1}/{num_runs}...")
        
        start_time = time.time()
        outputs = llm.generate(inputs, sampling_params)
        end_time = time.time()
        
        latency = end_time - start_time
        total_output_tokens = sum(len(out.outputs[0].token_ids) for out in outputs)
        throughput = total_output_tokens / latency
        
        print(f"Run {i+1} latency: {latency:.4f} seconds | Throughput: {throughput:.2f} tokens/second | Total Output tokens: {total_output_tokens}")
        if i == 0:
            print("-" * 50)
            print("Generated Text (Run 1, Seq 0):")
            print(outputs[0].outputs[0].text.strip())
            print("-" * 50)

    print("Shutting down engine...")
    llm.llm_engine.engine_core.shutdown()
    gc.collect()

if __name__ == "__main__":
    run_inference()
