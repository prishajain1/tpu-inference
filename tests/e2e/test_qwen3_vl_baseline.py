import os
import time

# Set XLA flags before any JAX/vLLM imports
os.environ['LIBTPU_INIT_ARGS'] = (
    '--xla_tpu_enable_latency_hiding_scheduler=true '
    '--xla_tpu_enable_async_collective_fusion=true '
    '--xla_tpu_enable_async_collective_fusion_fuse_all_reduce=true '
    '--xla_tpu_overlap_compute_collective_tc=true '
    '--xla_tpu_dvfs_p_state=7'
)
os.environ['VLLM_XLA_CHECK_RECOMPILATION'] = '0'
os.environ['SKIP_JAX_PRECOMPILE'] = '1'

import gc
from dataclasses import asdict
from vllm import LLM, EngineArgs, SamplingParams
from vllm.assets.image import ImageAsset
from vllm.multimodal.image import convert_image_mode

def run_inference():
    model = "Qwen/Qwen3-VL-8B-Instruct"
    tensor_parallel_size = 8
    temperature = 0.0
    max_tokens = 128  # Generate 128 tokens for profiling
    max_model_len = 4096
    gpu_memory_utilization = 0.8  # Increase utilization for 8B model
    modality = "image"

    print("Preparing for Qwen3-VL inference...")
    image = convert_image_mode(ImageAsset("cherry_blossom").pil_image, "RGB")
    question = "What is the content of this image? pad pad"

    prompt = ("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
              f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
              f"{question}<|im_end|>\n"
              "<|im_start|>assistant\n")

    engine_args = EngineArgs(
        model=model,
        max_model_len=max_model_len,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_num_seqs=1,
        mm_processor_kwargs={
            "size": {
                "longest_edge": 1003520,
                "shortest_edge": 3136
            },
            "fps": 1,
        },
        limit_mm_per_prompt={modality: 1},
    )
    engine_args = asdict(engine_args)
    if engine_args.get("additional_config") is None:
        engine_args["additional_config"] = {}

    # Enable continue decode optimizations (V3)
    engine_args["additional_config"]["enable_continue_decode"] = True
    engine_args["additional_config"]["max_decode_steps"] = 128
    engine_args["async_scheduling"] = False
    
    engine_args["compilation_config"]["cudagraph_capture_sizes"] = []
    
    pass_config = engine_args["compilation_config"].get("pass_config") or {}
    pass_config = {k: v for k, v in pass_config.items() if v is not None}
    pass_config["enable_sp"] = True
    pass_config["sp_min_token_num"] = 1
    engine_args["compilation_config"]["pass_config"] = pass_config

    print("Initializing LLM engine...")
    start_load = time.time()
    llm = LLM(**engine_args)
    print(f"Model loaded in {time.time() - start_load:.2f} seconds")

    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
    )

    inputs = {
        "prompt": prompt,
        "multi_modal_data": {
            "image": image
        },
    }

    # Run multiple times to measure warm throughput
    num_runs = 5
    for i in range(num_runs):
        print(f"\nRunning inference {i+1}/{num_runs}...")
        
        # Start profiling on Run 3 (steady state)
        is_profile_run = (i == 2)
        if is_profile_run:
            print("Starting profile...")
            
        start_time = time.time()
        outputs = llm.generate(inputs, sampling_params)
        end_time = time.time()
        
        latency = end_time - start_time
        generated_text = outputs[0].outputs[0].text.strip()
        num_output_tokens = len(outputs[0].outputs[0].token_ids)
        throughput = num_output_tokens / latency
        
        print(f"Run {i+1} latency: {latency:.4f} seconds | Throughput: {throughput:.2f} tokens/second | Output tokens: {num_output_tokens}")
        if i == 0:
            print("-" * 50)
            print("Generated Text (Run 1):")
            print(generated_text)
            print("-" * 50)

    print("Shutting down engine...")
    llm.llm_engine.engine_core.shutdown()
    gc.collect()

if __name__ == "__main__":
    run_inference()
