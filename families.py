"""Model families and their serving setups. Each setup is one vLLM configuration of the same model.
  bf16 is always the reference, and fp16 is a control that changes arithmetic without quantizing.
  'bnb' setups quantize in flight with bitsandbytes, so they need no third-party checkpoint.
Run check_models.py on the GPU machine first: it confirms every repository exists and that your token can reach it.
"""
FAMILIES = {
    "qwen2.5-7b": {
        "bf16":      dict(model="Qwen/Qwen2.5-7B-Instruct", quantization=None, dtype="bfloat16"),
        "fp16":      dict(model="Qwen/Qwen2.5-7B-Instruct", quantization=None, dtype="float16"),
        "fp8":       dict(model="Qwen/Qwen2.5-7B-Instruct", quantization="fp8", dtype="bfloat16"),
        "int8-gptq": dict(model="Qwen/Qwen2.5-7B-Instruct-GPTQ-Int8", quantization=None, dtype="float16"),
        "int4-gptq": dict(model="Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4", quantization=None, dtype="float16"),
        "int4-awq":  dict(model="Qwen/Qwen2.5-7B-Instruct-AWQ", quantization=None, dtype="float16"),
    },
    # Meta's checkpoint is gated: accept the license on Hugging Face and set HF_TOKEN.
    "llama3.1-8b": {
        "bf16":       dict(model="meta-llama/Meta-Llama-3.1-8B-Instruct", quantization=None, dtype="bfloat16"),
        "fp16":       dict(model="meta-llama/Meta-Llama-3.1-8B-Instruct", quantization=None, dtype="float16"),
        "fp8":        dict(model="meta-llama/Meta-Llama-3.1-8B-Instruct", quantization="fp8", dtype="bfloat16"),
        "int8-w8a16": dict(model="RedHatAI/Meta-Llama-3.1-8B-Instruct-quantized.w8a16", quantization=None, dtype="float16"),
        "int4-gptq":  dict(model="hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4", quantization=None, dtype="float16"),
        "int4-awq":   dict(model="hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4", quantization=None, dtype="float16"),
        "int4-bnb":   dict(model="meta-llama/Meta-Llama-3.1-8B-Instruct", quantization="bitsandbytes", dtype="bfloat16"),
    },
    # The model that Zhu et al. test RUT on. Gated: accept the terms on Hugging Face and set HF_TOKEN.
    "mistral-7b": {
        "bf16":     dict(model="mistralai/Mistral-7B-Instruct-v0.3", quantization=None, dtype="bfloat16"),
        "fp16":     dict(model="mistralai/Mistral-7B-Instruct-v0.3", quantization=None, dtype="float16"),
        "fp8":      dict(model="mistralai/Mistral-7B-Instruct-v0.3", quantization="fp8", dtype="bfloat16"),
        "int4-bnb": dict(model="mistralai/Mistral-7B-Instruct-v0.3", quantization="bitsandbytes", dtype="bfloat16"),
    },
    # Meta's Muse Glimmer, released openly under Apache 2.0, so no license approval is needed. About 30B parameters with a
    # vision encoder; BF16 needs over 55 GB, so use an 80 GB GPU. Meta's own quantized builds are GGUF and ExecuTorch,
    # which vLLM providers do not serve, so the quantized setups here are vLLM's own FP8 and bitsandbytes 4-bit.
    "muse-glimmer-30b": {
        "bf16":     dict(model="meta-models/Muse-Glimmer-30B", quantization=None, dtype="bfloat16"),
        "fp16":     dict(model="meta-models/Muse-Glimmer-30B", quantization=None, dtype="float16"),
        "fp8":      dict(model="meta-models/Muse-Glimmer-30B", quantization="fp8", dtype="bfloat16"),
        "int4-bnb": dict(model="meta-models/Muse-Glimmer-30B", quantization="bitsandbytes", dtype="bfloat16"),
    },
    # A larger model. BF16 needs about 66 GB, so use an 80 GB A100 or H100.
    "qwen2.5-32b": {
        "bf16":      dict(model="Qwen/Qwen2.5-32B-Instruct", quantization=None, dtype="bfloat16"),
        "fp8":       dict(model="Qwen/Qwen2.5-32B-Instruct", quantization="fp8", dtype="bfloat16"),
        "int8-gptq": dict(model="Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8", quantization=None, dtype="float16"),
        "int4-gptq": dict(model="Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4", quantization=None, dtype="float16"),
        "int4-awq":  dict(model="Qwen/Qwen2.5-32B-Instruct-AWQ", quantization=None, dtype="float16"),
    },
}
NAMES = {"bf16": "BF16", "fp16": "FP16", "fp8": "FP8", "int8-gptq": "INT8 GPTQ", "int8-w8a16": "INT8 W8A16",
         "int4-gptq": "INT4 GPTQ", "int4-awq": "INT4 AWQ", "int4-bnb": "INT4 bitsandbytes"}
ORDER = list(NAMES)

def configs(family):
    if family not in FAMILIES:
        raise SystemExit(f"unknown family {family}; choose from {', '.join(FAMILIES)}")
    return FAMILIES[family]

def data_dir(family):
    """Qwen 7B keeps its original folder so every published number still reproduces."""
    return "data-nocache" if family == "qwen2.5-7b" else f"data-{family}"
