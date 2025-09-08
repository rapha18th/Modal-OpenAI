# get_started.py
import modal

image = (
    modal.Image.debian_slim(python_version="3.10")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})  # faster pulls
    .uv_pip_install(
        # Core ML packages
        "transformers>=4.44",
        "torch>=2.3", 
        "accelerate>=0.33",
        "huggingface-hub>=0.24",
        "sentencepiece",
        "safetensors",
        "triton>=3.4.0",   # required for MXFP4 kernels
        "hf_transfer",     # Add hf_transfer for faster downloads
        "kernels",         # MXFP4 fused kernels
    )
)

app = modal.App("time_capsule_infer", image=image)

# Keep the A100 as requested
@app.function(gpu="A100", timeout=900)
def infer(prompt: str, model_id: str = "openai/gpt-oss-20b"):
    from transformers import pipeline

    pipe = pipeline(
        task="text-generation",
        model=model_id,
        torch_dtype="auto",   # uses MXFP4/BF16 as supported
        device_map="auto",
    )

    # Use Harmony chat format via the Transformers chat template
    messages = [{"role": "user", "content": prompt}]
    outputs = pipe(messages, max_new_tokens=256)
    # outputs[0]["generated_text"] is a list of chat turns; last is the assistant
    last = outputs[0]["generated_text"][-1]
    return last["content"] if isinstance(last, dict) and "content" in last else str(last)

# Add HTTP endpoint
@app.function(gpu="A100", timeout=900)
@modal.web_endpoint(method="POST")
def generate_endpoint(request_data: dict):
    """HTTP endpoint for text generation"""
    prompt = request_data.get("prompt", "Tell me a story.")
    model_id = request_data.get("model_id", "openai/gpt-oss-20b")
    
    try:
        result = infer(prompt, model_id)
        return {
            "success": True,
            "generated_text": result,
            "model_used": model_id
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "model_used": model_id
        }

@app.local_entrypoint()
def main(
    prompt: str = "You are a cultural historian. Tell me about Paris in 1920.",
    model_id: str = "openai/gpt-oss-20b",
):
    print(infer.remote(prompt, model_id))
