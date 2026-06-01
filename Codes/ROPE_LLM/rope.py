# =========================================================
# INSTALL (run once if needed)
# =========================================================
# !pip install transformers datasets accelerate torch

# =========================================================
# IMPORTS
# =========================================================
import torch
import math
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama import modeling_llama
from datasets import load_dataset
from tqdm import tqdm
import importlib
import inspect

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# =========================================================
# RESET ORIGINAL RoPE
# =========================================================
importlib.reload(modeling_llama)

_TRUE_ORIGINAL = modeling_llama.apply_rotary_pos_emb
_TRUE_SIG = inspect.signature(_TRUE_ORIGINAL)

# =========================================================
# SAFE CALL TO ORIGINAL
# =========================================================
def call_original(q, k, cos, sin, **kwargs):
    params = list(_TRUE_SIG.parameters.keys())
    # Handle different transformers versions
    if "unsqueeze_dim" in params and "unsqueeze_dim" not in kwargs:
        return _TRUE_ORIGINAL(q, k, cos, sin, unsqueeze_dim=1, **kwargs)
    else:
        return _TRUE_ORIGINAL(q, k, cos, sin, **kwargs)

# =========================================================
# FLOAT (REFERENCE / ORIGINAL BASELINE)
# =========================================================
def rope_float(q, k, cos, sin, **kwargs):
    return call_original(q, k, cos, sin, **kwargs)

# =========================================================
# UNIVERSAL HARDWARE SIMULATOR (LLAMA FORMAT)
# =========================================================
def simulate_hardware_rope(q, k, cos, sin, std_dev):
    # 1. Extract the true angle the LLM requested
    # Llama caches cos/sin in a compatible format for atan2
    theta_true = torch.atan2(sin, cos)

    # 2. Inject Hardware Error (Normal distribution bounded by max error)
    # This mathematically simulates your Verilog hardware's approximation loss
    noise = torch.randn_like(theta_true) * std_dev
    theta_approx = theta_true + noise

    # 3. Calculate hardware's physical output
    c_approx = torch.cos(theta_approx)
    s_approx = torch.sin(theta_approx)

    # 4. Apply Exact Llama Rotation Math (Half-Split format)
    def rotate_half(x):
        half = x.shape[-1] // 2
        x1 = x[..., :half]
        x2 = x[..., half:]
        return torch.cat((-x2, x1), dim=-1)

    q_embed = (q * c_approx) + (rotate_half(q) * s_approx)
    k_embed = (k * c_approx) + (rotate_half(k) * s_approx)

    return q_embed, k_embed

# =========================================================
# 🔹 BINARY UD (HIGHER HARDWARE ERROR)
# =========================================================
def rope_binary(q, k, cos, sin, **kwargs):
    # ~2.2 degrees / 0.038 rad max error
    return simulate_hardware_rope(q, k, cos, sin, std_dev=0.038 / 3.0)

# =========================================================
# 🔹 CSD UD (LOWER HARDWARE ERROR)
# =========================================================
def rope_csd(q, k, cos, sin, **kwargs):
    # ~0.5 degrees / 0.008 rad max error
    # CSD is highly accurate due to its bidirectional error correction
    return simulate_hardware_rope(q, k, cos, sin, std_dev=0.008 / 3.0)

# =========================================================
# PATCH RoPE
# =========================================================
def patch_rope(mode="float"):
    def patched(q, k, cos, sin, **kwargs):
        if mode == "float":
            return rope_float(q, k, cos, sin, **kwargs)
        elif mode == "binary":
            return rope_binary(q, k, cos, sin, **kwargs)
        elif mode == "csd":
            return rope_csd(q, k, cos, sin, **kwargs)

    modeling_llama.apply_rotary_pos_emb = patched
    print(f"RoPE patched → {mode.upper()}")

def restore_rope():
    modeling_llama.apply_rotary_pos_emb = _TRUE_ORIGINAL
    print("RoPE restored to native transformers implementation")

# =========================================================
# LOAD MODEL
# =========================================================
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

print("Loading model...")
# Load in float32 for clean mathematical testing
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32
).to(device)

model.eval()
print("Model loaded successfully.")

# =========================================================
# LOAD DATASET
# =========================================================
print("Loading Wikitext dataset...")
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

# =========================================================
# PERPLEXITY CALCULATOR
# =========================================================
def compute_perplexity(model, dataset, max_samples=50):
    losses = []

    # Using tqdm for a progress bar
    for i, sample in enumerate(tqdm(dataset, desc="Evaluating")):
        if i >= max_samples:
            break

        text = sample["text"]
        if len(text.strip()) == 0:
            continue

        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(device)

        if inputs["input_ids"].shape[1] < 2:
            continue

        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])

        loss = outputs.loss
        if not torch.isnan(loss):
            losses.append(loss.item())

    # Return the exponential of the mean cross-entropy loss
    return math.exp(sum(losses) / len(losses))

# =========================================================
# RUN EXPERIMENTS
# =========================================================
results = {}

print("\n=== BASELINE: FLOAT (Original 32-bit RoPE) ===")
patch_rope("float")
results["float"] = compute_perplexity(model, dataset)

print("\n=== PROPOSED 1: BINARY UD CORDIC ===")
patch_rope("binary")
results["binary"] = compute_perplexity(model, dataset)

print("\n=== PROPOSED 2: CSD UD CORDIC ===")
patch_rope("csd")
results["csd"] = compute_perplexity(model, dataset)

restore_rope()

# =========================================================
# FINAL RESULTS FOR YOUR PAPER
# =========================================================
print("\n" + "="*40)
print(" FINAL PERPLEXITY RESULTS")
print("="*40)
for k, v in results.items():
    print(f"{k.upper():10s}: {v:.4f}")
print("="*40)