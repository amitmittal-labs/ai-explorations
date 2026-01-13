import streamlit as st
import torch
import torch.nn as nn
import tiktoken
import math
import os

# ==========================================
# 1. CONFIGURATION
# ==========================================
# Path to your FIRST model (Pre-trained)
# Update this path to where your file actually is!
MODEL_PATH_1 = "./models/pretrained.pth"
MODEL_PATH_2 = "./models/pretrained_healed_model.pth"

TINY_TM_CONFIG = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 384,
    "n_heads": 6,
    "n_layers": 6,
    "drop_rate": 0.1,
    "qkv_bias": False
}

# ==========================================
# 2. MODEL ARCHITECTURE
# ==========================================
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        return self.scale * (x - mean) / torch.sqrt(var + self.eps) + self.shift

class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

class MultiHeadAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_heads = cfg["n_heads"]
        self.head_dim = cfg["emb_dim"] // cfg["n_heads"]
        self.W_q = nn.Linear(cfg["emb_dim"], cfg["emb_dim"], bias=cfg["qkv_bias"])
        self.W_k = nn.Linear(cfg["emb_dim"], cfg["emb_dim"], bias=cfg["qkv_bias"])
        self.W_v = nn.Linear(cfg["emb_dim"], cfg["emb_dim"], bias=cfg["qkv_bias"])
        self.out_proj = nn.Linear(cfg["emb_dim"], cfg["emb_dim"])
        self.register_buffer("mask", torch.triu(torch.ones(cfg["context_length"], cfg["context_length"]), diagonal=1))

    def forward(self, x):
        b, seq, _ = x.shape
        q = self.W_q(x).view(b, seq, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.W_k(x).view(b, seq, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.W_v(x).view(b, seq, self.n_heads, self.head_dim).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores.masked_fill_(self.mask.bool()[:seq, :seq], -float("inf"))
        weights = torch.softmax(scores, dim=-1)
        return self.out_proj((weights @ v).transpose(1, 2).contiguous().view(b, seq, -1))

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(cfg)
        self.ff = nn.Sequential(nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]), GELU(), nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]))
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
    def forward(self, x):
        x = x + self.att(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x

class ThinkingMachine(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.blocks = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)
    def forward(self, idx):
        seq = idx.shape[1]
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(seq, device=idx.device))
        x = self.blocks(x)
        return self.out_head(self.final_norm(x))

# ==========================================
# 3. UTILITIES
# ==========================================
@st.cache_resource
def load_resources():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = tiktoken.get_encoding("gpt2")
    return device, tokenizer

@st.cache_resource
def load_model(path, device):
    """Loads a model instance from a specific path."""
    if not os.path.exists(path):
        return None, f"Path not found: {path}"

    model = ThinkingMachine(TINY_TM_CONFIG).to(device)
    try:
        checkpoint = torch.load(path, map_location=device)
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        return model, "OK"
    except Exception as e:
        return None, str(e)

def generate_stream(model, tokenizer, device, prompt, max_tokens, temp, top_k, placeholder):
    """Generates text and updates a Streamlit placeholder in real-time."""
    if model is None:
        placeholder.error("Model not loaded.")
        return

    input_ids = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0).to(device)
    curr_text = prompt
      # ID for <|endoftext|>
    eos_id = tokenizer.encode('<|endoftext|>', allowed_special={'<|endoftext|>'})[0]

    with torch.no_grad():
        for step in range(max_tokens):
            idx_cond = input_ids[:, -TINY_TM_CONFIG["context_length"]:]
            logits = model(idx_cond)[:, -1, :] / temp

            if top_k is not None:
                v = torch.topk(logits, min(top_k, logits.size(-1))).values
                logits[logits < v[:, [-1]]] = -float('Inf')

            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            if next_token.item() == eos_id: break

            input_ids = torch.cat((input_ids, next_token), dim=1)
            word = tokenizer.decode([next_token.item()])
            curr_text += word

            # Update UI
            if step % 2 == 0:
               placeholder.markdown(f"**Generating:**\n{curr_text}")

    placeholder.markdown(f"**Final Output:**\n{curr_text}")

# ==========================================
# 4. STREAMLIT UI LAYOUT
# ==========================================
st.set_page_config(layout="wide", page_title="Thinking Machine")

st.title("🤖 Thinking Machine Duel")
device, tokenizer = load_resources()

# Sidebar Settings
st.sidebar.header("⚙️ Settings")
mode = st.sidebar.radio("Comparison Mode", ["Same Prompt (Compare)", "Separate Prompts (Independent)"])
temp = st.sidebar.slider("Temperature", 0.1, 1.5, 0.8)
max_len = st.sidebar.slider("Max Tokens", 50, 500, 150)

# Load Models
col1, col2 = st.columns(2)

with col1:
    st.subheader("Model 1 (e.g. Pre-trained)")
    model1, msg1 = load_model(MODEL_PATH_1, device)
    if model1: st.success(f"Loaded: {os.path.basename(MODEL_PATH_1)}")
    else: st.error(f"Error: {msg1}")

with col2:
    st.subheader("Model 2 (e.g. Fine-tuned)")
    model2, msg2 = load_model(MODEL_PATH_2, device)
    if model2: st.success(f"Loaded: {os.path.basename(MODEL_PATH_2)}")
    else: st.error(f"Error: {msg2}")

st.markdown("---")

# Input Area
if mode == "Same Prompt (Compare)":
    common_prompt = st.text_area("Enter Prompt for BOTH models:", height=100, value="Once upon a time")
    if st.button("🚀 Generate Both"):
        c1, c2 = st.columns(2)
        with c1:
            box1 = st.empty()
            generate_stream(model1, tokenizer, device, common_prompt, max_len, temp, 40, box1)
        with c2:
            box2 = st.empty()
            generate_stream(model2, tokenizer, device, common_prompt, max_len, temp, 40, box2)

else: # Independent Mode
    c1, c2 = st.columns(2)
    with c1:
        prompt1 = st.text_area("Prompt for Model 1:", height=100, value="Once upon a time")
        if st.button("Generate Model 1"):
            box1 = st.empty()
            generate_stream(model1, tokenizer, device, prompt1, max_len, temp, 40, box1)

    with c2:
        prompt2 = st.text_area("Prompt for Model 2:", height=100, value="Question: 2+2? <think>")
        if st.button("Generate Model 2"):
            box2 = st.empty()
            generate_stream(model2, tokenizer, device, prompt2, max_len, temp, 40, box2)