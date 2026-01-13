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
PRETRAINED_PATH = "./models/pretrained.pth"

# Model Architecture Config (Must match training)
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
# 2. MODEL ARCHITECTURE (The "Brain")
# ==========================================
# We include the class definitions so the app is self-contained.

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
        self.ff = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"])
        )
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
# 3. HELPER FUNCTIONS
# ==========================================

@st.cache_resource
def load_resources():
    """Loads tokenizer and device only once."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = tiktoken.get_encoding("gpt2")
    return device, tokenizer

def load_model_weights(model, path):
    
    if not os.path.exists(path):
        return False, f"File not found: {path}"

    try:
        checkpoint = torch.load(path, map_location='cpu')
        # Support both full checkpoint dicts and direct state_dicts
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint

        # Load logic
        model.load_state_dict(state_dict, strict=False)
        return True, "Loaded successfully"
    except Exception as e:
        return False, str(e)

def generate_text(model, tokenizer, device, prompt, max_tokens, temp, top_k):
    """Generates text from the model."""
    model.eval()
    input_ids = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0).to(device)

    # Placeholder for the UI to update in real-time
    output_container = st.empty()
    generated_text = prompt

    with torch.no_grad():
        for _ in range(max_tokens):
            # Crop context if it gets too long
            idx_cond = input_ids[:, -TINY_TM_CONFIG["context_length"]:]

            # Forward pass
            logits = model(idx_cond)[:, -1, :] / temp

            # Top-K Sampling
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Stop if EOS
            if next_token.item() == 50256:
                break

            input_ids = torch.cat((input_ids, next_token), dim=1)

            # Decode and Update UI
            word = tokenizer.decode([next_token.item()])
            generated_text += word
            output_container.markdown(f"**Output:**\n\n{generated_text}")

    return generated_text


st.set_page_config(page_title="Thinking Machine", layout="centered")

st.title("🧠 Thinking Machine (V1)")
st.caption("Currently running: **Pre-trained Model** (TinyStories only)")

# --- Sidebar: Configuration ---
st.sidebar.header("⚙️ Settings")
temp = st.sidebar.slider("Temperature (Creativity)", 0.1, 2.0, 0.8, 0.1)
max_tokens = st.sidebar.slider("Max Output Tokens", 10, 500, 100, 10)
top_k = st.sidebar.slider("Top-K Sampling", 1, 100, 40, 5)

st.sidebar.markdown("---")
st.sidebar.info("Tip: Lower temperature makes the model more deterministic. Higher makes it more creative.")

# --- Main Area ---
device, tokenizer = load_resources()

# Load Model logic
@st.cache_resource
def get_model():
    model = ThinkingMachine(TINY_TM_CONFIG).to(device)
    success, msg = load_model_weights(model, PRETRAINED_PATH)
    if not success:
        st.error(f"❌ Error loading weights: {msg}")
        return None
    return model

model = get_model()

if model:
    st.success("Model loaded and ready on " + str(device))

    user_prompt = st.text_area("Enter your prompt:", value="Once upon a time, there was a little girl named", height=150)

    if st.button("🚀 Generate"):
        if not user_prompt.strip():
            st.warning("Please enter a prompt first.")
        else:
            with st.spinner("Thinking..."):
                final_output = generate_text(model, tokenizer, device, user_prompt, max_tokens, temp, top_k)
            st.success("Generation Complete!")