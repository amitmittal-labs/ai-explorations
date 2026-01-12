import streamlit as st
import torch
import torch.nn as nn
import tiktoken
import math
import os
import copy

# ==========================================
# 1. MODEL CONFIGURATION & ARCHITECTURE
# ==========================================
TINY_TM_CONFIG = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 384,
    "n_heads": 6,
    "n_layers": 6,
    "drop_rate": 0.1,
    "qkv_bias": False
}

# --- Model Components (Copy-Paste from Training Script) ---
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
# 2. HELPER FUNCTIONS
# ==========================================

@st.cache_resource
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_tokenizer():
    return tiktoken.get_encoding("gpt2")

def load_weights(model, path):
    if not os.path.exists(path):
        st.error(f"❌ Checkpoint not found: {path}")
        return False
    
    try:
        checkpoint = torch.load(path, map_location='cpu')
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        
        # Handle "Surgery" logic for old 128 context models if needed
        if state_dict['pos_emb.weight'].shape[0] != TINY_TM_CONFIG['context_length']:
            st.warning(f"⚠️ Model {path} has old context length. Auto-resizing...")
            old_pos = state_dict['pos_emb.weight']
            new_pos = torch.randn(TINY_TM_CONFIG['context_length'], TINY_TM_CONFIG['emb_dim']) * 0.02
            new_pos[:old_pos.shape[0], :] = old_pos
            state_dict['pos_emb.weight'] = new_pos
            
        model.load_state_dict(state_dict, strict=False)
        return True
    except Exception as e:
        st.error(f"Error loading {path}: {e}")
        return False

def generate_text(model, tokenizer, prompt, max_tokens, temp, top_k):
    device = get_device()
    model.to(device)
    model.eval()
    
    input_ids = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0).to(device)
    
    # Progress bar for generation
    progress_text = st.empty()
    full_text = prompt
    
    with torch.no_grad():
        for _ in range(max_tokens):
            idx_cond = input_ids[:, -TINY_TM_CONFIG["context_length"]:]
            logits = model(idx_cond)[:, -1, :] / temp
            
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
                
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            if next_token.item() == 50256: # EOS
                break
                
            input_ids = torch.cat((input_ids, next_token), dim=1)
            
            # Decode just the new token to stream it
            new_word = tokenizer.decode([next_token.item()])
            full_text += new_word
            # Update UI occasionally to be fast
            if _ % 2 == 0:
                progress_text.markdown(f"**Generating:** {full_text} ...")

    return full_text

# ==========================================
# 3. STREAMLIT UI
# ==========================================

st.set_page_config(page_title="Thinking Machine Playground", layout="wide")

st.title("🧠 Thinking Machine: The Evolution")
st.markdown("""
This app demonstrates the evolution of a Tiny Language Model (28M Params) through 4 stages of training:
1. **Pre-trained:** Only knows simple stories.
2. **Healed:** Knows basic logic/grammar but still raw.
3. **SFT (SFT):** Follows instructions and formats.
4. **Reasoning (GRPO):** Actually thinks step-by-step.
""")

# --- Sidebar Controls ---
st.sidebar.header("⚙️ Settings")

# Checkpoint Paths (Adjust these to match your Drive paths)
CHECKPOINTS = {
    "1. Pre-trained (TinyStories)": "/content/drive/MyDrive/ThinkingMachine_Checkpoints/pretrained_model.pth",
    "2. Healed (Mixed Data)": "/content/drive/MyDrive/ThinkingMachine_Checkpoints/pretrained_healed_model.pth",
    "3. Instruction Tuned (SFT)": "/content/drive/MyDrive/ThinkingMachine_Checkpoints/final_instr_finetuned_model.pth",
    "4. Reasoning (GRPO)": "/content/drive/MyDrive/ThinkingMachine_Checkpoints/reasoning_model_grpo.pth"
}

selected_mode = st.sidebar.radio("Mode", ["Single Model", "Compare All Models"])
selected_model_name = st.sidebar.selectbox("Select Model", list(CHECKPOINTS.keys())) if selected_mode == "Single Model" else None

temp = st.sidebar.slider("Temperature", 0.1, 2.0, 0.8, 0.1)
max_tokens = st.sidebar.slider("Max Output Tokens", 10, 500, 150, 10)
top_k = st.sidebar.slider("Top-K Sampling", 1, 100, 40, 5)

# --- Main Interface ---

default_prompt = "Question: If I have 3 apples and eat 1, how many do I have? Answer:"
user_prompt = st.text_area("Enter your prompt:", value=default_prompt, height=100)
generate_btn = st.button("🚀 Generate Response")

if generate_btn:
    tokenizer = load_tokenizer()
    model = ThinkingMachine(TINY_TM_CONFIG)
    
    if selected_mode == "Single Model":
        path = CHECKPOINTS[selected_model_name]
        st.info(f"Loading {selected_model_name}...")
        
        if load_weights(model, path):
            st.success("Weights Loaded. Generating...")
            output = generate_text(model, tokenizer, user_prompt, max_tokens, temp, top_k)
            st.markdown("### 🤖 Model Output")
            st.markdown(f"```\n{output}\n```")
            
    else: # Compare All Mode
        st.info("Running Prompt through ALL models sequentially...")
        
        cols = st.columns(len(CHECKPOINTS))
        
        for idx, (name, path) in enumerate(CHECKPOINTS.items()):
            with cols[idx]:
                st.subheader(name.split(" ")[1]) # Short name
                if load_weights(model, path):
                    output = generate_text(model, tokenizer, user_prompt, max_tokens, temp, top_k)
                    
                    # Highlight <think> tags if present
                    if "<think>" in output:
                        output = output.replace("<think>", "<span style='color:blue'><b><think></b>").replace("</think>", "<b></think></b></span>")
                        st.markdown(output, unsafe_allow_html=True)
                    else:
                        st.text(output)
                else:
                    st.error("Failed to load")
