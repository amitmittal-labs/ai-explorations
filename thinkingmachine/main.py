import streamlit as st
import torch
import torch.nn as nn
import tiktoken
import math
import os

MODEL_PATHS = {
    "Pretrained": "./models/pretrain_latest.pth",
    "SFT": "./models/sft_latest.pth",
    "GRPO": "./models/grpo_latest.pth"
}


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
# 2. PROMPT TEMPLATES (Matching YOUR Training Data)
# ==========================================
PROMPT_TEMPLATES = {
    "Story Generation": {
        "template": "{}",
        "placeholder": "Once upon a time, there was a brave knight",
        "description": "Natural story continuation (Pretrained on TinyStories)",
        "best_for": "Pretrained",
        "trained_on": "TinyStories dataset"
    },
    "Alpaca Instruction (SFT)": {
        "template": "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{}\n\n### Response:\n",
        "placeholder": "Explain how photosynthesis works in simple terms",
        "description": "EXACT Alpaca format used in SFT training",
        "best_for": "SFT",
        "trained_on": "Alpaca dataset (52k instructions)"
    },
    "Math Reasoning (GSM8K/GRPO)": {
        "template": "Question: {}\nSolve step by step.\n<think>\n",
        "placeholder": "Sarah has 15 apples. She gives 1/3 to her friend and then buys 8 more. How many apples does she have now?",
        "description": "EXACT GSM8K format with <think> tags (SFT + GRPO)",
        "best_for": "GRPO",
        "trained_on": "GSM8K math problems + GRPO optimization"
    },
    "OpenOrca Style (SFT)": {
        "template": "{}\n\n",
        "placeholder": "You are an AI assistant. User will give you a task. Your goal is to complete the task faithfully.\n\nExplain the water cycle",
        "description": "System prompt + question format (OpenOrca)",
        "best_for": "SFT",
        "trained_on": "OpenOrca dataset (50k samples)"
    },
    "Educational Content": {
        "template": "{}",
        "placeholder": "The process of evaporation occurs when",
        "description": "Natural text continuation (Pretrained on FineWeb-edu)",
        "best_for": "Pretrained",
        "trained_on": "FineWeb-edu dataset"
    },
    "Simple Math (GRPO)": {
        "template": "Question: {}\nSolve step by step.\n<think>\n",
        "placeholder": "What is 25 multiplied by 4?",
        "description": "Simple arithmetic with reasoning (GRPO optimized)",
        "best_for": "GRPO",
        "trained_on": "GSM8K with GRPO reinforcement"
    },
    "Code Generation (SFT)": {
        "template": "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{}\n\n### Response:\n",
        "placeholder": "Write a Python function to calculate the factorial of a number",
        "description": "Programming tasks via Alpaca format",
        "best_for": "SFT",
        "trained_on": "Alpaca dataset (includes code tasks)"
    },
    "Custom Prompt": {
        "template": "{}",
        "placeholder": "Enter your own prompt here",
        "description": "Use your own custom prompt format",
        "best_for": "Any",
        "trained_on": "N/A"
    }
}

# ==========================================
# 3. MODEL ARCHITECTURE
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

@st.cache_resource
def load_resources():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = tiktoken.get_encoding("gpt2")
    return device, tokenizer

@st.cache_resource
def load_all_models(device):
    """Loads all three models and returns them in a dictionary."""
    models = {}
    status = {}
    
    for name, path in MODEL_PATHS.items():
        if not os.path.exists(path):
            models[name] = None
            status[name] = f"Not found: {path}"
            continue
        
        try:
            model = ThinkingMachine(TINY_TM_CONFIG).to(device)
            checkpoint = torch.load(path, map_location=device)
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            model.load_state_dict(state_dict, strict=False)
            model.eval()
            models[name] = model
            status[name] = f"Loaded successfully"
        except Exception as e:
            models[name] = None
            status[name] = f"Error: {str(e)[:50]}"
    
    return models, status

def generate_complete(model, tokenizer, device, prompt, max_tokens, temp, top_k):
    """Generates text and returns the complete result."""
    if model is None:
        return "Model not loaded"

    input_ids = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0).to(device)
    curr_text = prompt
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

            if next_token.item() == eos_id:
                break

            input_ids = torch.cat((input_ids, next_token), dim=1)
            word = tokenizer.decode([next_token.item()])
            curr_text += word

    return curr_text

# ==========================================
# 5. STREAMLIT UI LAYOUT
# ==========================================
st.set_page_config(layout="wide", page_title="Thinking Machine Comparison")

st.title("Thinking Machine: Model Comparison Tool")
st.markdown("Compare **Pretrained**, **SFT**, and **GRPO** models with different prompts")

device, tokenizer = load_resources()
models, status = load_all_models(device)

# ==========================================
# TOP SECTION: Model Status
# ==========================================
st.subheader("Model Status")
col_status = st.columns(3)
for idx, (name, msg) in enumerate(status.items()):
    with col_status[idx]:
        if "✅" in msg:
            st.success(f"**{name}**\n{msg}")
        else:
            st.error(f"**{name}**\n{msg}")

st.markdown("---")

# ==========================================
# CONFIGURATION SECTION
# ==========================================
st.subheader("Configuration")

config_col1, config_col2 = st.columns([2, 1])

with config_col1:
    # Template Selection
    template_choice = st.selectbox(
        "Select Prompt Template:",
        list(PROMPT_TEMPLATES.keys()),
        help="Choose a template that matches your use case"
    )
    
    template_info = PROMPT_TEMPLATES[template_choice]
    
    # Show template info
    st.info(f"**Description:** {template_info['description']}\n\n**Best for:** {template_info['best_for']}\n\n**Trained on:** {template_info['trained_on']}")

with config_col2:
    # Model Selection
    st.markdown("**Select Models to Run:**")
    
    run_pretrained = st.checkbox("Pretrained", value=True, disabled=models["Pretrained"] is None)
    run_sft = st.checkbox("SFT", value=True, disabled=models["SFT"] is None)
    run_grpo = st.checkbox("GRPO", value=True, disabled=models["GRPO"] is None)
    
    # Generation Settings
    st.markdown("**Generation Settings:**")
    temp = st.slider("Temperature", 0.1, 2.0, 0.8, 0.1, help="Higher = more creative")
    max_len = st.slider("Max Tokens", 50, 500, 200, 10)
    top_k = st.slider("Top-K", 1, 100, 40, 1, help="Limits sampling to top K tokens")

st.markdown("---")

# ==========================================
# PROMPT INPUT SECTION
# ==========================================
st.subheader("✏️ Enter Your Prompt")

if template_choice == "Custom Prompt":
    user_input = st.text_area(
        "Custom Prompt:",
        height=150,
        value=template_info['placeholder'],
        help="Enter any custom prompt",
        key="user_input"
    )
    final_prompt = user_input
else:
    user_input = st.text_area(
        f"Fill in the template:",
        height=120,
        value=template_info['placeholder'],
        help=f"This will be inserted into the template below",
        key="user_input"
    )
    final_prompt = template_info['template'].format(user_input)
    
    # Show formatted prompt
    with st.expander("🔍 Preview Full Prompt (What models will see)"):
        st.code(final_prompt, language="text")
        st.caption(f"Template: `{template_info['template'][:80]}...`")

# Generate Button
generate_button = st.button("Generate Responses", type="primary", use_container_width=True)

st.markdown("---")

# ==========================================
# RESULTS SECTION - TABBED VIEW
# ==========================================
if generate_button:
    st.subheader("Generated Responses")
    
    # Determine which models to run
    models_to_run = []
    if run_pretrained and models["Pretrained"]: models_to_run.append("Pretrained")
    if run_sft and models["SFT"]: models_to_run.append("SFT")
    if run_grpo and models["GRPO"]: models_to_run.append("GRPO")
    
    if not models_to_run:
        st.warning("Please select at least one model to run!")
    else:
        # Create tabs for each selected model
        tabs = st.tabs([f"{name}" for name in models_to_run])
        
        # Generate for each model in its own tab
        for idx, model_name in enumerate(models_to_run):
            with tabs[idx]:
                with st.spinner(f"Generating with {model_name}..."):
                    result = generate_complete(
                        models[model_name], 
                        tokenizer, 
                        device, 
                        final_prompt, 
                        max_len, 
                        temp, 
                        top_k
                    )
                
                # Display results in larger text areas
                st.markdown(f"### Input Prompt")
                st.text_area(
                    "Prompt sent to model:",
                    value=final_prompt,
                    height=150,
                    key=f"prompt_{model_name}",
                    disabled=True
                )
                
                st.markdown(f"### Generated Output")
                generated_only = result[len(final_prompt):]
                st.text_area(
                    "Generated text:",
                    value=generated_only,
                    height=300,
                    key=f"output_{model_name}",
                    disabled=True
                )
                
                st.markdown(f"### Complete Response")
                st.text_area(
                    "Full response (prompt + generated):",
                    value=result,
                    height=400,
                    key=f"complete_{model_name}",
                    disabled=True
                )
                
                # Token count
                token_count = len(tokenizer.encode(generated_only))
                st.caption(f"Generated {token_count} tokens")

# ==========================================
# SIDEBAR - HELP & EXAMPLES
# ==========================================
with st.sidebar:
    st.header("ℹModel Information")
    st.markdown("""
    **Pretrained:**
    - Trained on TinyStories + FineWeb-edu
    - Good at: Story generation, text completion
    - No instruction following
    
    **SFT (Supervised Fine-Tuning):**
    - Trained on: Alpaca (52k) + GSM8K + OpenOrca (50k)
    - Good at: Following instructions, Q&A, code
    - Formats: `### Instruction:` and `Question: ... <think>`
    
    **GRPO (Reinforcement Learning):**
    - Optimized on GSM8K math reasoning
    - Good at: Step-by-step thinking, math problems
    - Uses `<think>` tags for chain-of-thought
    """)
    
    st.markdown("---")
    
    with st.expander("💡 Example Prompts"):
        st.markdown("""
        ### For Pretrained:
        **Template:** Story Generation
        - "Once upon a time, in a magical forest there lived"
        - "The young scientist discovered that water"
        
        ### For SFT:
        **Template:** Alpaca Instruction
        - "Write a short poem about nature"
        - "Explain photosynthesis in simple terms"
        
        ### For GRPO:
        **Template:** Math Reasoning
        - "A store sells apples for $0.50 each. If you buy 12 apples and pay with a $10 bill, how much change do you get?"
        - "John has 3 times as many marbles as Sarah. Together they have 48 marbles. How many does John have?"
        """)
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
    <small>Thinking Machine v1.0</small>
    </div>
    """, unsafe_allow_html=True)