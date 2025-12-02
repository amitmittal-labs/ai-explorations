# Demystifying Reinforcement Learning: From Snake Game to LLMs

---

## Introduction: Why Reinforcement Learning Matters

Reinforcement Learning (RL) is fundamentally different from supervised learning. Instead of learning from labeled examples, an agent learns by **interacting with an environment** and receiving rewards or penalties for its actions.

Think about it:
- **Supervised Learning**: "Here's a cat image. Label it as 'cat'." ✅ Direct feedback
- **Reinforcement Learning**: "Try moving around. You'll figure out what's good." 🎮 Delayed feedback

This makes RL perfect for:
- 🎮 **Games** (Snake, Chess, Go)
- 🤖 **Robotics** (Walking, grasping)
- 💬 **LLMs** (Training models to be helpful via RLHF)
- 🚗 **Autonomous systems** (Self-driving cars)

Today, I'll walk you through **PPO (Proximal Policy Optimization)**, one of the most popular RL algorithms, using a **Snake game**. We'll demystify the math with real calculations and see how it applies to different scenarios.

---

## The RL Framework: Core Concepts

### The Agent-Environment Loop

```
┌─────────┐
│  Agent  │ (Your Snake AI)
└────┬────┘
     │ Action (turn left/right/straight)
     ↓
┌─────────────┐
│ Environment │ (Game world)
└─────┬───────┘
      │ State + Reward
      ↓
   Observation (danger ahead? food location?)
```

**Key Components:**

1. **State (s)**: What the agent observes
   - Snake game: `[danger_straight, danger_right, danger_left, direction, food_location]`

2. **Action (a)**: What the agent does
   - Snake game: `{0: straight, 1: turn_right, 2: turn_left}`

3. **Reward (r)**: Feedback signal
   - Snake game: `+10` for food, `-10` for dying, `-0.01` per step

4. **Policy (π)**: Agent's strategy
   - Maps states to actions: π(a|s)

5. **Value Function (V)**: Expected cumulative reward
   - "How good is this state?"

---

## Snake Game Implementation

### State Representation (11 dimensions)

```python
observation = [
    danger_straight,  # 1 if collision ahead
    danger_right,     # 1 if collision to right
    danger_left,      # 1 if collision to left
    dir_up,           # 1 if moving up
    dir_right,        # 1 if moving right
    dir_down,         # 1 if moving down
    dir_left,         # 1 if moving left
    food_up,          # 1 if food is above
    food_down,        # 1 if food is below
    food_left,        # 1 if food is on left
    food_right        # 1 if food is on right
]
```

**Example state:**
```python
state = [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1]
# Interpretation:
# - Danger straight ahead 
# - Moving right →
# - Food is above ↑ and to the right →
```

### Reward Function

```python
class RewardCalculator:
    def calculate_reward(self, done, self_hit, had_food):
        reward = 0.0
        
        # Alive penalty (encourages efficiency)
        if not done: 
            reward += -0.01
        
        # Food reward
        if had_food:
            reward += 10.0
        
        # Death penalty
        if done:
            if self_hit:  # Hit own body
                reward += -20.0  # Double penalty
            else:  # Hit wall
                reward += -10.0
        
        return reward
```

**Sample trajectory with rewards:**

| Step | Action | State | Event | Reward | Cumulative |
|------|--------|-------|-------|--------|------------|
| 1 | straight | safe | alive | -0.01 | -0.01 |
| 2 | right | safe | alive | -0.01 | -0.02 |
| 3 | straight | food ahead | **ate food** | +9.99 | +9.97 |
| 4 | left | danger ahead | alive | -0.01 | +9.96 |
| 5 | straight | wall ahead | **died** | -10.01 | -0.05 |

---

## PPO: The Algorithm Behind the Magic

### The Core Idea

PPO solves a fundamental problem in RL: **How do we improve our policy without making drastic changes that break everything?**

**The Challenge:**
- Too small updates → Learning is slow 🐌
- Too large updates → Policy becomes unstable 💥

**PPO's Solution:**
Clip the policy updates to stay within a "trust region"

### Mathematical Foundation

#### 1. The Policy Network (Actor)

Our 2-layer neural network:

```
State (11) → Hidden (128) → Actions (3)
           ReLU         Softmax
```

**Forward pass:**
```python
def forward(state):
    h = ReLU(W1 @ state + b1)     # Hidden layer
    logits = W2 @ h + b2           # Output layer
    probs = softmax(logits)        # Action probabilities
    return probs
```

**Example:**
```python
state = [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1]
probs = policy_net(state)
# Output: [0.1, 0.7, 0.2]  # [straight, right, left]
# Agent likely chooses "turn right"
```

#### 2. The Value Network (Critic)

Estimates how good a state is:

```
State (11) → Hidden (128) → Value (1)
           ReLU
```

```python
def forward(state):
    h = ReLU(W1 @ state + b1)
    value = W2 @ h + b2
    return value  # Scalar value
```

**Example:**
```python
state_safe = [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1]
V(state_safe) = 5.2  # Good state

state_danger = [1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1]
V(state_danger) = -2.3  # Bad state
```

---

## The Math: Step by Step

### Step 1: Calculate Returns (G)

**Discounted cumulative reward:**

$$G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + ... = \sum_{k=0}^{\infty} \gamma^k r_{t+k}$$

Where γ (gamma) = 0.99 (discount factor)

**Real example from my Snake game:**

| t | Reward (r) | Calculation | Return (G) |
|---|-----------|-------------|------------|
| 5 | -10.01 | -10.01 | -10.01 |
| 4 | -0.01 | -0.01 + 0.99×(-10.01) | -9.92 |
| 3 | +9.99 | 9.99 + 0.99×(-9.92) | 0.17 |
| 2 | -0.01 | -0.01 + 0.99×(0.17) | 0.16 |
| 1 | -0.01 | -0.01 + 0.99×(0.16) | 0.15 |

**Code implementation:**
```python
def calculate_returns(rewards, gamma=0.99):
    returns = []
    running_return = 0
    
    for r in reversed(rewards):
        running_return = r + gamma * running_return
        returns.insert(0, running_return)
    
    return returns
```

### Step 2: Calculate Advantages (A)

**Advantage = How much better is this action than average?**

$$A_t = G_t - V(s_t)$$

**Intuition:**
- A > 0: This action was **better than expected** ✅
- A < 0: This action was **worse than expected** ❌
- A ≈ 0: This action was **as expected** 😐

**Real calculation:**

| State | G (Return) | V (Value) | A (Advantage) | Meaning |
|-------|-----------|-----------|---------------|---------|
| s₁ | 0.15 | 0.20 | -0.05 | Slightly worse |
| s₂ | 0.16 | 0.18 | -0.02 | Slightly worse |
| s₃ | 0.17 | -1.50 | **+1.67** | Much better! 🎉 |
| s₄ | -9.92 | -5.00 | -4.92 | Much worse |
| s₅ | -10.01 | -8.00 | -2.01 | Worse |

**After normalization:**
```python
advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
# Normalized: [-0.34, -0.29, +1.82, -0.89, -0.30]
```

**Why normalize?**
- Stabilizes training
- Prevents one large advantage from dominating

### Step 3: PPO Clipped Objective

**The magic formula:**

$$L^{CLIP}(\theta) = \mathbb{E}_t[\min(r_t(\theta) A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t)]$$

Where:
- $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$ (probability ratio)
- $\epsilon = 0.2$ (clipping parameter)

**Breaking it down:**

**Step 3a: Calculate ratio**

```python
old_log_prob = -1.2  # log π_old(a|s)
new_log_prob = -0.8  # log π_new(a|s)

ratio = exp(new_log_prob - old_log_prob)
      = exp(-0.8 - (-1.2))
      = exp(0.4)
      = 1.49
```

**Step 3b: Apply clipping**

```python
advantage = 1.67  # From our example above

# Unclipped loss
loss_unclipped = ratio * advantage
               = 1.49 * 1.67
               = 2.49

# Clipped loss
clipped_ratio = clip(1.49, 1-0.2, 1+0.2)  # clip(1.49, 0.8, 1.2)
              = 1.2  # Capped at 1.2
loss_clipped = 1.2 * 1.67
             = 2.00

# Final loss (take minimum - pessimistic)
loss = min(2.49, 2.00) = 2.00
```

**Visual representation:**

```
Ratio (r)
    ↑
1.2 |----[Clipped region]----
    |         ✓ Safe updates
1.0 |------------------------  No change
    |         
0.8 |----[Clipped region]----
    |
    └─────────────────────→ Advantage (A)
```

**Why clip?**
- **Positive advantage (A > 0)**: Don't increase probability too much
- **Negative advantage (A < 0)**: Don't decrease probability too much
- **Result**: Stable, conservative updates

### Step 4: Value Loss

Simple Mean Squared Error:

$$L^{VF} = \frac{1}{2}(V_\theta(s_t) - G_t)^2$$

**Example:**
```python
predicted_value = 0.20  # V(s₁)
target_return = 0.15    # G₁

value_loss = 0.5 * (0.20 - 0.15)²
           = 0.5 * 0.0025
           = 0.00125
```

## Training Loop: Putting It All Together

```python
# Hyperparameters
n_steps = 2048        # Collect 2048 transitions
epochs = 10           # 10 PPO updates per batch
gamma = 0.99          # Discount factor
clip_epsilon = 0.2    # PPO clipping

for iteration in range(1000):
    # 1. Collect trajectories
    for step in range(n_steps):
        action, log_prob, value = agent.select_action(state)
        next_state, reward, done, info = env.step(action)
        
        # Store transition
        memory.store(state, action, log_prob, reward, value)
        
        if done:
            state = env.reset()
        else:
            state = next_state
    
    # 2. Calculate returns and advantages
    returns = calculate_returns(memory.rewards, gamma)
    advantages = returns - memory.values
    advantages = normalize(advantages)
    
    # 3. PPO update (multiple epochs)
    for epoch in range(epochs):
        # Get new predictions
        new_log_probs = policy_net.get_log_prob(memory.states, memory.actions)
        new_values = value_net(memory.states)
        
        # Calculate losses
        policy_loss = ppo_loss(new_log_probs, memory.log_probs, 
                               advantages, clip_epsilon)
        value_loss = mse_loss(new_values, returns)
        
        # Update policy network
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        # Update value network
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()
    
    # 4. Clear memory
    memory.clear()
```

**Training progression (my Snake game):**

| Iteration | Avg Score | Avg Length | Policy Loss | Value Loss |
|-----------|-----------|------------|-------------|------------|
| 0 | 0.2 | 5 | 2.45 | 8.23 |
| 100 | 1.5 | 12 | 1.82 | 4.15 |
| 200 | 3.8 | 28 | 1.23 | 2.34 |
| 500 | 8.2 | 45 | 0.67 | 1.12 |
| 1000 | 12.5 | 68 | 0.34 | 0.56 |

---

## Comparing RL Across Different Domains

Now that we understand PPO with Snake, let's see how it differs in other applications:

### 1. Snake Game (Step-by-Step Rewards)

**Characteristics:**
- ✅ **Dense rewards**: Every action gets immediate feedback
- ✅ **Fast episodes**: Games end in 30-200 steps
- ✅ **Clear objectives**: Eat food, avoid death

**Reward structure:**
```python
rewards_per_episode = [
    -0.01,  # step 1: alive
    -0.01,  # step 2: alive
    +9.99,  # step 3: ate food!
    -0.01,  # step 4: alive
    -10.01  # step 5: died
]
```

**Learning curve:** Relatively fast (500k steps)

---

### 2. Tic-Tac-Toe with Human (Sparse Rewards)

**Characteristics:**
- ❌ **Sparse rewards**: Only at game end (win/lose/draw)
- ⏱️ **Medium episodes**: 5-9 moves per game
- 🤝 **Human in the loop**: Unpredictable opponent

**Reward structure:**
```python
rewards_per_episode = [
    0,    # move 1: placed X
    0,    # move 2: human played O
    0,    # move 3: placed X
    0,    # move 4: human played O
    0,    # move 5: placed X
    0,    # move 6: human played O
    0,    # move 7: placed X
    0,    # move 8: human played O
    +10   # move 9: WON! 🎉
]
```

**Key differences:**
```python
# Snake: Immediate feedback
if ate_food:
    reward = +10  # RIGHT NOW

# Tic-Tac-Toe: Delayed feedback  
if game_over:
    if won:
        reward = +10  # After 5-9 moves
    elif lost:
        reward = -10
    else:
        reward = 0
```

**Challenges:**
- **Credit assignment**: Which move was responsible for winning?
- **Longer discount horizon**: Need to look many steps ahead
- **Opponent variability**: Human plays differently each time

**PPO adaptations:**
```python
# Higher discount factor (look further ahead)
gamma = 0.99  # Snake
gamma = 0.95  # Tic-Tac-Toe (care less about distant future)

# More exploration
entropy_coef = 0.01  # Snake
entropy_coef = 0.05  # Tic-Tac-Toe (try more strategies)
```

---

### 3. LLM Training (RLHF - Reinforcement Learning from Human Feedback)

**Characteristics:**
- ❌❌ **Very sparse rewards**: Only at completion
- ⏱️⏱️ **Long episodes**: 100-1000 tokens
- 🎯 **Complex objectives**: Helpfulness, harmlessness, honesty

**Example: Training ChatGPT**

```python
# User prompt
user: "Explain quantum computing to a 5-year-old"

# LLM generates response (50-200 tokens)
llm_response = """
Quantum computing is like having a super 
special computer that can try many answers 
at the same time, instead of one by one 
like a normal computer...
""" 

# Reward ONLY at the end
if human_liked_response:
    reward = +1.0  # After generating 200 tokens!
else:
    reward = -1.0
```

**Reward structure:**
```python
rewards_per_episode = [
    0,  # token 1: "Quantum"
    0,  # token 2: "computing"
    0,  # token 3: "is"
    ...
    0,  # token 198: "computer"
    0,  # token 199: "..."
    +1.0  # token 200: Human feedback!
]
```

**Extreme credit assignment problem:**
- Which of the 200 tokens made the response good?
- Early tokens? Late tokens? The overall structure?

**PPO in RLHF:**

```python
# 1. Reward Model (trained separately)
class RewardModel:
    """Predicts human preference"""
    def score(self, prompt, response):
        # Returns: -1 to +1
        return score

# 2. Policy = LLM
llm = GPT(...)

# 3. Training loop
for batch in dataset:
    # Generate response
    response = llm.generate(prompt)
    
    # Get reward from reward model
    reward = reward_model.score(prompt, response)
    
    # Calculate advantages (challenging!)
    advantages = calculate_advantages_for_entire_sequence(
        rewards=[0, 0, 0, ..., 0, reward],
        values=value_net(all_tokens)
    )
    
    # Update LLM
    ppo_update(llm, advantages)
```

**Key differences from Snake:**

| Aspect | Snake | Tic-Tac-Toe | LLM (RLHF) |
|--------|-------|-------------|------------|
| **Reward frequency** | Every step | End of game | End of sequence |
| **Episode length** | 30-200 | 5-9 | 100-1000 |
| **Credit assignment** | Easy | Moderate | Very hard |
| **Training time** | Hours | Hours-Days | Days-Weeks |
| **State space** | Small (11D) | Small (9 cells) | Huge (vocab × context) |
| **Action space** | 3 actions | 9 actions | 50k+ tokens |

**Special techniques for LLMs:**

1. **Reward shaping**: Add intermediate rewards
```python
# Don't just reward at the end
if response_length < 100:
    intermediate_reward = -0.1  # Too short
if uses_toxic_word:
    intermediate_reward = -1.0  # Immediate penalty
```

2. **KL divergence penalty**: Don't drift too far from original model
```python
loss = ppo_loss - β * KL(π_new || π_old)
```

3. **Value function warm-start**: Pre-train value network

---

## Implementation Highlights from My Code

### 1. Policy Network (2-layer)

```python
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim=11, action_dim=3, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, state):
        x = self.relu(self.fc1(state))
        x = self.softmax(self.fc2(x))
        return x  # Action probabilities
```

**Design choices:**
- 2 layers: Simple enough for Snake, complex enough to learn
- 128 hidden units: Good balance of capacity and speed
- Softmax: Ensures probabilities sum to 1

### 2. Clean Separation: Environment Calculates Rewards

```python
# snake_env.py
class RewardCalculator:
    def calculate_reward(self, done, self_hit, had_food):
        reward = 0.0
        if not done: 
            reward += self.step_penalty
        if had_food:
            reward += self.food_reward
        if done:
            reward += self.death_penalty * (2 if self_hit else 1)
        return reward

# Environment owns reward logic
class SnakeEnv:
    def step(self, action):
        obs, terminated, self_hit, had_food, truncated, info = \
            self.game.take_action(action)
        
        # Calculate reward using RewardCalculator
        reward = self.reward_calculator.calculate_reward(
            terminated or truncated, self_hit, had_food
        )
        
        return obs, reward, terminated, truncated, info
```

**Why this design?**
- ✅ Agent focuses on learning, not reward engineering
- ✅ Easy to experiment with different reward functions
- ✅ Follows RL best practices (environment owns rewards)

### 3. Inference-Optimized Save/Load

```python
# Save only policy network (50% smaller!)
agent.save("model.pt")  # Only policy, no value network

# Load for inference
agent.load("model.pt", inference_only=True)

# During inference: Only policy network used
action, _, _ = agent.select_action(state)
```

**Why?**
- Value network only needed for training
- Smaller models → Faster deployment
- 250KB vs 500KB (50% reduction)

---

## Results: My Snake Agent Performance

After training for 500k timesteps (~45 minutes):

**Performance metrics:**
```
Average Score: 12.5
Max Score: 28
Average Episode Length: 68 steps
Success Rate (score > 5): 78%
```

**Learning progression:**

```
Timestep     |  Avg Score  |  Strategy
-------------|-------------|----------------------------------
0-50k        |  0.5        |  Random wandering
50k-150k     |  3.2        |  Learns to avoid walls
150k-300k    |  7.8        |  Starts chasing food
300k-500k    |  12.5       |  Efficient food collection
```

**Behavioral milestones:**
1. **0-100k**: Learns "don't hit walls"
2. **100k-200k**: Learns "food is good"
3. **200k-400k**: Learns basic navigation
4. **400k+**: Develops strategies (spiraling, efficient paths)

---

## Key Takeaways

### 1. PPO Core Principles
- ✅ **Clipped updates**: Prevents catastrophic policy changes
- ✅ **On-policy**: Learns from current policy's experiences
- ✅ **Sample efficient**: Reuses data for multiple epochs
- ✅ **Stable**: Works across many domains

### 2. Reward Engineering Matters

**Snake Game:**
```python
# Good
food_reward = 10.0
death_penalty = -10.0
step_penalty = -0.01

# Bad
food_reward = 1.0      # Too weak signal
death_penalty = -100.0  # Overly conservative
step_penalty = -1.0    # Too harsh, discouragessexploration
```

### 3. Architecture Choices Impact Learning

**My 2-layer network:**
- Fast training: ~45 minutes for 500k steps
- Good performance: Score of 12.5
- Generalizes well: Works on different grid sizes

**Could improve with:**
- Convolutional layers (for visual input)
- LSTM/GRU (for temporal patterns)
- Attention (for long-range dependencies)

---

## What's Next?

In upcoming posts, I'll cover:

### 1. **Tic-Tac-Toe with Human Opponent**
- Handling sparse rewards
- Opponent modeling
- Interactive RL
- Credit assignment over longer horizons

### 2. **Tiny LLM with RL**
- RLHF implementation
- Reward modeling
- Extremely sparse rewards
- Scaling challenges

### 3. **Comparison Study**
- Snake (dense rewards) vs Tic-Tac-Toe (sparse) vs LLM (very sparse)
- When does each reward structure work best?
- Practical tips for reward shaping

---

## Try It Yourself!

My complete PPO implementation is available on GitHub:
- Full Snake game with PPO from scratch
- Documented code with explanations

**Quick start:**
# Clone and install
git clone https://github.com/yourusername/snake-ppo
cd snake-ppo
pip install -r requirements.txt

# Train
python train_ppo.py train 50000  # Quick test
python train_ppo.py train 500000  # Full training

# Play
python train_ppo.py play


**Coming up next:** Building a Tic-Tac-Toe agent that learns to play against humans using the same PPO framework!