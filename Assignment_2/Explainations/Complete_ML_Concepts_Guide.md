# Complete Deep Learning Concepts Guide
### Using RNA Structure Prediction as Our Teaching Example

---

## 🎯 The Big Picture: Training a Robot to Fold Origami

Imagine you're training a robot to fold origami. The robot has:
- **Sensors** (input): Sees paper dimensions and creases
- **Motors** (weights): Adjustable strength/angles for folding
- **Memory** (model): Remembers folding patterns
- **Trainer** (you): Shows correct vs incorrect folds

**Your RNA assignment works exactly like this:**
- **Input**: RNA sequences (like paper dimensions)
- **Model**: CNN with adjustable weights (like robot motors)
- **Output**: Contact map predictions (like fold patterns)
- **Training**: Comparing predictions to known structures

---

## 📊 Part 1: Understanding Your Data

### What is `df.iloc[0]['sequence']` and `df.iloc[0]['structure']`?

```python
# Your dataset is a pandas DataFrame (like an Excel table)
df.iloc[0]  # Gets the FIRST ROW (index 0)
           # iloc = "integer location"

# Example row:
# | sequence              | structure          |
# |----------------------|-------------------|
# | "AUGCGAUUCGAU"       | "(((...)))"       |

seq = df.iloc[0]['sequence']    # = "AUGCGAUUCGAU" (RNA bases)
struct = df.iloc[0]['structure'] # = "(((...)))" (pairing info)
```

**What this means:**
- `sequence`: The RNA letters (A, U, G, C) - like DNA but with U instead of T
- `structure`: Shows which bases pair together using parentheses
  - `(` and `)` at same depth = those bases pair/bond
  - `.` = unpaired base

---

## 🧬 Part 2: Encoding RNA - Why Map A=0, U=1, G=2, C=3?

### The Problem: Computers Don't Understand Letters

```python
# Computer sees letters as... nothing useful for math
"A" → ??? Can't multiply or add letters!

# Solution: Map to numbers
mapping = {'A': 0, 'U': 1, 'G': 2, 'C': 3}

# Now can use these for calculations
```

### One-Hot Encoding: The Robot's Language

```python
# Why not just use 0,1,2,3 directly?
# Because: 3 ≠ "more important" than 0

# One-hot encoding treats each base equally:
A → [1, 0, 0, 0]  # "First slot"
U → [0, 1, 0, 0]  # "Second slot"
G → [0, 0, 1, 0]  # "Third slot"
C → [0, 0, 0, 1]  # "Fourth slot"

# Like turning on exactly ONE light switch
```

**In your assignment:**
```python
one_hot = np.zeros(4)
one_hot[base_to_int[base]] = 1

# If base = 'G':
# base_to_int['G'] = 2
# one_hot[2] = 1
# Result: [0, 0, 1, 0]
```

---

## 🗺️ Part 3: The 2D Feature Map - Building the Puzzle

### Understanding the 8 Features

```python
# For EVERY PAIR of positions (i, j) in sequence:
# Position i has base 'A' → [1,0,0,0]
# Position j has base 'U' → [0,1,0,0]

# Concatenate them:
feature[i,j] = concat([1,0,0,0], [0,1,0,0])
             = [1,0,0,0,0,1,0,0]
             # ↑↑↑↑  ↑↑↑↑
             # base i base j
             # 4 features + 4 features = 8 TOTAL
```

### Visual Example:

```
RNA Sequence: "AUGC" (length L=4)

Step 1: One-hot encode each base
A → [1,0,0,0]
U → [0,1,0,0]
G → [0,0,1,0]
C → [0,0,0,1]

Step 2: Create 2D grid (4×4×8)
For position (0,1) - bases A and U:
┌─────────────────────┐
│ [1,0,0,0, 0,1,0,0] │ ← 8 features!
│      ↑        ↑     │
│   base A   base U   │
└─────────────────────┘

For position (0,2) - bases A and G:
┌─────────────────────┐
│ [1,0,0,0, 0,0,1,0] │
└─────────────────────┘

Do this for ALL 16 pairs (4×4 grid)
Result: 4×4×8 tensor
```

### Why 2D?

```
RNA Structure is about RELATIONSHIPS between positions:
   
   Position:  0   1   2   3
   Sequence:  A   U   G   C
              ↓   ↓   ↓   ↓
   Contact    0 [0,1,0,0]  # A-U pairing at (0,1)
   Map:       1 [1,0,0,0]  # Shows if bases bond
              2 [0,0,0,1]  # 1 = bonded, 0 = not bonded
              3 [0,0,1,0]
```

**Better Visualization:**

```
INPUT: RNA Sequence (1D String)
──────────────────────────────────────
   AUGCGAUUCGAU... (length L)
   
   ↓↓↓ ENCODE TO 2D ↓↓↓
   
   For each pair of positions (i,j):
   Combine their one-hot encodings
   
┌─────────────────────────────────────┐
│         L × L × 8 TENSOR            │
│    (Features for all pairs)         │
│                                     │
│    Position pairs:                  │
│    (0,0) (0,1) (0,2) ... (0,L)     │
│    (1,0) (1,1) (1,2) ... (1,L)     │
│     ...   ...   ...  ...  ...      │
│    (L,0) (L,1) (L,2) ... (L,L)     │
│                                     │
│    Each pair has 8 numbers          │
└─────────────────────────────────────┘
   
   ↓↓↓ PASS THROUGH CNN ↓↓↓
   
   [Conv → ReLU → BatchNorm → Pool] × N layers
   
   ↓↓↓ OUTPUT ↓↓↓
   
┌─────────────────────────────────────┐
│      CONTACT MAP (L × L)            │
│    (Predicted base pairings)        │
│                                     │
│     0.0  0.9  0.1  0.0  0.8  ...   │
│     0.9  0.0  0.0  0.2  0.1  ...   │
│     0.1  0.0  0.0  0.0  0.0  ...   │
│     0.0  0.2  0.0  0.0  0.7  ...   │
│     0.8  0.1  0.0  0.7  0.0  ...   │
│     ...  ...  ...  ...  ...  ...   │
│                                     │
│   Values close to 1 = likely bond   │
│   Values close to 0 = no bond       │
└─────────────────────────────────────┘
   
   ↓↓↓ THRESHOLD (e.g., > 0.5) ↓↓↓
   
┌─────────────────────────────────────┐
│    BINARY CONTACT MAP (L × L)       │
│    (Final prediction)               │
│                                     │
│     0  1  0  0  1  ...              │
│     1  0  0  0  0  ...              │
│     0  0  0  0  0  ...              │
│     0  0  0  0  1  ...              │
│     1  0  0  1  0  ...              │
│     ...  ...  ...  ...  ...        │
│                                     │
│   1 = bases bond                    │
│   0 = bases don't bond              │
└─────────────────────────────────────┘
```

---

## 🤖 Part 4: What is a Tensor?

**Tensor = Multi-dimensional array of numbers**

```python
# 0D Tensor (scalar)
x = 5

# 1D Tensor (vector)
x = [1, 2, 3, 4]

# 2D Tensor (matrix)
x = [[1, 2],
     [3, 4]]

# 3D Tensor (your RNA features!)
x = [[[1,0,0,0,0,1,0,0], [1,0,0,0,0,0,1,0]],
     [[0,1,0,0,1,0,0,0], [0,1,0,0,0,1,0,0]]]
     # Shape: (2, 2, 8)
     #        ↑  ↑  ↑
     #        L  L  Features
```

### Converting NumPy to PyTorch Tensor:

```python
# NumPy array (for traditional computing)
numpy_array = np.array([[1, 2], [3, 4]])

# PyTorch tensor (for deep learning/GPU)
tensor = torch.from_numpy(numpy_array).float()

# Why .float()? CNNs need floating-point numbers
# (decimals) for gradient calculations
```

---

## 🏗️ Part 5: CNN Architecture Components

### nn.Conv2d: The Pattern Detective

```python
self.conv1 = nn.Conv2d(
    in_channels=8,    # Input: 8 feature maps (your concatenated one-hots!)
    out_channels=32,  # Output: 32 new feature maps
    kernel_size=3,    # 3×3 sliding window
    padding=1         # Add border to keep size
)
```

**What it does:**
```
Input: 8 channels (8 numbers per position pair)

   ┌───┐
   │ 8 │  ← 8 feature maps
   │ # │
   │ # │
   └───┘

Kernel slides across, looking for patterns:
   
   ┌─────┐        32 different filters
   │ 3×3 │  →  each learns different patterns
   │filter│      (edges, curves, base pair motifs)
   └─────┘

Output: 32 channels
   
   ┌────┐
   │ 32 │  ← 32 feature maps
   │ ## │     (detected patterns)
   │ ## │
   └────┘
```

### nn.ReLU: The "Keep Good, Kill Bad" Filter

```python
# ReLU(x) = max(0, x)

Input:  [-2, -1, 0, 3, 5]
         ↓   ↓  ↓  ↓  ↓
ReLU:   [0,  0, 0, 3, 5]  # Negative → 0, Positive → keep

# Why? Adds non-linearity (lets network learn complex patterns)
# Without it: stacked layers = just one big linear function
```

### nn.BatchNorm2d: The Stabilizer

```python
# Normalizes values to have mean=0, std=1
# Like standardizing test scores across classes

Before BatchNorm:  [100, 200, 1000, 50]   # Wildly different scales
After BatchNorm:   [-0.5, 0.2, 2.1, -1.8] # Consistent scale

# Benefits:
# 1. Faster training
# 2. More stable (less explosion/vanishing)
# 3. Less sensitive to initialization
```

### nn.BCEWithLogitsLoss: The Scorekeeper

```python
# Binary Cross-Entropy Loss with Logits
# "How wrong are your binary predictions?"

# Your task: Predict 0 or 1 for each position pair
# (bonded or not bonded)

# BCEWithLogitsLoss combines:
# 1. Sigmoid (convert raw scores → probabilities 0-1)
# 2. Binary Cross-Entropy (measure error)

# Example:
True:      [1, 0, 1, 0]  # Ground truth
Predicted: [0.9, 0.2, 0.7, 0.3]  # Model output
Loss:      0.15  # Low = good!

True:      [1, 0, 1, 0]
Predicted: [0.1, 0.8, 0.2, 0.9]  # Backwards!
Loss:      2.5   # High = bad!
```

---

## 🎓 Part 6: Training Process

### What is an Epoch?

```
Epoch = ONE complete pass through ALL training data

Your dataset: 1000 RNA sequences
Batch size: 32

Epoch 1:
├─ Batch 1:  samples 0-31    → forward → loss → backward → update
├─ Batch 2:  samples 32-63   → forward → loss → backward → update
├─ Batch 3:  samples 64-95   → forward → loss → backward → update
│  ...
└─ Batch 32: samples 992-999 → forward → loss → backward → update

Epoch 2: (Start over with SAME data)
├─ Batch 1:  samples 0-31    → forward → loss → backward → update
│  ...
```

### "How Does It Improve If It Uses the SAME Data?"

**The Origami Robot Analogy:**

```
Day 1 (Epoch 1):
├─ Paper 1: Robot folds badly → You adjust motors slightly
├─ Paper 2: Robot folds badly → Adjust motors more
├─ Paper 3: Robot folds better! → Small adjustment
└─ End of day: Motors are DIFFERENT than start

Day 2 (Epoch 2):
├─ Paper 1 AGAIN: But robot has NEW motor settings!
│              → Folds better than Day 1
├─ Paper 2 AGAIN: Further improvement
└─ End of day: Motors even BETTER

Day 3, 4, 5... Keep refining until perfect
```

**In your CNN:**
```python
# Epoch 1: Start with random weights
Initial weights: [0.5, -0.2, 0.8, ...]

Sequence 1 → Prediction wrong → Update weights → [0.52, -0.18, 0.79, ...]
Sequence 2 → Prediction wrong → Update weights → [0.54, -0.15, 0.77, ...]
...
Sequence 1000 → Weights now: [0.95, 0.42, 1.23, ...]

# Epoch 2: Same sequences, DIFFERENT weights!
Sequence 1 → Prediction BETTER → Smaller update → [0.951, 0.421, 1.229, ...]
Sequence 2 → Prediction BETTER → Smaller update → [0.952, 0.422, 1.228, ...]
...

# Each epoch: weights get closer to optimal
```

### What is a Batch?

```python
# Instead of updating after EACH sample (slow):
for sample in data:
    prediction = model(sample)      # Process 1
    loss = criterion(prediction)    # Compute loss for 1
    loss.backward()                 # Gradients for 1
    optimizer.step()                # Update (expensive!)

# Process samples in BATCHES (faster):
for batch in data:  # batch = 32 samples
    predictions = model(batch)      # Process 32 at once! (GPU parallel)
    loss = criterion(predictions)   # Average loss for 32
    loss.backward()                 # Gradients for batch
    optimizer.step()                # One update for 32 samples

# Benefits:
# 1. Faster (GPU processes 32 in parallel)
# 2. Smoother updates (averaged over 32 samples)
# 3. Less noise in training
```

---

## 📉 Part 7: Loss, Gradients, and Learning

### What is Loss?

```
Loss = "How wrong is your model?"

Perfect prediction:  Loss = 0
Terrible prediction: Loss = high

# Like a golf score: lower is better!
```

**In your RNA task:**
```python
# True contact map (ground truth)
true = [[0, 1, 0],
        [1, 0, 0],
        [0, 0, 1]]

# Model prediction
pred = [[0.1, 0.8, 0.2],
        [0.9, 0.1, 0.1],
        [0.0, 0.1, 0.9]]

# Loss measures difference
loss = BCEWithLogitsLoss(pred, true)  # = 0.15 (pretty good!)
```

### What are Gradients? (The Core of Deep Learning!)

**Gradient = Direction and magnitude to change weights to reduce loss**

```
Imagine loss as a MOUNTAIN and you're trying to reach the VALLEY (minimum loss):

        🏔️
       /  \
      /    \
     /      \
    /   YOU  \
   /    ⛷️    \
  /           \
 /    VALLEY   \
/_____🎯_______\

Gradient tells you:
1. Which direction to ski (down!)
2. How steep it is (how far to move)

# In math terms:
gradient = ∂Loss/∂weight  # "How much does loss change when weight changes?"
```

**Concrete Example:**

```python
# Weight in your CNN
weight = 0.5

# Compute loss with current weight
loss = compute_loss(weight=0.5)  # loss = 2.5

# Gradient calculation (automatically done by PyTorch)
gradient = ∂loss/∂weight  # = -0.8

# Meaning:
# gradient = -0.8 → negative means "decrease weight to reduce loss"
# gradient = -0.8 → magnitude 0.8 means "strong effect"

# If gradient was +0.8:
# → positive means "increase weight to reduce loss"
```

### What is Backpropagation? (How Gradients Are Computed)

```
FORWARD PASS (prediction):
Input → Conv1 → ReLU → Conv2 → ReLU → Output → Loss
 X        h1      h2      h3      h4      ŷ       L

BACKWARD PASS (gradients):
Input ← Conv1 ← ReLU ← Conv2 ← ReLU ← Output ← Loss
        ∂L/∂W1          ∂L/∂W2                  ∂L/∂ŷ

# Chain rule of calculus:
∂L/∂W1 = ∂L/∂h4 × ∂h4/∂h3 × ∂h3/∂W1
         ↑ Backprop computes this automatically!
```

### The Training Loop:

```python
for epoch in tqdm(range(num_epochs), desc='Training'):
    for batch in train_loader:
        # 1. Forward pass
        predictions = model(batch)      # Get predictions
        loss = criterion(predictions)   # Compute loss
        
        # 2. Backward pass
        optimizer.zero_grad()  # Clear old gradients (important!)
        loss.backward()        # Compute new gradients (backprop!)
        
        # 3. Update weights
        optimizer.step()       # Adjust weights using gradients
```

**Detailed breakdown:**

```python
# 1. optimizer.zero_grad()
# WHY? Gradients ACCUMULATE by default
# Without this, new gradients ADD to old ones → wrong!

Iteration 1: grad = 0.5
Iteration 2: grad = 0.3
Without zero_grad(): total grad = 0.5 + 0.3 = 0.8 ❌
With zero_grad():    total grad = 0.3 ✅


# 2. loss.backward()
# Computes gradients for ALL weights using backpropagation
# PyTorch does this automatically using chain rule!

Before: weight.grad = None
After:  weight.grad = -0.8  # Direction to move


# 3. optimizer.step()
# Updates weights using gradients and learning rate

Old weight: 0.5
Gradient:  -0.8
Learning rate: 0.01

New weight = old - (learning_rate × gradient)
           = 0.5 - (0.01 × -0.8)
           = 0.5 + 0.008
           = 0.508

# Weight increased slightly (because gradient was negative)
```

---

## 🎚️ Part 8: Learning Rate

### What is Learning Rate?

```
Learning Rate = Size of each weight update step

weight_new = weight_old - (learning_rate × gradient)
                          ↑
                    Controls step size!
```

**The Robot Knob Analogy:**

```
Training robot to tighten screws:

Learning Rate = 0.001 (too small)
┌─────────────────────────────────┐
│ Turn knob 0.001° per attempt   │
│ Takes FOREVER to tighten screw  │
│ 10,000 attempts → barely moved  │
└─────────────────────────────────┘

Learning Rate = 0.1 (good!)
┌─────────────────────────────────┐
│ Turn knob 0.1° per attempt      │
│ Steady progress                 │
│ 100 attempts → properly tightened│
└─────────────────────────────────┘

Learning Rate = 10.0 (too large)
┌─────────────────────────────────┐
│ Turn knob 10° per attempt       │
│ Overshoots → strips screw!      │
│ Oscillates back and forth       │
│ NEVER settles at correct point  │
└─────────────────────────────────┘
```

**In Loss Space:**

```
Learning Rate = 0.0001 (too small)
    🏔️
   /│\      You: ⛷️
  / │ \         ↓ tiny steps
 /  │  \        ↓
/__Valley\     Takes forever!

Learning Rate = 0.01 (good)
    🏔️
   / \     You: ⛷️
  /   \         ↓
 /     \        ↓ 
/_Valley_\    ⛷️ → 🎯 Reaches valley!

Learning Rate = 1.0 (too large)
    🏔️
   / \     You: ⛷️
  /   \         ↓ huge jump!
 /     \        ↑ bounces back!
/_______\      ⛷️↕️⛷️ Oscillates, never settles!
```

---

## ⚠️ Part 9: Overfitting

### What is Overfitting?

**Overfitting = Model memorizes training data instead of learning patterns**

**The Exam Analogy:**

```
GOOD LEARNING (Generalization):
Student studies concepts, understands principles
→ Can solve NEW problems on exam ✅

OVERFITTING (Memorization):
Student memorizes exact homework answers
→ Fails on NEW exam questions ❌
```

**In Your RNA Model:**

```
GOOD MODEL:
Training: 95% accuracy
Testing:  92% accuracy  ← Close to training! ✅
→ Learned actual RNA folding patterns

OVERFIT MODEL:
Training: 99% accuracy
Testing:  65% accuracy  ← Much worse! ❌
→ Memorized training sequences
→ Can't predict NEW sequences
```

**Visual:**

```
True Pattern (what you want to learn):
    •     •
  •   •••   •
 •           •
•    DATA    •

Good Fit:
    •  ___•___
  • /   •••   \•
 •/            \•
•      CURVE    •
← Captures general trend


Overfit:
    •╱╲___•
  •╱  ╲•••╲  •
 •╱    ╲   ╲  •
•       ╲   ╲  •
← Connects every point exactly
   Won't work for NEW data!
```

**How to Prevent:**

```python
# 1. Dropout: Randomly "turn off" neurons during training
nn.Dropout(0.5)  # 50% neurons disabled each iteration
                 # Forces network to learn robust features

# 2. Early Stopping: Stop when validation loss increases
if val_loss > prev_val_loss:
    break  # Stop training!

# 3. Regularization: Penalize large weights
loss = prediction_error + λ × Σ(weights²)
       ↑                   ↑
       Main loss          Penalty for complex model

# 4. More Data: Harder to memorize 1M samples than 100
```

---

## 📊 Part 10: Evaluation Metrics

### Predictions - What Type?

```python
# Your model outputs PROBABILITIES (0.0 to 1.0)
raw_output = model(input)
# Example: [[0.92, 0.15, 0.03, 0.88],
#           [0.14, 0.67, 0.91, 0.22]]

# Apply threshold to get BINARY predictions
predictions = (raw_output > 0.5).int()
# Result:  [[1, 0, 0, 1],
#           [0, 1, 1, 0]]

# Types:
# - Raw: Float probabilities (for loss calculation)
# - Binary: 0 or 1 (for accuracy, F1, etc.)
```

### F1-Score

```
F1 = Harmonic mean of Precision and Recall

Confusion Matrix:
                 Predicted
                 0      1
Actual    0    TN     FP
          1    FN     TP

Precision = TP / (TP + FP)  # "Of predicted positives, how many correct?"
Recall    = TP / (TP + FN)  # "Of actual positives, how many found?"
F1        = 2 × (Precision × Recall) / (Precision + Recall)
```

**RNA Example:**

```
True bonds:      100 pairs should bond
Predicted bonds: 120 pairs
Correct bonds:   85 pairs

TP (True Positive):  85  # Correctly predicted bonds
FP (False Positive): 35  # Predicted bond, but wrong
FN (False Negative): 15  # Missed real bonds
TN (True Negative):  ~10000  # Correctly predicted no-bond

Precision = 85 / (85 + 35) = 0.708  # 70.8% of predictions are correct
Recall    = 85 / (85 + 15) = 0.850  # Found 85% of real bonds
F1        = 2 × (0.708 × 0.850) / (0.708 + 0.850) = 0.773
```

### AUC (Area Under ROC Curve)

```
ROC Curve: Plot of True Positive Rate vs False Positive Rate
           at different threshold values

Threshold = 0.1: Almost everything predicted as 1
           → High TPR (catch all real positives)
           → High FPR (lots of false alarms)

Threshold = 0.9: Almost everything predicted as 0
           → Low TPR (miss many real positives)
           → Low FPR (few false alarms)

Plot all thresholds:
    TPR
    1.0│    ╱──────
       │   ╱
       │  ╱   ← AUC = area under curve
    0.5│ ╱       Higher = better
       │╱
    0.0└────────────
       0.0   0.5   1.0  FPR

AUC = 1.0: Perfect classifier
AUC = 0.5: Random guessing
AUC = 0.8: Your model (pretty good!)
```

### Imbalanced Datasets

```
Your RNA contact map:
Bonded pairs (1):    500   ← Minority class
Non-bonded pairs (0): 9500  ← Majority class

Problem: Model can get 95% accuracy by predicting ALL zeros!
But misses ALL bonds (useless!)

# Predicted: all 0s
# Actual:    9500 zeros, 500 ones
# Accuracy:  9500/10000 = 95% ✅ (looks good!)
# But F1:    0.0 ❌ (terrible!)

Solutions:
1. Use F1-score (balances precision & recall)
2. Weighted loss: penalty more for missing 1s
3. Oversampling: duplicate minority class samples
4. Undersampling: reduce majority class samples
```

---

## 🔧 Part 11: Optimizer (Adam)

### What Does an Optimizer Do?

```python
# Basic gradient descent:
weight = weight - learning_rate × gradient

# But Adam is SMARTER:
# 1. Momentum: Uses history of gradients (smooths updates)
# 2. Adaptive learning rates: Different rate per parameter
# 3. Bias correction: Fixes initial estimates
```

**Why Adam is Better:**

```
Gradient Descent (basic):
    ╱╲    Gradient changes ± wildly
   ╱  ╲   
  ╱    ╲  Updates zigzag
 ╱      ╲ Slow convergence

Adam (momentum + adaptive):
    ───   Smooth trajectory
   ╱     Uses running average
  ╱      Converges faster!
 ╱
```

**Under the Hood:**

```python
# Adam tracks:
m = 0  # First moment (mean of gradients)
v = 0  # Second moment (variance of gradients)

# Each update:
m = β1 × m + (1-β1) × gradient       # Momentum
v = β2 × v + (1-β2) × gradient²      # Adaptive
weight = weight - lr × m / (√v + ε)  # Update

# Benefits:
# - Stable even with noisy gradients
# - Automatically adjusts learning rate
# - Works well for most problems
```

---

## 🧪 Part 12: The Complete Training Workflow

### From RNA Sequence to Prediction

```python
# ============================================
# STAGE 1: PREPARATION
# ============================================
# Load data
df = pd.read_csv('TR0.csv')  # Training data
seq = df.iloc[0]['sequence']  # "AUGCGAUU..."
struct = df.iloc[0]['structure']  # "((...))"

# Encode sequence → 2D feature map (L×L×8)
features = encode_sequence_2d(seq)  # One-hot pairs

# Convert structure → contact map (L×L binary)
contact_map = structure_to_contact_map(struct)

# Convert to tensors
X = torch.from_numpy(features).float()
y = torch.from_numpy(contact_map).float()

# ============================================
# STAGE 2: MODEL ARCHITECTURE
# ============================================
model = nn.Sequential(
    # Layer 1: 8 → 32 channels
    nn.Conv2d(8, 32, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.BatchNorm2d(32),
    
    # Layer 2: 32 → 64 channels
    nn.Conv2d(32, 64, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.BatchNorm2d(64),
    
    # Layer 3: 64 → 1 channel (final prediction)
    nn.Conv2d(64, 1, kernel_size=1),
    # Output: (batch, 1, L, L) probabilities
)

# ============================================
# STAGE 3: TRAINING SETUP
# ============================================
criterion = nn.BCEWithLogitsLoss()  # Loss function
optimizer = torch.optim.Adam(
    model.parameters(), 
    lr=0.001  # Learning rate
)

# ============================================
# STAGE 4: TRAINING LOOP
# ============================================
num_epochs = 50

for epoch in tqdm(range(num_epochs), desc='Training'):
    for batch_X, batch_y in train_loader:
        # === FORWARD PASS ===
        predictions = model(batch_X)  # (batch, 1, L, L)
        predictions = predictions.squeeze(1)  # (batch, L, L)
        loss = criterion(predictions, batch_y)
        
        # === BACKWARD PASS ===
        optimizer.zero_grad()  # Clear old gradients
        loss.backward()        # Compute gradients
        optimizer.step()       # Update weights
        
    # Print progress
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

# ============================================
# STAGE 5: EVALUATION
# ============================================
model.eval()  # Switch to evaluation mode
with torch.no_grad():  # Don't compute gradients
    test_pred = model(test_X)
    test_pred_binary = (test_pred > 0.5).int()
    
    # Compute metrics
    accuracy = (test_pred_binary == test_y).float().mean()
    f1 = f1_score(test_y.flatten(), test_pred_binary.flatten())
    
print(f"Test Accuracy: {accuracy:.4f}")
print(f"Test F1-Score: {f1:.4f}")
```

---

## 🔄 Part 13: Special Syntax Explained

### `desc` in tqdm

```python
for epoch in tqdm(range(num_epochs), desc='Training'):
    # Process...
    pass

# Output in terminal:
# Training: |████████░░░░░░░░| 50/100 [00:30<00:30, 1.2it/s]
#           ↑
#           desc parameter - labels the progress bar
```

### `.item()` for Loss

```python
loss = criterion(predictions, targets)
# loss is a TENSOR (for backprop)
# loss.item() extracts the Python NUMBER

print(loss)        # tensor(0.5234, grad_fn=<BinaryCrossEntropy>)
print(loss.item()) # 0.5234  ← Clean number for logging
```

---

## 📚 Recommended YouTube Resources

### Must-Watch Channels:

1. **3Blue1Brown - Neural Networks Series**
   - "But what is a neural network?"
   - "Gradient descent, how neural networks learn"
   - "What is backpropagation really doing?"
   - https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi
   - *Best for: Intuition and visualization*

2. **Andrej Karpathy**
   - "The spelled-out intro to neural networks and backpropagation"
   - "Neural Networks: Zero to Hero"
   - https://www.youtube.com/c/AndrejKarpathy
   - *Best for: Coding from scratch, deep understanding*

3. **StatQuest with Josh Starmer**
   - "Neural Networks Pt. 1-4: Backpropagation Main Ideas"
   - "Gradient Descent, Step-by-Step"
   - https://www.youtube.com/c/joshstarmer
   - *Best for: Clear step-by-step explanations*

4. **Welch Labs**
   - "Neural Networks Demystified"
   - "Learning to See"
   - https://www.youtube.com/c/WelchLabsVideo
   - *Best for: Mathematical foundations*

5. **Yannic Kilcher**
   - "Attention Is All You Need"
   - Various paper explanations
   - https://www.youtube.com/c/YannicKilcher
   - *Best for: Latest research*

### Specific Topics:

**Convolutional Neural Networks:**
- 3Blue1Brown: "Convolutional Neural Networks"
- Stanford CS231n: Lecture 5 (CNNs) by Fei-Fei Li
- https://www.youtube.com/watch?v=KuXjwB4LzSA

**Backpropagation Deep Dive:**
- Andrej Karpathy: "Building Micrograd"
- Shows backprop implementation line-by-line

**Loss Functions & Optimization:**
- StatQuest: "Gradient Descent"
- "Adam Optimizer Explained"

**Overfitting & Regularization:**
- StatQuest: "Regularization Part 1-3"
- 3Blue1Brown mentions in main series

---

## 🎯 Summary: The Complete Picture

```
┌────────────────────────────────────────────────────────────┐
│                    YOUR RNA CNN WORKFLOW                     │
├────────────────────────────────────────────────────────────┤
│                                                              │
│  1. DATA: RNA sequences + structures                        │
│     └─→ Encode to (L×L×8) tensors                          │
│                                                              │
│  2. MODEL: Stack of Conv2d + ReLU + BatchNorm               │
│     └─→ Learns patterns in 2D feature maps                 │
│                                                              │
│  3. LOSS: BCEWithLogitsLoss                                 │
│     └─→ Measures prediction error                          │
│                                                              │
│  4. GRADIENTS: Backpropagation                              │
│     └─→ Computes ∂Loss/∂Weight for all weights             │
│                                                              │
│  5. OPTIMIZER: Adam                                          │
│     └─→ Updates weights using gradients                     │
│                                                              │
│  6. TRAINING: Repeat for many epochs                        │
│     └─→ Weights improve, loss decreases                    │
│                                                              │
│  7. EVALUATION: F1-score, AUC, accuracy                     │
│     └─→ Measure performance on test set                    │
│                                                              │
└────────────────────────────────────────────────────────────┘

KEY INSIGHT:
Deep learning = finding weights that minimize loss
                through iterative gradient-based updates
                
Your CNN learns RNA folding patterns the same way
a robot learns to fold origami: through repeated
practice with feedback (gradients) that guide
adjustments (weight updates) toward better 
performance (lower loss).
```

---

**Final Note:** Every concept here connects to your RNA structure prediction assignment. The CNN "sees" base pair patterns in the 2D encoded features, learns to recognize which patterns indicate bonding, and predicts contact maps. Training is the process of adjusting millions of weights to make these predictions accurate. Understanding this workflow is key to debugging, improving, and innovating in deep learning!

