# CNN, Deep Learning, and Training - Complete Guide

## 🧠 What is Deep Learning?

### Simple Definition
**Deep Learning** = Teaching computers to learn from examples, using artificial neural networks with many layers.

### The Learning Process
```
1. Show the model examples (RNA sequences + correct pairings)
2. Model makes predictions (guesses which bases pair)
3. Calculate how wrong the predictions are (loss)
4. Adjust model parameters to reduce errors
5. Repeat thousands of times
6. Eventually, model learns patterns and can predict new examples
```

### Why "Deep"?
**"Deep"** = Many layers of processing

```
Input → [Layer 1] → [Layer 2] → [Layer 3] → ... → [Layer 10] → Output
        Find edges   Find shapes  Find objects     Complex patterns
```

Each layer learns increasingly complex patterns.

---

## 🔲 What is a CNN (Convolutional Neural Network)?

### Core Concept
A type of neural network that's **exceptionally good at finding local patterns** in grid-like data.

### Why CNNs Work
- **Local patterns matter**: Nearby pixels/positions are related
- **Pattern reuse**: Same pattern can appear anywhere (translation invariance)
- **Hierarchical learning**: Simple patterns → complex patterns

### Where CNNs Excel
1. **Image recognition** (original use case)
2. **Video analysis**
3. **Medical imaging**
4. **Any grid-structured data** (like contact maps!)

---

## 🎬 How Convolution Works (Visual Explanation)

### The Convolution Operation

#### **Example: 1D Convolution**
```
Input sequence: [1, 2, 3, 4, 5, 6]
Filter (kernel): [a, b, c]

Slide the filter across the input:

Position 0: [1, 2, 3] · [a, b, c] = 1a + 2b + 3c = output[0]
Position 1: [2, 3, 4] · [a, b, c] = 2a + 3b + 4c = output[1]
Position 2: [3, 4, 5] · [a, b, c] = 3a + 4b + 5c = output[2]
Position 3: [4, 5, 6] · [a, b, c] = 4a + 5b + 6c = output[3]

Result: [output[0], output[1], output[2], output[3]]
```

#### **Example: 2D Convolution**
```
Input image (5×5):
┌─────────────┐
│ 1  2  3  4  5│
│ 6  7  8  9 10│
│11 12 13 14 15│
│16 17 18 19 20│
│21 22 23 24 25│
└─────────────┘

Filter (3×3):
┌─────┐
│ a b c│
│ d e f│
│ g h i│
└─────┘

Slide filter over input:

Position (0,0):          Position (0,1):
┌─────────┐              ┌─────────┐
│[1  2  3]·····│          │ 1[2  3  4]····│
│[6  7  8] │              │ 6[7  8  9] │
│[11 12 13]····│          │11[12 13 14]···│
└─────────┘              └─────────┘

Compute: 1a+2b+3c+       Compute: 2a+3b+4c+
         6d+7e+8f+                7d+8e+9f+
         11g+12h+13i               12g+13h+14i
       = output[0,0]             = output[0,1]

Continue sliding to fill entire output map
```

### What the Filter Learns
The values [a, b, c, d, e, f, g, h, i] are **learned during training** to detect useful patterns:
- Edge detector
- Corner detector
- Texture detector
- RNA pairing patterns (in our case)

---

## 📐 1D vs 2D vs 3D CNNs

### **1D CNN**
- **Input**: 1D sequence (time series, text, DNA)
- **Filter**: Slides along one dimension
- **Use**: Pattern detection in sequences

```
Input: [x₀, x₁, x₂, x₃, x₄, x₅] ← 1D
Filter: [w₀, w₁, w₂] ← Slides left/right
```

**Example Applications**:
- Text classification
- Time series forecasting
- DNA sequence analysis

### **2D CNN** (What you're using!)
- **Input**: 2D grid (images, contact maps)
- **Filter**: Slides along two dimensions (height and width)
- **Use**: Spatial pattern detection

```
Input:        Filter:
┌───────┐     ┌───┐
│ × × × │     │ w w│  ← Slides up/down
│ × × × │     │ w w│     AND left/right
│ × × × │     └───┘
└───────┘
```

**Example Applications**:
- Image classification
- Object detection
- Contact map prediction (this assignment!)

### **3D CNN**
- **Input**: 3D volume (video, medical scans)
- **Filter**: Slides along three dimensions
- **Use**: Spatio-temporal patterns

```
Input: Video (frames stacked)
┌───────┐
│Frame 1│
│Frame 2│  ← Slides in width, height, AND time
│Frame 3│
└───────┘
```

**Example Applications**:
- Video action recognition
- Medical CT scan analysis
- 3D object recognition

---

## 🧬 Why 2D CNN for RNA Structure?

### The Problem
- **Input**: 1D sequence (A, U, G, C)
- **Output**: 2D contact map (which pairs with which)

### The Solution: Transform 1D → 2D

#### **Step 1: One-Hot Encode Sequence**
```python
Sequence: AUGC
One-hot: 
[[1,0,0,0],  # A
 [0,1,0,0],  # U
 [0,0,1,0],  # G
 [0,0,0,1]]  # C
Shape: (4, 4) - still 1D conceptually
```

#### **Step 2: Create 2D Feature Grid**
For each position pair (i, j), concatenate their features:

```python
# For position (0, 1): A and U
feature[0,1] = concat(one_hot[0], one_hot[1])
             = concat([1,0,0,0], [0,1,0,0])
             = [1,0,0,0,0,1,0,0]  # 8 features

# Do this for all pairs (i, j)
# Result: (L, L, 8) 2D feature map
```

#### **Step 3: Apply 2D CNN**
```python
Input: (L, L, 8)
      ↓
   [Conv2D layers]
      ↓
Output: (L, L, 1) - probability of pairing
```

### Why This Works
- **Local patterns**: Nearby bases form structures (stems, loops)
- **Symmetry**: Convolution respects pairing symmetry
- **Efficient**: Reuses filters across all position pairs

---

## 🔄 The Training Process - Complete Breakdown

### What is an Epoch?
**Epoch** = One complete pass through the entire training dataset

```
Epoch 1: See all 10,000 training samples once
Epoch 2: See all 10,000 training samples again (different order)
...
Epoch 10: See all 10,000 training samples for the 10th time
```

### What is a Batch?
**Batch** = A subset of training samples processed together

```
Dataset: 10,000 samples
Batch size: 32 samples

Number of batches per epoch: 10,000 / 32 = 313 batches
```

**Why batches?**
1. **Faster**: Parallel processing on GPU
2. **Memory**: Can't fit all data in GPU memory
3. **Better gradients**: Average gradients across batch

---

## 🎯 One Complete Training Iteration (Detailed)

### The Training Loop Anatomy

```python
for epoch in range(num_epochs):  # e.g., 10 epochs
    for batch in train_loader:   # e.g., 313 batches
        # === ONE ITERATION ===
        
        # 1. Get batch data
        sequences, contact_maps = batch  # (32, L, 4), (32, L, L)
        
        # 2. Zero gradients
        optimizer.zero_grad()
        
        # 3. Forward pass
        predictions = model(sequences)  # (32, L, L)
        
        # 4. Compute loss
        loss = criterion(predictions, contact_maps)  # Scalar
        
        # 5. Backward pass
        loss.backward()  # Compute gradients
        
        # 6. Update weights
        optimizer.step()  # Adjust parameters
```

### Step-by-Step Explanation

#### **Step 1: Get Batch Data**
```python
sequences, contact_maps = next(train_loader)

# sequences: (batch_size=32, max_len=100, features=4)
# contact_maps: (batch_size=32, max_len=100, max_len=100)

# Example for first sample in batch:
# sequences[0]: One-hot encoded RNA sequence
# contact_maps[0]: Ground truth contact map
```

#### **Step 2: Zero Gradients**
```python
optimizer.zero_grad()

# Why? Gradients accumulate by default in PyTorch
# Must clear old gradients before computing new ones
```

#### **Step 3: Forward Pass**
```python
predictions = model(sequences)

# What happens inside:
# 1. Expand 1D to 2D: (32, 100, 4) → (32, 100, 100, 8)
# 2. Conv layer 1: (32, 100, 100, 8) → (32, 100, 100, 32)
# 3. Conv layer 2: (32, 100, 100, 32) → (32, 100, 100, 16)
# 4. Conv layer 3: (32, 100, 100, 16) → (32, 100, 100, 1)
# 5. Sigmoid: Convert to probabilities [0, 1]
# 6. Output: (32, 100, 100) - predicted contact maps
```

#### **Step 4: Compute Loss**
```python
loss = criterion(predictions, contact_maps)

# criterion = BCEWithLogitsLoss (Binary Cross-Entropy)
# Measures how different predictions are from ground truth

# Example:
# If prediction = 0.9 and truth = 1 → low loss (good)
# If prediction = 0.1 and truth = 1 → high loss (bad)

# Loss is averaged across all positions and all samples in batch
# Result: Single scalar value (e.g., 0.65)
```

#### **Step 5: Backward Pass**
```python
loss.backward()

# Automatic differentiation (calculus)
# Computes gradient of loss w.r.t. every parameter

# For each weight w in the model:
# gradient = ∂loss/∂w (partial derivative)
# Tells us: "If we increase w, how much does loss change?"

# PyTorch does this automatically for all millions of parameters!
```

#### **Step 6: Update Weights**
```python
optimizer.step()

# Update rule (simplified):
# new_weight = old_weight - learning_rate × gradient

# If gradient is positive → decrease weight
# If gradient is negative → increase weight
# Goal: Move weights in direction that reduces loss
```

---

## 📊 What Happens Over Many Epochs

### Training Progression

```
EPOCH 1:
├─ Batch 1:  Loss = 0.8, Model is random
├─ Batch 2:  Loss = 0.79, Slightly better
├─ Batch 3:  Loss = 0.78
...
└─ Batch 313: Loss = 0.5, Model learned some patterns
   → Validation: F1 = 0.05 (still poor)

EPOCH 2:
├─ Batch 1:  Loss = 0.48, Starting from Epoch 1's end
├─ Batch 2:  Loss = 0.47
...
└─ Batch 313: Loss = 0.35
   → Validation: F1 = 0.10 (improving!)

...

EPOCH 10:
├─ Batch 1:  Loss = 0.15
├─ Batch 2:  Loss = 0.14
...
└─ Batch 313: Loss = 0.12
   → Validation: F1 = 0.25 (best so far!)
```

### Key Observations
1. **Loss decreases**: Model gets better at predictions
2. **Validation improves**: Model generalizes to unseen data
3. **Eventually plateaus**: Model reaches its capacity

---

## 🎛️ Important Hyperparameters

### Learning Rate
**What**: How big each weight update is

```python
optimizer = optim.Adam(model.parameters(), lr=0.001)
                                           ↑
                                      learning rate
```

- **Too high** (lr=0.1): Model diverges, loss explodes
- **Too low** (lr=0.00001): Model learns too slowly
- **Good range**: 0.001 - 0.0001 for Adam

### Batch Size
**What**: Number of samples processed together

```python
DataLoader(dataset, batch_size=32)
                    ↑
                 batch size
```

- **Larger** (64, 128): Faster training, more stable gradients
- **Smaller** (8, 16): More updates per epoch, better for small datasets
- **Good range**: 16-64 for this assignment

### Number of Epochs
**What**: How many times to see entire dataset

```python
for epoch in range(10):  # 10 epochs
```

- **Too few** (2-3): Underfitting, model doesn't learn enough
- **Too many** (100+): Overfitting, model memorizes training data
- **Good range**: 10-30 for this assignment

---

## 🧪 Training vs Validation vs Testing

### Three Dataset Splits

#### **Training Set** (TR0.csv)
- **Purpose**: Model learns from this
- **Size**: Largest (60-80% of data)
- **Usage**: Update weights, backpropagation
- **Metrics**: Training loss (should decrease)

#### **Validation Set** (VL0.csv)
- **Purpose**: Tune hyperparameters, early stopping
- **Size**: Medium (10-20% of data)
- **Usage**: Check generalization after each epoch
- **Metrics**: Validation loss, F1 score

#### **Test Set** (TS0.csv)
- **Purpose**: Final evaluation (touch ONCE at the end)
- **Size**: Smallest (10-20% of data)
- **Usage**: Report final performance
- **Metrics**: Final F1, AUC scores

### Why Three Splits?

```
Training: Model learns patterns
    ↓
Validation: Check if patterns generalize (adjust if needed)
    ↓
Testing: Final unbiased evaluation (no adjustment)
```

**Important**: Never use test set during training!

---

## 🎨 Visualization of Training

### Loss Curves (What You'll Plot)

```
Loss
 ↑
 │     Training Loss
 │    ╱╲
 │   ╱  ╲          Validation Loss
 │  ╱    ╲        ╱
 │ ╱      ╲╱╲   ╱
 │╱          ╲╱────────────  ← Plateaus
 └──────────────────────────→ Epochs
  1    5    10   15   20

Good training:
- Training loss decreases
- Validation loss decreases (but slower)
- Both plateau around same value
```

### Overfitting (Bad)

```
Loss
 ↑
 │              Validation Loss
 │             ╱───────  ← Starts increasing!
 │            ╱
 │     Training Loss
 │    ╱
 │   ╱
 │  ╱
 │ ╱
 │╱
 └──────────────────────────→ Epochs

Overfitting:
- Training loss keeps decreasing
- Validation loss starts increasing
- Model memorizes training data
```

---

## 🔑 Key Concepts Summary

### 1. **Forward Pass**
Data flows through network: Input → Layers → Output

### 2. **Loss Function**
Measures prediction error (how wrong we are)

### 3. **Backward Pass**
Compute gradients using calculus (automatic)

### 4. **Gradient Descent**
Update weights to reduce loss

### 5. **Epoch**
One complete pass through training data

### 6. **Batch**
Subset of data processed together

### 7. **Learning Rate**
Step size for weight updates

### 8. **Overfitting**
Model memorizes training data, fails on new data

---

## 💡 Intuitive Analogies

### Learning to Play Basketball

#### **Epochs** = Seasons
- Each season, you practice all the drills (dataset)
- After 10 seasons, you're much better

#### **Batches** = Practice sessions
- Can't practice everything at once
- Break into smaller sessions (batches)

#### **Forward Pass** = Taking a shot
- Execute your learned technique

#### **Loss** = Miss distance
- How far did you miss?

#### **Backward Pass** = Analyzing the miss
- What did I do wrong?

#### **Gradient Descent** = Adjusting form
- Make small adjustments to improve

#### **Learning Rate** = Adjustment size
- Too big: Over-correct, lose balance
- Too small: Takes forever to improve

#### **Overfitting** = Only good in practice gym
- Great on practice court (training set)
- Terrible in real games (test set)

---

## 🎯 What You'll Actually Code

### Minimal Training Loop (Skeleton)

```python
# Setup
model = RNAFoldingCNN()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCEWithLogitsLoss()

# Training
for epoch in range(10):  # 10 epochs
    
    # Training phase
    model.train()
    for sequences, contact_maps in train_loader:
        optimizer.zero_grad()
        predictions = model(sequences)
        loss = criterion(predictions, contact_maps)
        loss.backward()
        optimizer.step()
    
    # Validation phase
    model.eval()
    with torch.no_grad():  # Don't compute gradients
        for sequences, contact_maps in val_loader:
            predictions = model(sequences)
            # Calculate F1, AUC, etc.
```

This is the core of Part 3!

---

## 📚 Quick Reference

| Term | Simple Definition |
|------|------------------|
| **CNN** | Network that finds patterns using convolution |
| **2D CNN** | Convolution on 2D grids (images, contact maps) |
| **Epoch** | One pass through entire dataset |
| **Batch** | Subset processed together |
| **Forward Pass** | Input → Network → Output |
| **Loss** | How wrong predictions are |
| **Backward Pass** | Compute gradients |
| **Optimizer** | Updates weights using gradients |
| **Learning Rate** | Step size for updates |
| **Overfitting** | Memorizing training data |
| **Validation** | Check generalization |
| **Test** | Final evaluation |

You now have the complete conceptual foundation to tackle this assignment!
