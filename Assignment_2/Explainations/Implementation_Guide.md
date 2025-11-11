# RNA Structure Prediction - Implementation Guide

## Overview

This guide explains every component of the CNN-based RNA structure prediction implementation. It covers design decisions, parameter choices, and experimentation options.

---

## Part 1: Data Loading and Preprocessing

### 1.1 Data Loading Function

```python
def load_data_from_csv(file_path):
    df = pd.read_csv(file_path)
    data_tuples = [(row['sequence'], row['structure']) for _, row in df.iterrows()]
    return data_tuples
```

**What it does:**
- Reads CSV file containing RNA sequences and their secondary structures
- Extracts two columns: `sequence` (AUGC letters) and `structure` (dot-bracket notation)
- Returns list of tuples for easy iteration

**Design choice:** Simple list of tuples is memory-efficient for datasets that fit in RAM. For larger datasets, consider using generators.

---

### 1.2 One-Hot Encoding

```python
def one_hot_encode(sequence, max_len):
    base_to_int = {'A': 0, 'C': 1, 'G': 2, 'U': 3, 'T': 3}
    encoded = np.zeros((max_len, 4), dtype=np.float32)
    sequence = sequence[:max_len]
    
    for i, base in enumerate(sequence):
        if base in base_to_int:
            encoded[i, base_to_int[base]] = 1
    
    return encoded
```

**What it does:**
- Converts RNA sequence letters to numerical format
- Creates matrix of size (max_len, 4) where each row is one-hot vector
- Handles truncation if sequence is too long

**Key decisions:**

**1. Base mapping:**
```
A -> [1, 0, 0, 0]
C -> [0, 1, 0, 0]
G -> [0, 0, 1, 0]
U -> [0, 0, 0, 1]
T -> [0, 0, 0, 1]  (same as U)
```
- Why one-hot? Treats all bases equally with no ordinal relationship
- Why T=U? DNA uses T, RNA uses U, but they're chemically similar

**2. Data type: float32**
- Choice: `dtype=np.float32` instead of float64
- Why? CNNs use float32 by default, saves memory, faster on GPU
- Alternative: float64 for higher precision (rarely needed)

**3. Unknown bases:**
- Current: Ignored (stays as zeros)
- Alternative: Add 5th category for unknown bases
- Trade-off: More parameters vs cleaner data

**What to experiment with:**
- Different base orderings (minimal impact)
- Adding position encoding (may help with long sequences)

---

### 1.3 Contact Map Creation

```python
def create_contact_map(dot_bracket, max_len):
    dot_bracket = dot_bracket[:max_len]
    contact_map = np.zeros((max_len, max_len), dtype=np.float32)
    
    stack = []
    for i, char in enumerate(dot_bracket):
        if char == '(':
            stack.append(i)
        elif char == ')' and stack:
            j = stack.pop()
            contact_map[i, j] = 1
            contact_map[j, i] = 1
    
    return contact_map
```

**What it does:**
- Converts dot-bracket notation to 2D binary matrix
- Uses stack algorithm to match opening/closing parentheses
- Creates symmetric matrix (if i pairs with j, then j pairs with i)

**Algorithm explanation:**
```
Structure: "((...))"
Position:   0123456

Step-by-step:
i=0, char='(' -> push 0 to stack -> stack=[0]
i=1, char='(' -> push 1 to stack -> stack=[0,1]
i=2, char='.' -> nothing -> stack=[0,1]
i=3, char='.' -> nothing -> stack=[0,1]
i=4, char='.' -> nothing -> stack=[0,1]
i=5, char=')' -> pop 1 from stack, mark (1,5) and (5,1) as paired
i=6, char=')' -> pop 0 from stack, mark (0,6) and (6,0) as paired

Result: Positions 0-6 paired, 1-5 paired
```

**Key decisions:**

**1. Only handles '(' and ')'**
- Current: Only processes parentheses
- Why? Standard dot-bracket uses () for base pairs, . for unpaired
- Alternative: Handle brackets [] and braces {} for pseudoknots
- Trade-off: More complex structures vs simpler implementation

**2. Symmetry enforcement:**
- Sets both contact_map[i,j] = 1 AND contact_map[j,i] = 1
- Why? Base pairs are bidirectional (if A pairs with B, B pairs with A)
- Critical for model training and evaluation

**What to experiment with:**
- Add support for pseudoknots (nested brackets)
- Weight different pair types differently

---

## Part 2: Dataset and DataLoader

### 2.1 PyTorch Dataset Class

```python
class RNADataset(Dataset):
    def __init__(self, data, max_len):
        self.data = data
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sequence, structure = self.data[idx]
        seq_encoded = one_hot_encode(sequence, self.max_len)
        contact_map = create_contact_map(structure, self.max_len)
        seq_tensor = torch.from_numpy(seq_encoded).float()
        contact_tensor = torch.from_numpy(contact_map).float()
        return seq_tensor, contact_tensor
```

**What it does:**
- Wraps data in PyTorch's Dataset interface for efficient batching
- Applies encoding transformations on-the-fly
- Returns tensors ready for GPU processing

**Key decisions:**

**1. On-the-fly encoding vs pre-encoding:**
- Current: Encode during `__getitem__` call
- Pro: Lower memory usage
- Con: Slightly slower per epoch
- Alternative: Pre-encode all data in `__init__`
- When to pre-encode? If dataset is small and training is slow

**2. Tensor conversion:**
```python
torch.from_numpy(array).float()
```
- `.float()` ensures float32 type (required for CNN)
- Alternative: `.double()` for float64 (usually unnecessary)

---

### 2.2 Hyperparameter Choices

```python
MAX_LEN = 128
BATCH_SIZE = 16
```

**MAX_LEN = 128**

**What it controls:** Maximum sequence length, sequences are truncated or padded to this size

**Current choice:** 128 bases

**Why?**
- Most RNA sequences in dataset are 100-150 bases
- Power of 2 (efficient for GPU operations)
- Reasonable memory usage

**Trade-offs:**
- Too small (64): Lose information from longer sequences, faster training
- Too large (256, 512): More memory, slower training, captures longer sequences
- Check your data: `max([len(s) for s, _ in train_data])`

**What to experiment with:**
```python
MAX_LEN = 64   # Faster, less memory, may truncate important sequences
MAX_LEN = 256  # Slower, more memory, captures longer sequences
```

**BATCH_SIZE = 16**

**What it controls:** Number of samples processed together before weight update

**Current choice:** 16 samples per batch

**Why?**
- Balanced between memory usage and gradient stability
- Fits comfortably in 4-8GB GPU memory
- Good compromise for small datasets

**Trade-offs:**
- Smaller (4, 8): Less memory, noisier gradients, slower convergence
- Larger (32, 64): More memory, smoother gradients, may overfit
- Rule of thumb: As large as your GPU memory allows

**What to experiment with:**
```python
BATCH_SIZE = 8   # If running out of GPU memory
BATCH_SIZE = 32  # If you have 8GB+ GPU and want faster training
BATCH_SIZE = 1   # Extreme: stochastic gradient descent, very noisy
```

**Memory calculation:**
```
Memory per sample = MAX_LEN × MAX_LEN × 4 bytes (for float32)
                  = 128 × 128 × 4 = 65,536 bytes = 64 KB

Memory per batch = BATCH_SIZE × 64 KB
                 = 16 × 64 KB = 1 MB (just for contact maps)

Plus: Model parameters, activations, gradients (multiplier of ~3-5x)
Total estimate: 5-10 MB per batch + model size
```

---

### 2.3 DataLoader Configuration

```python
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
```

**Key parameters:**

**1. shuffle:**
- Train: `shuffle=True` - randomize order each epoch (prevents overfitting to order)
- Val/Test: `shuffle=False` - consistent evaluation order

**2. num_workers:**
- Current: `num_workers=0` - single process data loading
- Why? Simplicity, avoids multiprocessing issues on Windows
- Alternative: `num_workers=2` or `4` for faster data loading on Linux/Mac
- Trade-off: Faster loading vs compatibility issues

**What to experiment with:**
```python
num_workers=2  # Parallel data loading, faster on Unix systems
num_workers=4  # Even faster, but may cause issues on Windows
```

---

## Part 3: Model Architecture

### 3.1 CNN Structure

```python
class RNAFoldingCNN(nn.Module):
    def __init__(self, input_channels=8):
        super(RNAFoldingCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.relu3 = nn.ReLU()
        
        self.conv4 = nn.Conv2d(32, 1, kernel_size=1)
```

**Architecture decisions:**

**1. Input channels = 8**
- Why 8? Concatenation of two 4-dimensional one-hot vectors
- Each position pair (i,j) has 4 features for base i + 4 features for base j = 8 total

**2. Channel progression: 8 -> 32 -> 64 -> 32 -> 1**
- Expansion (32, 64): Learn increasingly complex features
- Reduction (32, 1): Compress to final prediction
- Why this pattern? Common in encoder-decoder architectures

**What to experiment with:**
```python
# Deeper network
self.conv4 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
self.bn4 = nn.BatchNorm2d(16)
self.relu4 = nn.ReLU()
self.conv5 = nn.Conv2d(16, 1, kernel_size=1)

# Wider network
self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

# Smaller network (faster, less overfitting)
self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, padding=1)
self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
self.conv3 = nn.Conv2d(32, 1, kernel_size=1)
```

**3. Kernel size = 3**
- Current: 3x3 kernels for all conv layers except final
- Why? Standard choice, captures local patterns
- Receptive field: Each layer sees 3x3 neighborhood

**What to experiment with:**
```python
kernel_size=5  # Larger receptive field, more parameters
kernel_size=7  # Even larger, may capture longer-range interactions
```

**4. Padding = 1**
- Maintains spatial dimensions (output size = input size)
- Formula: output_size = (input_size + 2*padding - kernel_size) / stride + 1
- With padding=1, kernel_size=3, stride=1: output_size = input_size

**5. Final layer: 1x1 convolution**
- Reduces channels to 1 (binary classification per position)
- Acts as learned linear combination of features

---

### 3.2 Forward Pass

```python
def forward(self, x_1d):
    batch_size = x_1d.shape[0]
    max_len = x_1d.shape[1]
    
    # Expand 1D to 2D
    x_2d_i = x_1d.unsqueeze(2).repeat(1, 1, max_len, 1)
    x_2d_j = x_1d.unsqueeze(1).repeat(1, max_len, 1, 1)
    x_2d = torch.cat([x_2d_i, x_2d_j], dim=-1)
    x_2d = x_2d.permute(0, 3, 1, 2)
    
    # Convolutional layers
    x = self.relu1(self.bn1(self.conv1(x_2d)))
    x = self.relu2(self.bn2(self.conv2(x)))
    x = self.relu3(self.bn3(self.conv3(x)))
    x = self.conv4(x)
    
    x = x.squeeze(1)
    
    # Symmetry and sigmoid
    x = torch.sigmoid(x)
    x = (x + x.transpose(1, 2)) / 2
    
    return x
```

**Critical steps:**

**1. 1D to 2D expansion:**
```python
x_1d.shape = (batch, max_len, 4)

x_2d_i = x_1d.unsqueeze(2).repeat(1, 1, max_len, 1)
# shape: (batch, max_len, max_len, 4)
# For each row i, repeat the same vector across all columns

x_2d_j = x_1d.unsqueeze(1).repeat(1, max_len, 1, 1)
# shape: (batch, max_len, max_len, 4)
# For each column j, repeat the same vector across all rows

x_2d = torch.cat([x_2d_i, x_2d_j], dim=-1)
# shape: (batch, max_len, max_len, 8)
# Concatenate features: [base_i_features, base_j_features]
```

**Visual example:**
```
Input: sequence "AUGC" encoded as (4, 4) one-hot

x_2d_i: Rows repeat their encoding
    A A A A
    U U U U
    G G G G
    C C C C

x_2d_j: Columns repeat their encoding
    A U G C
    A U G C
    A U G C
    A U G C

Concatenated (each cell has 8 features):
    [A,A] [A,U] [A,G] [A,C]
    [U,A] [U,U] [U,G] [U,C]
    [G,A] [G,U] [G,G] [G,C]
    [C,A] [C,U] [C,G] [C,C]
```

**2. Permute for CNN:**
```python
x_2d = x_2d.permute(0, 3, 1, 2)
# From: (batch, max_len, max_len, 8)
# To:   (batch, 8, max_len, max_len)
```
- PyTorch CNNs expect channels-first format: (batch, channels, height, width)

**3. Symmetry enforcement:**
```python
x = (x + x.transpose(1, 2)) / 2
```
- Ensures contact_map[i,j] = contact_map[j,i]
- Why? Base pairs are symmetric by nature
- Applied after sigmoid to preserve probability interpretation

**What to experiment with:**
```python
# Apply symmetry before sigmoid
x = (x + x.transpose(1, 2)) / 2
x = torch.sigmoid(x)

# No symmetry enforcement (let model learn it)
x = torch.sigmoid(x)
# May result in asymmetric predictions
```

---

### 3.3 Activation Functions

**ReLU (Rectified Linear Unit):**
```python
self.relu1 = nn.ReLU()
```
- Function: f(x) = max(0, x)
- Why? Simple, fast, prevents vanishing gradients
- Alternative activations:
  - LeakyReLU: `nn.LeakyReLU(0.1)` - allows small negative values
  - ELU: `nn.ELU()` - smooth negative values
  - GELU: `nn.GELU()` - used in transformers

**Sigmoid (final layer):**
```python
x = torch.sigmoid(x)
```
- Function: f(x) = 1 / (1 + e^(-x))
- Outputs values between 0 and 1 (probabilities)
- Why? Binary classification at each position

---

### 3.4 Batch Normalization

```python
self.bn1 = nn.BatchNorm2d(32)
```

**What it does:**
- Normalizes activations to have mean=0, std=1 per batch
- Adds learnable scale and shift parameters

**Benefits:**
1. Faster training (higher learning rates possible)
2. Reduced sensitivity to initialization
3. Slight regularization effect

**When to remove:**
- Very small batch sizes (< 4)
- When batch statistics are unreliable

**Alternative:**
```python
nn.LayerNorm([32, max_len, max_len])  # Normalize per sample instead of per batch
```

---

## Part 4: Training Configuration

### 4.1 Loss Function

```python
criterion = nn.BCELoss()
```

**Binary Cross-Entropy Loss:**
- Formula: -[y*log(pred) + (1-y)*log(1-pred)]
- For binary classification (paired vs not paired)

**Why BCELoss instead of BCEWithLogitsLoss?**
- Model already applies sigmoid in forward pass
- BCEWithLogitsLoss combines sigmoid + BCE for numerical stability
- Current setup is fine since we use sigmoid explicitly

**Alternative (more stable):**
```python
# Remove sigmoid from model forward pass
criterion = nn.BCEWithLogitsLoss()
# Loss function applies sigmoid internally
```

**Weighted loss for imbalanced data:**
```python
# Count positive samples
num_positives = sum([contact_map.sum() for _, contact_map in train_dataset])
num_total = len(train_dataset) * MAX_LEN * MAX_LEN
pos_weight = (num_total - num_positives) / num_positives

criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
```

---

### 4.2 Optimizer

```python
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

**Adam optimizer:**
- Adaptive learning rate per parameter
- Combines momentum and RMSProp
- Default choice for most deep learning tasks

**Learning rate = 0.001:**
- Standard default for Adam
- Not too fast (unstable), not too slow

**What to experiment with:**
```python
# Lower learning rate (more stable, slower)
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Higher learning rate (faster, may be unstable)
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3
)
# Reduces LR when validation loss plateaus
```

**Alternative optimizers:**
```python
# SGD with momentum
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# AdamW (Adam with weight decay)
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
```

---

### 4.3 Number of Epochs

```python
num_epochs = 20
```

**Current choice:** 20 epochs

**Why?**
- Reasonable for initial experiments
- Usually sufficient to see convergence trends
- Not so long that you wait forever

**How to determine optimal number:**
- Monitor validation loss curve
- Stop when validation loss stops improving (early stopping)

**What to experiment with:**
```python
num_epochs = 10   # Quick experiment
num_epochs = 50   # More thorough training
num_epochs = 100  # Full training run
```

**Early stopping implementation:**
```python
patience = 5
patience_counter = 0
best_val_loss = float('inf')

for epoch in range(num_epochs):
    # ... training code ...
    
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
```

---

## Part 5: Training Loop

### 5.1 Training Phase

```python
model.train()
train_loss = 0.0

for batch_x, batch_y in train_loader:
    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
    
    optimizer.zero_grad()
    outputs = model(batch_x)
    loss = criterion(outputs, batch_y)
    loss.backward()
    optimizer.step()
    
    train_loss += loss.item()
```

**Key steps:**

**1. model.train()**
- Enables training mode
- Turns on dropout (if used) and batch norm updates

**2. .to(device)**
- Moves data to GPU if available
- Critical for GPU acceleration

**3. optimizer.zero_grad()**
- Clears gradients from previous iteration
- Must be called before each backward pass

**4. loss.backward()**
- Computes gradients via backpropagation
- Fills .grad attribute of all parameters

**5. optimizer.step()**
- Updates weights using computed gradients
- Implements weight = weight - lr * gradient (with Adam modifications)

---

### 5.2 Validation Phase

```python
model.eval()
val_loss = 0.0

with torch.no_grad():
    for batch_x, batch_y in val_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        val_loss += loss.item()
```

**Key differences from training:**

**1. model.eval()**
- Disables training mode
- Turns off dropout, uses batch norm running statistics

**2. torch.no_grad()**
- Disables gradient computation
- Saves memory and speeds up inference
- Critical for validation/testing

**Why separate validation?**
- Detect overfitting (train loss low, val loss high)
- Select best model based on unseen data performance

---

### 5.3 Metrics Calculation

```python
def calculate_metrics(y_pred, y_true, threshold=0.5):
    y_pred_flat = y_pred.cpu().detach().numpy().flatten()
    y_true_flat = y_true.cpu().detach().numpy().flatten()
    
    y_pred_binary = (y_pred_flat >= threshold).astype(int)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true_flat, y_pred_binary, average='binary', zero_division=0
    )
    
    try:
        auc = roc_auc_score(y_true_flat, y_pred_flat)
    except:
        auc = 0.0
    
    return f1, recall, auc
```

**Metrics explained:**

**1. F1 Score:**
- Harmonic mean of precision and recall
- Formula: F1 = 2 * (precision * recall) / (precision + recall)
- Good for imbalanced data
- Values: 0 (worst) to 1 (best)

**2. Recall:**
- Fraction of actual positives correctly identified
- Formula: TP / (TP + FN)
- Important for not missing base pairs

**3. AUC (Area Under ROC Curve):**
- Measures ranking quality across all thresholds
- Values: 0.5 (random) to 1.0 (perfect)
- Less sensitive to class imbalance than F1

**Threshold parameter:**
```python
threshold=0.5  # Default
```
- Probabilities >= 0.5 classified as 1 (paired)
- Probabilities < 0.5 classified as 0 (not paired)

**What to experiment with:**
```python
threshold=0.3  # More sensitive, higher recall, lower precision
threshold=0.7  # More conservative, lower recall, higher precision
```

**Why .cpu().detach().numpy()?**
- `.detach()`: Remove from computation graph
- `.cpu()`: Move from GPU to CPU
- `.numpy()`: Convert PyTorch tensor to NumPy array (required by sklearn)

---

## Part 6: Saving and Loading Models

### 6.1 Saving Best Model

```python
if avg_val_f1 > best_val_f1:
    best_val_f1 = avg_val_f1
    torch.save(model.state_dict(), 'best_model.pth')
```

**What gets saved:**
- Model weights (parameters)
- Optimizer state NOT saved (only weights)

**Alternative (save everything):**
```python
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'val_f1': avg_val_f1,
}, 'checkpoint.pth')
```

### 6.2 Loading Model

```python
model.load_state_dict(torch.load('best_model.pth'))
model.eval()
```

**Critical:** Call `.eval()` after loading for inference

---

## Part 7: Visualization and Analysis

### 7.1 Training Curves

```python
axes[0, 0].plot(train_losses, label='Train Loss', marker='o')
axes[0, 0].plot(val_losses, label='Val Loss', marker='s')
```

**What to look for:**

**1. Normal training:**
- Both losses decrease
- Val loss follows train loss closely
- Gap is small and stable

**2. Overfitting:**
- Train loss continues decreasing
- Val loss plateaus or increases
- Large gap between curves

**3. Underfitting:**
- Both losses high and not decreasing
- Model too simple or learning rate too low

**4. Good regularization:**
- Small gap between train and val loss
- Both decrease smoothly

---

### 7.2 Contact Map Visualization

```python
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].imshow(true_contact, cmap='Blues')
axes[1].imshow(pred_contact, cmap='Reds')
axes[2].imshow(pred_contact_binary, cmap='Greens')
```

**What to analyze:**

**1. True contact map:**
- Shows actual base pairings
- Symmetric matrix
- Diagonal should be empty (base can't pair with itself)

**2. Predicted probabilities:**
- Smooth gradients indicate model uncertainty
- Sharp predictions indicate confidence

**3. Binary predictions:**
- After thresholding
- Compare with ground truth
- Look for patterns: near diagonal (local pairs) vs far from diagonal (long-range)

---

## Part 8: Experimentation Guide

### Quick Experiments (Fast to Test)

**1. Learning rate:**
```python
lr = 0.01    # Try this first if training is slow
lr = 0.0001  # Try this if loss is unstable
```

**2. Batch size:**
```python
BATCH_SIZE = 8   # Reduce if out of memory
BATCH_SIZE = 32  # Increase for smoother gradients
```

**3. Threshold:**
```python
threshold = 0.3  # Improve F1 for imbalanced data
threshold = 0.7  # Improve precision
```

### Medium Experiments (Require Retraining)

**1. Max length:**
```python
MAX_LEN = 64   # Faster training
MAX_LEN = 256  # Capture longer sequences
```

**2. Model size:**
```python
# Smaller (faster, less overfitting)
nn.Conv2d(input_channels, 16, kernel_size=3, padding=1)
nn.Conv2d(16, 32, kernel_size=3, padding=1)

# Larger (more capacity, slower)
nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
nn.Conv2d(64, 128, kernel_size=3, padding=1)
```

**3. Number of epochs:**
```python
num_epochs = 10   # Quick test
num_epochs = 50   # Full training
```

### Advanced Experiments (Significant Changes)

**1. Add dropout:**
```python
self.dropout = nn.Dropout2d(0.2)

# In forward:
x = self.relu1(self.bn1(self.conv1(x_2d)))
x = self.dropout(x)
```

**2. Weighted loss:**
```python
pos_weight = calculate_positive_weight(train_dataset)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
```

**3. Different architecture:**
```python
# Add residual connections
class ResidualBlock(nn.Module):
    def forward(self, x):
        residual = x
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out + residual
```

**4. Data augmentation:**
```python
# Reverse complement
def augment_sequence(seq):
    complement = {'A': 'U', 'U': 'A', 'G': 'C', 'C': 'G'}
    return ''.join([complement[b] for b in reversed(seq)])
```

---

## Part 9: Troubleshooting

### Problem: Out of Memory

**Solutions:**
1. Reduce batch size: `BATCH_SIZE = 8` or `4`
2. Reduce max length: `MAX_LEN = 64`
3. Use gradient accumulation:
```python
accumulation_steps = 4
optimizer.zero_grad()

for i, (batch_x, batch_y) in enumerate(train_loader):
    loss = criterion(model(batch_x), batch_y)
    loss = loss / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### Problem: Model Not Learning

**Check:**
1. Learning rate too high or too low
2. Data encoding correct (verify one sample manually)
3. Loss function appropriate for task
4. Model architecture has enough capacity

**Solutions:**
1. Try different learning rates: 0.0001, 0.001, 0.01
2. Visualize first batch predictions
3. Check gradient flow:
```python
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: {param.grad.abs().mean()}")
```

### Problem: Overfitting

**Signs:**
- Train loss much lower than val loss
- Val F1 decreases while train F1 increases

**Solutions:**
1. Add dropout
2. Reduce model size
3. Early stopping
4. Increase training data (if possible)
5. Data augmentation

### Problem: Very Low F1 Score

**Expected for this task:**
- F1 around 0.1-0.3 is normal for simple CNN on RNA structure
- Contact maps are highly imbalanced

**Solutions:**
1. Lower threshold: `threshold=0.3`
2. Weighted loss function
3. Focus on AUC instead (less sensitive to imbalance)

---

## Part 10: GPU Utilization

### Checking GPU Usage

```python
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Current device: {torch.cuda.get_device_name(0)}")
print(f"Memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"Memory cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
```

### Optimizing for GPU

**1. Pin memory:**
```python
train_loader = DataLoader(
    train_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True,
    pin_memory=True  # Faster GPU transfer
)
```

**2. Mixed precision training (faster, less memory):**
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch_x, batch_y in train_loader:
    optimizer.zero_grad()
    
    with autocast():
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

---

## Summary of Key Parameters to Experiment With

| Parameter | Current | Easy Alternatives | Expected Effect |
|-----------|---------|-------------------|-----------------|
| MAX_LEN | 128 | 64, 256 | Memory, sequence coverage |
| BATCH_SIZE | 16 | 8, 32 | Memory, gradient smoothness |
| Learning Rate | 0.001 | 0.0001, 0.01 | Training speed, stability |
| Num Epochs | 20 | 10, 50 | Training time, convergence |
| Threshold | 0.5 | 0.3, 0.7 | F1 score, precision/recall trade-off |
| Model Width | 32, 64 | 16, 128 | Capacity, training time |
| Kernel Size | 3 | 5, 7 | Receptive field, parameters |

---

## Recommended Experimentation Order

**1. Baseline (current implementation):**
- Run as-is to get baseline metrics

**2. Quick wins:**
- Adjust threshold for better F1
- Try different learning rates

**3. Data experiments:**
- Vary MAX_LEN based on your data
- Adjust BATCH_SIZE for your GPU

**4. Model experiments:**
- Try different model sizes
- Add/remove layers

**5. Advanced techniques:**
- Implement weighted loss
- Add dropout
- Try data augmentation

**6. Analysis:**
- Compare all experiments
- Identify best configuration
- Document findings

---

## File Outputs

The code generates:
1. `best_model.pth` - Saved model weights
2. `training_metrics.png` - Training curves
3. `contact_map_comparison.png` - Visualization of predictions

These files are created in the current working directory.
