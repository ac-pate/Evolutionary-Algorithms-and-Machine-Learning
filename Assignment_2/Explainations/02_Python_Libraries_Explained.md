# Python Libraries Explained - Deep Dive

## 🎯 Complete Import Breakdown for This Assignment

```python
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
```

---

## 📊 NumPy (`import numpy as np`)

### What It Is
Numerical computing library for arrays and matrices in Python.

### Why You Already Know It
- Core library for scientific computing
- Foundation for pandas, matplotlib, and PyTorch
- Fast array operations (written in C)

### How It's Used in THIS Assignment

#### 1. **Creating Contact Maps**
```python
# Initialize empty contact map
contact_map = np.zeros((max_len, max_len))  # 2D array

# Set paired positions to 1
contact_map[i, j] = 1
contact_map[j, i] = 1  # Symmetric
```

#### 2. **Array Manipulations**
```python
# One-hot encoding matrix
encoded = np.zeros((max_len, 4))
encoded[position, nucleotide_index] = 1
```

#### 3. **Before Converting to PyTorch**
```python
# NumPy array → PyTorch tensor
tensor = torch.from_numpy(numpy_array).float()
```

### Key Operations You'll Use
- `np.zeros()` - Create zero-filled arrays
- `np.array()` - Convert lists to arrays
- Array indexing: `arr[i, j]`
- Array slicing: `arr[:10]`

---

## 🔥 PyTorch Core (`import torch`)

### What It Is
Facebook's deep learning framework - competitor to TensorFlow.

### Why It's Popular
- More "Pythonic" and intuitive than TensorFlow
- Dynamic computation graphs (easier debugging)
- Strong academic adoption
- Excellent GPU support

### How It's Used in THIS Assignment

#### 1. **Tensors (Like NumPy Arrays, but GPU-capable)**
```python
# Create tensors
x = torch.tensor([1, 2, 3])
x = torch.zeros(10, 10)  # Like np.zeros

# Move to GPU
x = x.to(device)  # device = 'cuda' or 'cpu'
```

#### 2. **GPU Operations**
```python
# Check GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# All operations on tensors automatically use GPU
result = x @ y  # Matrix multiplication on GPU
```

#### 3. **Automatic Differentiation**
```python
# PyTorch tracks gradients automatically
x = torch.tensor([1.0, 2.0], requires_grad=True)
y = x ** 2
y.backward()  # Compute gradients
print(x.grad)  # dy/dx = 2x
```

### Key Differences from NumPy
| Feature | NumPy | PyTorch |
|---------|-------|---------|
| **GPU Support** | ❌ No | ✅ Yes |
| **Gradients** | ❌ Manual | ✅ Automatic |
| **Deep Learning** | ❌ Not built for it | ✅ Designed for it |

---

## 🧠 PyTorch Neural Networks (`torch.nn`)

### What It Is
Building blocks for neural networks.

### How It's Used in THIS Assignment

#### 1. **Base Class for Your Model**
```python
class RNAFoldingCNN(nn.Module):  # Inherit from nn.Module
    def __init__(self):
        super().__init__()
        # Define layers here
    
    def forward(self, x):
        # Define forward pass
        return output
```

#### 2. **Convolutional Layers**
```python
# 2D Convolution (the core of CNNs)
self.conv1 = nn.Conv2d(
    in_channels=8,    # Input: 8 feature maps
    out_channels=32,  # Output: 32 feature maps
    kernel_size=3,    # 3x3 filter
    padding=1         # Keep same size
)

# How it works:
# Input: (batch, 8, 100, 100)
# Output: (batch, 32, 100, 100)
# The 3x3 filter slides across the 100x100 grid
```

#### 3. **Activation Functions**
```python
self.relu = nn.ReLU()  # Rectified Linear Unit

# Converts negatives to 0, keeps positives
# f(x) = max(0, x)
# Example: [-1, 2, -3, 4] → [0, 2, 0, 4]
```

#### 4. **Normalization**
```python
self.bn = nn.BatchNorm2d(32)

# Normalizes layer outputs during training
# Helps training stability and speed
# μ=0, σ=1 for each feature map
```

#### 5. **Loss Functions**
```python
criterion = nn.BCEWithLogitsLoss()

# Binary Cross-Entropy loss
# For binary classification (paired vs unpaired)
# Combines sigmoid + cross-entropy for stability
```

### Complete Layer List You'll Use
```python
# In your CNN model:
nn.Conv2d(in, out, kernel_size)  # Convolutional layer
nn.ReLU()                         # Activation
nn.BatchNorm2d(channels)         # Normalization
nn.MaxPool2d(kernel_size)        # Downsampling (optional)
nn.Dropout2d(p=0.5)              # Regularization (optional)
```

---

## ⚙️ PyTorch Optimization (`torch.optim`)

### What It Is
Optimization algorithms that update model weights during training.

### How It's Used in THIS Assignment

#### 1. **Adam Optimizer (Most Popular)**
```python
optimizer = optim.Adam(
    model.parameters(),  # Which weights to update
    lr=0.001            # Learning rate (step size)
)
```

#### 2. **Training Loop Usage**
```python
# In each training iteration:
optimizer.zero_grad()     # Clear old gradients
loss.backward()           # Compute new gradients
optimizer.step()          # Update weights using gradients
```

### What Happens Internally
```python
# Simplified version of what optimizer.step() does:
for param in model.parameters():
    param.data -= learning_rate * param.grad
```

### Why Adam Instead of SGD?
| Optimizer | Speed | Stability | Tuning Needed |
|-----------|-------|-----------|---------------|
| **SGD** | Slow | Can be unstable | Lots of tuning |
| **Adam** | Fast | Stable | Works well by default |

Adam adapts learning rate per parameter - "smart" optimization.

---

## 📦 PyTorch Data Utilities (`torch.utils.data`)

### What It Is
Tools for efficient data loading and batching.

### How It's Used in THIS Assignment

#### 1. **Custom Dataset Class**
```python
class RNADataset(Dataset):
    def __init__(self, data, max_len):
        self.data = data
        self.max_len = max_len
    
    def __len__(self):
        # Return total number of samples
        return len(self.data)
    
    def __getitem__(self, idx):
        # Return one sample
        sequence, structure = self.data[idx]
        # Encode and return
        return encoded_seq, contact_map
```

#### 2. **DataLoader for Batching**
```python
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=32,        # Process 32 samples at once
    shuffle=True,         # Randomize order each epoch
    num_workers=4         # Parallel data loading
)

# Usage:
for batch_sequences, batch_contact_maps in train_loader:
    # batch_sequences: (32, max_len, 4)
    # batch_contact_maps: (32, max_len, max_len)
    predictions = model(batch_sequences)
```

### Why Use Dataset and DataLoader?

**Without DataLoader** (manual, inefficient):
```python
for i in range(0, len(data), batch_size):
    batch = data[i:i+batch_size]
    # Process batch...
```

**With DataLoader** (automatic, optimized):
```python
for batch in train_loader:
    # Automatically batched, shuffled, parallel loaded
    # Process batch...
```

---

## 📊 Matplotlib (`import matplotlib.pyplot as plt`)

### What You Already Know
- Plotting library for visualizations
- Similar to MATLAB plotting

### How It's Used in THIS Assignment

#### 1. **Loss Curves**
```python
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
```

#### 2. **Contact Map Visualization**
```python
plt.imshow(contact_map, cmap='hot', interpolation='nearest')
plt.colorbar()
plt.title('Predicted Contact Map')
plt.show()
```

#### 3. **Side-by-Side Comparison**
```python
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.imshow(true_contact_map)
ax1.set_title('Ground Truth')
ax2.imshow(predicted_contact_map)
ax2.set_title('Prediction')
plt.show()
```

### Key Functions You'll Use
- `plt.plot()` - Line plots
- `plt.imshow()` - Display 2D arrays as images
- `plt.xlabel/ylabel()` - Labels
- `plt.legend()` - Legend
- `plt.title()` - Title
- `plt.savefig()` - Save to file

---

## 🐼 Pandas (`import pandas as pd`)

### What You Already Know
- Data manipulation (like Excel in Python)
- DataFrames for tabular data

### How It's Used in THIS Assignment

#### **Reading CSV Files**
```python
# Load dataset
df = pd.read_csv('TR0.csv')

# Inspect
print(df.head())  # First 5 rows
print(df.columns)  # Column names
print(df.shape)    # (rows, columns)

# Extract data
sequences = df['sequence'].values  # NumPy array
structures = df['structure'].values

# Convert to list of tuples
data = list(zip(sequences, structures))
# Result: [('AUGC...', '((...'), ('CGAU...', '..(.')...]
```

### Likely CSV Structure
```
id,sequence,structure
1,AUGCGAUUCGAU,(((...)))...
2,CGAUUGCAUCGA,..((.))....
...
```

### Key Operations
```python
df.read_csv(path)         # Load CSV
df.head(n)                # First n rows
df['column'].values       # Get column as NumPy array
df.iloc[i]                # Get row by index
len(df)                   # Number of rows
```

---

## 📈 TQDM (`from tqdm import tqdm`)

### What It Is
Progress bar library - shows how long things will take.

### How It's Used in THIS Assignment

#### 1. **Training Loop Progress**
```python
for epoch in tqdm(range(num_epochs), desc='Training'):
    # Shows: Training: 40%|████      | 4/10 [00:12<00:18, 3.1s/it]
    train_one_epoch()
```

#### 2. **Batch Progress**
```python
for batch in tqdm(train_loader, desc=f'Epoch {epoch}'):
    # Shows progress through all batches
    train_batch(batch)
```

### Output Example
```
Training: 40%|████████████          | 4/10 [00:45<01:08, 11.4s/it]
         ↑       ↑                    ↑     ↑        ↑
      Percent  Progress bar      Current  Remaining Time/iter
```

### Simple Usage
```python
# Wrap any iterable
for item in tqdm(my_list):
    process(item)

# Custom description
for i in tqdm(range(100), desc='Processing'):
    work(i)
```

---

## 🎯 Scikit-learn Metrics (`from sklearn.metrics import roc_auc_score`)

### What It Is
Machine learning library - here we use it for evaluation metrics.

### How It's Used in THIS Assignment

#### **AUC Score (Area Under ROC Curve)**
```python
# Calculate AUC
auc = roc_auc_score(
    y_true=contact_map.flatten(),     # Ground truth (0s and 1s)
    y_score=predictions.flatten()      # Predicted probabilities
)

# AUC ranges from 0 to 1
# 0.5 = random guessing
# 1.0 = perfect predictions
```

### What AUC Measures
- How well the model **separates** positive and negative classes
- Robust to class imbalance (lots of 0s, few 1s)
- Measures **ranking quality**: Are positive samples scored higher than negative?

### Why AUC is High but F1 is Low (Important!)
```python
# In contact maps:
# - Most positions are unpaired (0s) - maybe 90%
# - Few positions are paired (1s) - maybe 10%

# AUC: Measures if model ranks 1s higher than 0s
# → Can be 0.9+ even if predictions are poor

# F1: Measures actual prediction accuracy
# → Low because hard to predict exact pairings
```

You'll analyze this in Part 4.2!

---

## 📚 Quick Reference Table

| Library | Main Purpose | Key Uses in Assignment |
|---------|-------------|------------------------|
| **NumPy** | Array operations | Contact maps, encoding |
| **PyTorch** | Deep learning framework | Tensors, GPU, automatic gradients |
| **torch.nn** | Neural network layers | CNN architecture |
| **torch.optim** | Optimization | Adam optimizer, weight updates |
| **torch.utils.data** | Data loading | Dataset class, DataLoader |
| **Matplotlib** | Plotting | Loss curves, contact map visualization |
| **Pandas** | Data manipulation | Read CSV files |
| **TQDM** | Progress bars | Show training progress |
| **Scikit-learn** | ML utilities | AUC score calculation |

---

## 🎯 The Complete Data Flow

```python
# 1. Load data (Pandas)
df = pd.read_csv('TR0.csv')

# 2. Convert to arrays (NumPy)
sequences = df['sequence'].values

# 3. Encode (NumPy)
encoded = np.zeros((len, 4))

# 4. Convert to tensors (PyTorch)
tensor = torch.from_numpy(encoded)

# 5. Create dataset (torch.utils.data)
dataset = RNADataset(data)

# 6. Create dataloader (torch.utils.data)
loader = DataLoader(dataset, batch_size=32)

# 7. Build model (torch.nn)
model = RNAFoldingCNN()

# 8. Set optimizer (torch.optim)
optimizer = optim.Adam(model.parameters())

# 9. Train (PyTorch + TQDM)
for epoch in tqdm(range(10)):
    for batch in loader:
        predictions = model(batch)
        loss.backward()
        optimizer.step()

# 10. Evaluate (scikit-learn)
auc = roc_auc_score(y_true, y_pred)

# 11. Visualize (Matplotlib)
plt.plot(losses)
plt.imshow(contact_map)
```

This is the exact pipeline you'll implement!
