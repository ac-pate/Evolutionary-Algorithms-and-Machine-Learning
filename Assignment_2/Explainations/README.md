# Quick Reference Guide - Assignment 2

## 📋 At-a-Glance Information

### Assignment Files
```
Assignment_2/
├── COEN432_Assignment2.ipynb  (Complete this)
├── TR0.csv                     (Training data)
├── VL0.csv                     (Validation data)
├── TS0.csv                     (Test data)
└── Explainations/             (These guides)
    ├── 00_Assignment_Overview.md
    ├── 01_GPU_Selection_Guide.md
    ├── 02_Python_Libraries_Explained.md
    ├── 03_Dataset_Explained.md
    ├── 04_CNN_and_Training_Explained.md
    └── 05_Time_Management_Strategy.md
```

---

## 🎯 Assignment at a Glance

**Goal**: Build CNN to predict RNA secondary structure (contact maps) from sequences

**Input**: RNA sequence string (e.g., "AUGCGAU")
**Output**: Contact map matrix (which bases pair)

**Pipeline**:
```
CSV → Pandas → NumPy encoding → PyTorch Dataset → CNN → Predictions → Evaluation
```

---

## 📊 Parts & Weights

| Part | Task | Weight | Time Est. |
|------|------|--------|-----------|
| 1 | Data Preprocessing | 35% | 3-5 hours |
| 2 | Model Implementation | 20% | 2-3 hours |
| 3 | Training & Evaluation | 25% | 3-4 hours |
| 4 | Analysis & Reporting | 20% | 2-3 hours |

---

## 🔧 Setup Checklist

### Installation (One-time)
```bash
# Install PyTorch with CUDA (for RTX 5070)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
pip install numpy pandas matplotlib tqdm scikit-learn jupyter
```

### Verification
```python
import torch
print(torch.cuda.is_available())  # Should be True
print(torch.cuda.get_device_name(0))  # Should show RTX 5070
```

---

## 📝 Key Code Snippets

### Load Data (Part 1.2)
```python
import pandas as pd

def load_data_from_csv(file_path):
    df = pd.read_csv(file_path)
    sequences = df['sequence'].values
    structures = df['structure'].values
    return list(zip(sequences, structures))

train_data = load_data_from_csv('TR0.csv')
```

### One-Hot Encode (Part 1.3)
```python
import numpy as np

def one_hot_encode(sequence, max_len):
    # Mapping: A=0, U=1, G=2, C=3
    mapping = {'A': 0, 'U': 1, 'G': 2, 'C': 3}
    
    # Initialize matrix
    encoded = np.zeros((max_len, 4))
    
    # Fill in one-hot values
    for i, base in enumerate(sequence[:max_len]):
        if base in mapping:
            encoded[i, mapping[base]] = 1
    
    return encoded
```

### Contact Map (Part 1.3)
```python
def create_contact_map(dot_bracket, max_len):
    # Initialize matrix
    contact_map = np.zeros((max_len, max_len))
    
    # Use stack to find pairs
    stack = []
    for i, char in enumerate(dot_bracket[:max_len]):
        if char in '([{':  # Opening brackets
            stack.append(i)
        elif char in ')]}':  # Closing brackets
            if stack:
                j = stack.pop()
                contact_map[i, j] = 1
                contact_map[j, i] = 1  # Symmetric
    
    return contact_map
```

### Dataset Class (Part 1.4)
```python
from torch.utils.data import Dataset
import torch

class RNADataset(Dataset):
    def __init__(self, data, max_len):
        self.data = data
        self.max_len = max_len
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sequence, structure = self.data[idx]
        
        # Encode
        seq_encoded = one_hot_encode(sequence, self.max_len)
        contact_map = create_contact_map(structure, self.max_len)
        
        # Convert to tensors
        seq_tensor = torch.from_numpy(seq_encoded).float()
        contact_tensor = torch.from_numpy(contact_map).float()
        
        return seq_tensor, contact_tensor
```

### CNN Model (Part 2.1)
```python
import torch.nn as nn

class RNAFoldingCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(8, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 16, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 1, 1)
    
    def forward(self, x_1d):
        # x_1d: (batch, max_len, 4)
        
        # Expand to 2D
        L = x_1d.shape[1]
        x_2d_i = x_1d.unsqueeze(2).repeat(1, 1, L, 1)  # (B, L, L, 4)
        x_2d_j = x_1d.unsqueeze(1).repeat(1, L, 1, 1)  # (B, L, L, 4)
        x = torch.cat([x_2d_i, x_2d_j], dim=-1)  # (B, L, L, 8)
        x = x.permute(0, 3, 1, 2)  # (B, 8, L, L) - channels first
        
        # Conv layers
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.conv3(x)
        
        # Make symmetric
        x = x.squeeze(1)
        x = (x + x.transpose(1, 2)) / 2
        
        return torch.sigmoid(x)
```

### Training Loop (Part 3.2)
```python
from tqdm import tqdm

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = RNAFoldingCNN().to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training
for epoch in tqdm(range(10), desc='Training'):
    model.train()
    epoch_loss = 0
    
    for sequences, contact_maps in train_loader:
        sequences = sequences.to(device)
        contact_maps = contact_maps.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        predictions = model(sequences)
        loss = criterion(predictions, contact_maps)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    print(f'Epoch {epoch+1}, Loss: {epoch_loss/len(train_loader):.4f}')
```

### Calculate F1 Score (Part 3.1)
```python
def calculate_metrics(y_pred, y_true, threshold=0.5):
    # Binarize predictions
    y_pred_binary = (y_pred > threshold).float()
    
    # Flatten
    y_pred_flat = y_pred_binary.flatten()
    y_true_flat = y_true.flatten()
    
    # Calculate TP, FP, FN
    TP = ((y_pred_flat == 1) & (y_true_flat == 1)).sum().item()
    FP = ((y_pred_flat == 1) & (y_true_flat == 0)).sum().item()
    FN = ((y_pred_flat == 0) & (y_true_flat == 1)).sum().item()
    
    # F1 Score
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return f1, recall
```

### Visualize Contact Map (Part 4.2)
```python
import matplotlib.pyplot as plt

# Get one sample
sequences, contact_maps = next(iter(test_loader))
seq = sequences[0:1].to(device)
true_map = contact_maps[0].cpu().numpy()

# Predict
model.eval()
with torch.no_grad():
    pred_map = model(seq).cpu().numpy()[0]

# Plot side-by-side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.imshow(true_map, cmap='hot', interpolation='nearest')
ax1.set_title('Ground Truth Contact Map')
ax1.set_xlabel('Position')
ax1.set_ylabel('Position')

ax2.imshow(pred_map, cmap='hot', interpolation='nearest')
ax2.set_title('Predicted Contact Map')
ax2.set_xlabel('Position')
ax2.set_ylabel('Position')

plt.tight_layout()
plt.show()
```

---

## 🎯 Common Gotchas & Solutions

### Issue 1: Shape Mismatches
**Error**: `RuntimeError: shape mismatch`

**Solution**: Print shapes everywhere!
```python
print(f"Sequence: {seq.shape}")  # Should be (batch, L, 4)
print(f"Contact: {contact.shape}")  # Should be (batch, L, L)
print(f"Prediction: {pred.shape}")  # Should be (batch, L, L)
```

### Issue 2: Contact Map Not Symmetric
**Problem**: `contact_map[i,j] != contact_map[j,i]`

**Solution**: Set both directions
```python
contact_map[i, j] = 1
contact_map[j, i] = 1  # Add this!
```

### Issue 3: Loss Not Decreasing
**Problem**: Loss stays constant or increases

**Solutions**:
- Check learning rate (try 0.001, 0.0001)
- Verify data is normalized (0-1 range)
- Check gradient flow: `print(model.conv1.weight.grad)`
- Reduce model complexity if overfitting

### Issue 4: Low F1 Score
**Not a bug!** F1 will be low (0.1-0.3) for this simple model.

**Why**: 
- Imbalanced data (90% unpaired)
- Simple CNN can't capture long-range dependencies
- This is expected and you'll analyze it in Part 4.2

---

## 📊 Expected Results

### Typical Performance
```
Training Loss:     0.6 → 0.1 (decreasing)
Validation Loss:   0.5 → 0.15 (decreasing)
Validation F1:     0.05 → 0.25 (low but improving)
Test F1:           0.15 - 0.30 (final)
AUC:               0.85 - 0.95 (high despite low F1)
```

### Why AUC High but F1 Low?
- **AUC**: Measures ranking (are 1s scored higher than 0s?)
  - Robust to imbalance
  - Can be high even with poor predictions
  
- **F1**: Measures precision/recall balance
  - Sensitive to imbalance
  - Low when predictions are imprecise

You'll explain this in Part 4.2!

---

## ⏱️ Time Checklist

### Must Do (Core)
- [ ] Load CSV data (30 min)
- [ ] One-hot encoding (30 min)
- [ ] Contact map creation (1-2 hours) ← Hardest part
- [ ] Dataset & DataLoader (30 min)
- [ ] CNN model (1-2 hours)
- [ ] Training loop (1 hour)
- [ ] Metrics (30 min)
- [ ] Visualization (30 min)
- [ ] Analysis (1 hour)

### Should Do (Important)
- [ ] Plot loss curves
- [ ] Test on test set
- [ ] Write architecture explanation
- [ ] Discuss limitations

### Nice to Have (Optional)
- [ ] Experiment with hyperparameters
- [ ] Try different architectures
- [ ] Additional visualizations

---

## 🚨 Before Submitting

### Final Checklist
- [ ] All TODO cells completed
- [ ] Run "Restart and run all" successfully
- [ ] All outputs visible in notebook
- [ ] Graphs display correctly
- [ ] Analysis sections written
- [ ] PDF generated from notebook
- [ ] PDF includes all outputs
- [ ] File names correct:
  - `Assignment2_[Student_ID].ipynb`
  - `Assignment2_[Student_ID].pdf`

---

## 💡 Quick Tips

### Debugging
1. **Print shapes constantly**
2. **Test with small data first** (100 samples)
3. **Use 2-3 epochs initially** to verify code works
4. **Visualize intermediate results**

### Efficiency
1. **Start simple** - basic model first
2. **Test incrementally** - don't write all code then test
3. **Use GPU** - move data/model to device
4. **Ask for help** - don't waste hours stuck

### Analysis
1. **Focus on understanding** over perfect metrics
2. **Explain imbalanced data** (AUC vs F1)
3. **Discuss limitations** honestly
4. **Propose realistic improvements**

---

## 🔗 Quick Links to Guides

- **Setup & GPU**: `01_GPU_Selection_Guide.md`
- **Libraries**: `02_Python_Libraries_Explained.md`
- **Dataset**: `03_Dataset_Explained.md`
- **CNN & Training**: `04_CNN_and_Training_Explained.md`
- **Time Management**: `05_Time_Management_Strategy.md`

---

## 📞 Getting Help

### When Stuck
1. Re-read relevant explanation file
2. Check common gotchas above
3. Google the specific error
4. Ask TA or peers
5. Use ChatGPT for syntax help

### What to Include When Asking
```
"I'm stuck on Part [X]. 

What I'm trying to do:
[Describe task]

What I tried:
[Show code]

Error I'm getting:
[Copy exact error]

Shapes I'm seeing:
[Print all relevant shapes]"
```

Good luck with your assignment! 🚀
