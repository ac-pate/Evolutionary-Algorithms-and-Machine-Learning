# Project Directions: Advanced Loss Functions & Model Improvements

## Overview

This document provides alternative loss functions and architectural improvements to enhance your RNA secondary structure prediction model. Moving beyond basic BCEWithLogitsLoss, we explore multiple strategies to improve F1 score and generalization.

---

## 🎯 Part 1: Alternative Loss Functions

### **1. Focal Loss (Recommended for Imbalanced Data)**

**Why Focal Loss?**
- Better handles class imbalance (90% zeros, 10% ones)
- Focuses on hard-to-classify pairs (misclassified base pairs)
- Reduces overwhelming contribution from easy negatives
- Often improves F1 score by 5-15%

**Implementation:**

```python
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        """
        Focal Loss for imbalanced binary classification
        
        Args:
            alpha: Weight for positive class (0-1). Try 0.25 for your 10% positive class
            gamma: Focusing parameter. Higher = more focus on hard examples. Try 2.0-5.0
            reduction: 'mean', 'sum', or 'none'
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        # Apply sigmoid to get probabilities
        p = torch.sigmoid(inputs)
        
        # Calculate focal loss components
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        p_t = p * targets + (1 - p) * (1 - targets)  # p if target=1, 1-p if target=0
        focal_weight = (1 - p_t) ** self.gamma
        
        # Apply alpha weighting
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        focal_loss = alpha_t * focal_weight * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# Usage in train_model():
criterion = FocalLoss(alpha=0.25, gamma=2.0)
```

**Hyperparameter Tuning:**
- `alpha`: Lower (0.1-0.2) = less weight on positives, Higher (0.3-0.5) = more weight on positives
- `gamma`: Lower (1.0-2.0) = gentler focusing, Higher (3.0-5.0) = aggressive focusing

---

### **2. Dice Loss (Recommended for Overlapping Predictions)**

**Why Dice Loss?**
- Directly optimizes overlap (similar to F1 score)
- Handles class imbalance naturally
- No need for pos_weight tuning
- Works well for segmentation-like tasks (contact maps are 2D segmentation)

**Implementation:**

```python
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        """
        Dice Loss - directly optimizes for F1-like metric
        
        Args:
            smooth: Smoothing factor to avoid division by zero
        """
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, inputs, targets):
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(inputs)
        
        # Flatten tensors
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)
        
        # Calculate Dice coefficient
        intersection = (probs_flat * targets_flat).sum()
        dice = (2. * intersection + self.smooth) / (probs_flat.sum() + targets_flat.sum() + self.smooth)
        
        # Return loss (1 - dice)
        return 1 - dice

# Usage:
criterion = DiceLoss(smooth=1e-6)
```

**Mathematical Foundation:**

$$\text{Dice} = \frac{2|X \cap Y|}{|X| + |Y|}$$

where $X$ is predicted pairs and $Y$ is true pairs.

---

### **3. Combined Loss (Best of Both Worlds)**

**Why Combined Loss?**
- Focal handles hard negatives
- Dice optimizes for overlap
- Best empirical results in medical segmentation (similar to contact maps)
- More stable training

**Implementation:**

```python
class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.5, focal_gamma=2.0, dice_smooth=1e-6):
        """
        Combines Focal Loss (for hard examples) + Dice Loss (for overlap)
        
        Args:
            alpha: Weight between focal (alpha) and dice (1-alpha). Try 0.3-0.7
            focal_gamma: Focusing parameter for focal loss
            dice_smooth: Smoothing for dice loss
        """
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.focal = FocalLoss(alpha=0.25, gamma=focal_gamma)
        self.dice = DiceLoss(smooth=dice_smooth)
    
    def forward(self, inputs, targets):
        focal_loss = self.focal(inputs, targets)
        dice_loss = self.dice(inputs, targets)
        return self.alpha * focal_loss + (1 - self.alpha) * dice_loss

# Usage:
criterion = CombinedLoss(alpha=0.5, focal_gamma=2.0)
```

**Loss Combination Tuning:**
- `alpha = 0.7`: More weight on Focal Loss (better for hard examples)
- `alpha = 0.5`: Balanced combination (recommended)
- `alpha = 0.3`: More weight on Dice Loss (better for overlap)

---

## 🚀 Part 2: Architecture Improvements

### **1. Dilated Convolutions for Long-Range Dependencies**

**Problem:** Standard 3x3 convolutions have limited receptive field. Distant base pairs (e.g., position 10 to 60) are hard to learn.

**Solution:** Dilated (atrous) convolutions expand receptive field exponentially.

```python
class ImprovedRNAFoldingCNN(nn.Module):
    def __init__(self, input_channels=8):
        super(ImprovedRNAFoldingCNN, self).__init__()
        
        # Initial convolution
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Dilated convolutions to capture long-range interactions
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=2, dilation=2)  # dilation=2
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=4, dilation=4)  # dilation=4
        self.bn3 = nn.BatchNorm2d(64)
        
        # Squeeze back down
        self.conv4 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(32)
        
        self.conv5 = nn.Conv2d(32, 1, kernel_size=1)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout2d(0.2)  # Add dropout for regularization

    def forward(self, x_1d):
        batch_size = x_1d.shape[0]
        max_len = x_1d.shape[1]
        
        # Expand to 2D
        x_2d_i = x_1d.unsqueeze(2).repeat(1, 1, max_len, 1)
        x_2d_j = x_1d.unsqueeze(1).repeat(1, max_len, 1, 1)
        x_2d = torch.cat([x_2d_i, x_2d_j], dim=-1)
        x_2d = x_2d.permute(0, 3, 1, 2)
        
        # Apply convolutions with dilations
        x = self.relu(self.bn1(self.conv1(x_2d)))
        x = self.dropout(x)
        
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.dropout(x)
        
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.conv5(x)
        
        x = x.squeeze(1)
        
        # Symmetrize
        if self.training:
            x = (x + x.transpose(1, 2)) / 2
            return x
        else:
            x = torch.sigmoid(x)
            x = (x + x.transpose(1, 2)) / 2
            return x
```

**Receptive Field Comparison:**

| Layer | Dilation | Kernel Size | Effective Receptive Field |
|-------|----------|------------|--------------------------|
| Conv1 | 1 | 3x3 | 3x3 |
| Conv2 | 2 | 3x3 | 7x7 |
| Conv3 | 4 | 3x3 | 15x15 |

**Benefits:**
- ✅ Captures long-range base pairing (e.g., position 10 pairing with position 50)
- ✅ Maintains parameter efficiency
- ✅ Better receptive field without max pooling

---

### **2. Residual Connections (ResNet-style)**

**Problem:** Deeper networks suffer from vanishing gradients and degradation.

**Solution:** Skip connections allow gradients to flow directly.

```python
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual  # Skip connection
        out = self.relu(out)
        return out

class ResNetRNAFolding(nn.Module):
    def __init__(self, input_channels=8):
        super(ResNetRNAFolding, self).__init__()
        
        self.conv_in = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.bn_in = nn.BatchNorm2d(64)
        
        self.res1 = ResidualBlock(64)
        self.res2 = ResidualBlock(64)
        self.res3 = ResidualBlock(64)
        
        self.conv_out = nn.Conv2d(64, 1, kernel_size=1)
        self.relu = nn.ReLU()
    
    def forward(self, x_1d):
        # Expand to 2D (same as before)
        batch_size, max_len, _ = x_1d.shape
        x_2d_i = x_1d.unsqueeze(2).repeat(1, 1, max_len, 1)
        x_2d_j = x_1d.unsqueeze(1).repeat(1, max_len, 1, 1)
        x_2d = torch.cat([x_2d_i, x_2d_j], dim=-1).permute(0, 3, 1, 2)
        
        x = self.relu(self.bn_in(self.conv_in(x_2d)))
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.conv_out(x).squeeze(1)
        
        # Symmetrize
        if self.training:
            return (x + x.transpose(1, 2)) / 2
        else:
            x = torch.sigmoid(x)
            return (x + x.transpose(1, 2)) / 2
```

**Benefits:**
- ✅ Deeper networks without vanishing gradients
- ✅ Better gradient flow
- ✅ Empirically better F1 scores (5-10% improvement)

---

## 📊 Part 3: Training Optimizations

### **1. Data Augmentation**

```python
def augment_rna_sequence(sequence, structure):
    """Data augmentation for RNA sequences"""
    import random
    
    # 1. Reverse complement (biological equivalence)
    if random.random() < 0.3:
        complement = {'A': 'U', 'U': 'A', 'G': 'C', 'C': 'G'}
        sequence = ''.join([complement.get(b, b) for b in sequence[::-1]])
        structure = structure[::-1]
    
    # 2. Random masking (10% of bases with noise)
    if random.random() < 0.3:
        seq_list = list(sequence)
        mask_indices = random.sample(range(len(seq_list)), k=int(0.1 * len(seq_list)))
        for idx in mask_indices:
            seq_list[idx] = random.choice(['A', 'C', 'G', 'U'])
        sequence = ''.join(seq_list)
    
    return sequence, structure

# Modify RNADataset:
class RNADataset(Dataset):
    def __init__(self, data, max_len, augment=False):
        self.data = data
        self.max_len = max_len
        self.augment = augment

    def __getitem__(self, idx):
        sequence, structure = self.data[idx]
        
        if self.augment:
            sequence, structure = augment_rna_sequence(sequence, structure)
        
        seq_encoded = one_hot_encode(sequence, self.max_len)
        contact_map = create_contact_map(structure, self.max_len)
        # ... rest of code

# Usage:
train_dataset = RNADataset(train_data, MAX_LEN, augment=True)
```

**Augmentation Strategies:**
- **Reverse Complement**: RNA is bidirectional; reverse complement is biologically equivalent
- **Noise Injection**: Adds robustness to sequence variations
- **Random Cropping**: Learn translation invariance (optional)

---

### **2. Learning Rate Scheduling**

```python
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR

# In train_model():
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

# Option 1: Reduce LR on plateau (when val F1 stops improving)
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)

# Option 2: Cosine annealing (smooth decay over epochs)
# scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

# In training loop, after each epoch:
scheduler.step(avg_val_f1)  # For ReduceLROnPlateau
# scheduler.step()  # For CosineAnnealingLR
```

**When to Use:**
- **ReduceLROnPlateau**: When validation metric plateaus (most flexible)
- **CosineAnnealingLR**: Smooth decay over fixed epochs (more predictable)

---

### **3. Gradient Clipping**

```python
# In training loop:
loss.backward()

# Add gradient clipping before optimizer.step()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

optimizer.step()
```

**Benefits:**
- ✅ Prevents gradient explosion (exploding gradients)
- ✅ More stable training
- ✅ Especially important with imbalanced data

---

### **4. Mixed Precision Training (Faster + Less Memory)**

```python
from torch.cuda.amp import autocast, GradScaler

# In train_model():
scaler = GradScaler()

# In training loop:
for batch_x, batch_y in train_loader:
    optimizer.zero_grad()
    
    with autocast():  # Mixed precision (FP16 + FP32)
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

**Benefits:**
- ✅ 2-3x faster training
- ✅ 30-40% less GPU memory
- ✅ No accuracy loss

---

## 📋 Recall: High or Low?

### **Answer: Recall Should Be HIGH**

**What is Recall?**
```
Recall = True Positives / (True Positives + False Negatives)
       = Correctly predicted pairs / All actual pairs
```

**Why High Recall Matters for RNA:**
- **Missing a base pair (False Negative)** = Incomplete structure → Wrong biological function ❌
- **Adding an extra pair (False Positive)** = Can be filtered later with constraints ✓
- **Better to be sensitive than specific** for structure prediction

**Your Results Analysis:**
- ✅ pos_weight = 7 gives recall ≈ 0.43 (good balance)
- ✅ pos_weight = 15 gives recall ≈ 0.56 (higher but F1 drops)
- ✓ Sweet spot is around pos_weight = 7 for best F1

---

## 🎯 Recommended Configuration for Best Results

```python
# BEST CONFIGURATION
num_epochs = 30  # Increase from 20
MAX_LEN = 128
BATCH_SIZE = 64  # Reduce from 256 (better for imbalanced data)
THRESHOLD = 0.4
POS_WEIGHT_START = 5
POS_WEIGHT_END = 10  # Focus on sweet spot

# Use Combined Loss
criterion = CombinedLoss(alpha=0.5, focal_gamma=2.0)

# Use Improved Architecture
model = ImprovedRNAFoldingCNN(input_channels=8).to(device)
# OR
# model = ResNetRNAFolding(input_channels=8).to(device)

# Use Learning Rate Scheduler
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

# Use Mixed Precision
scaler = GradScaler()

# Use Data Augmentation
train_dataset = RNADataset(train_data, MAX_LEN, augment=True)
```

---

## 📈 Expected Improvements

| Change | Expected F1 Improvement | Notes |
|--------|-------------------------|-------|
| Focal Loss | +5-10% | Focus on hard examples |
| Dilated Convolutions | +3-7% | Long-range dependencies |
| Residual Connections | +5-10% | Better gradient flow |
| Data Augmentation | +2-5% | More training data |
| LR Scheduling | +2-4% | Dynamic learning rate |
| Batch Size 64 | +3-5% | Better for imbalance |
| **Combined** | **+15-30%** | All together |

**Your F1 Trajectory:**
- **Current**: F1 ≈ 0.30 (pos_weight=7)
- **With improvements**: F1 ≈ 0.40-0.45 🚀

---

## 🔍 Comparing Loss Functions

### **BCEWithLogitsLoss (Current)**
- ✅ Simple, fast
- ❌ Doesn't focus on hard examples
- ❌ Poor handling of class imbalance alone

### **Focal Loss**
- ✅ Focuses on hard examples
- ✅ Better for imbalanced data
- ❌ Doesn't directly optimize F1

### **Dice Loss**
- ✅ Directly optimizes overlap/F1
- ✅ Natural class balance handling
- ❌ Can be less stable early in training

### **Combined Loss**
- ✅ Best of both worlds
- ✅ Stable training + F1 optimization
- ✅ **RECOMMENDED** ⭐

---

## 📚 Implementation Priority

### **Phase 1: Quick Wins (Day 1)**
1. ✅ Change `BATCH_SIZE = 64`
2. ✅ Add Focal Loss
3. ✅ Add LR Scheduling

**Expected improvement:** +8-12%

### **Phase 2: Architecture (Day 2)**
1. ✅ Replace with Dilated CNN
2. ✅ Add Dropout
3. ✅ Add Gradient Clipping

**Expected improvement:** +5-10%

### **Phase 3: Polish (Day 3)**
1. ✅ Add Data Augmentation
2. ✅ Mixed Precision Training
3. ✅ Fine-tune hyperparameters

**Expected improvement:** +2-8%

---

## 🏆 Final Recommendations

**For your assignment submission:**

```python
# Option 1: Quick wins (minimal code changes)
BATCH_SIZE = 64
criterion = FocalLoss(alpha=0.25, gamma=2.0)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

# Option 2: Full improvements (best results)
BATCH_SIZE = 64
criterion = CombinedLoss(alpha=0.5, focal_gamma=2.0)
model = ImprovedRNAFoldingCNN(input_channels=8).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
scaler = GradScaler()  # For mixed precision
train_dataset = RNADataset(train_data, MAX_LEN, augment=True)
```

**Go with Option 2** for best results! 🎯
