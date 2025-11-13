# Fixes Applied to Address Zero F1 Score Issue

## Problem Diagnosis

Your model was achieving:
- ✅ High AUC scores (0.88 → 0.96): Model can rank predictions correctly
- ❌ Zero F1 scores: Model predicts almost no base pairs (all predictions < 0.5)

This is a **classic class imbalance problem** where:
- ~90% of contact map positions are 0 (non-paired bases)
- ~10% of positions are 1 (paired bases)
- Model learns to predict low probabilities for everything

## Solutions Implemented

### 1. Weighted Loss Function
**Changed from:** `nn.BCELoss()`
**Changed to:** `nn.BCEWithLogitsLoss(pos_weight=torch.tensor([10.0]))`

**Why this helps:**
- Gives 10x more weight to positive class (base pairs)
- Forces model to pay more attention to rare positive examples
- BCEWithLogitsLoss is more numerically stable than BCE + Sigmoid

### 2. Model Architecture Update
**Modified:** `RNAFoldingCNN.forward()` to handle logits

**Changes:**
- During training: Returns raw logits (for BCEWithLogitsLoss)
- During inference: Applies sigmoid to get probabilities
- Maintains symmetry enforcement in both modes

### 3. Threshold Optimization
**Added:** New cell (3.4) to find optimal classification threshold

**Features:**
- Tests thresholds from 0.05 to 0.95
- Plots F1, Precision, and Recall vs threshold
- Shows prediction distribution histogram
- Automatically finds threshold that maximizes F1 score

**Why this matters:**
- Default threshold (0.5) is rarely optimal for imbalanced data
- Optimal threshold typically 0.2-0.4 for this problem
- Allows trading precision for recall to improve F1

## Expected Results After Re-training

You should now see:
1. **Non-zero F1 scores** during training (likely 0.15-0.35)
2. **Optimal threshold** around 0.2-0.4 (not 0.5)
3. **Better balance** between precision and recall
4. **Prediction distribution** showing model actually predicts some 1s

## How to Use

1. **Re-run training cells** (Parts 2 and 3)
2. **Run new threshold analysis** (Section 3.4)
3. **Use optimal threshold** in test evaluation and visualization cells

Example:
```python
# Instead of:
calculate_metrics(outputs, batch_y, threshold=0.5)

# Use:
calculate_metrics(outputs, batch_y, threshold=optimal_threshold)
```

## Additional Recommendations

### For even better results:
1. **Increase pos_weight**: Try 15.0 or 20.0 if F1 still low
2. **Lower learning rate**: Try 0.0005 for more stable training
3. **More epochs**: Train for 20-30 epochs if you have time
4. **Focal Loss**: Consider using focal loss for extreme imbalance

### For the report:
Discuss:
- Why weighted loss is necessary for imbalanced data
- How threshold selection impacts precision-recall trade-off
- Why AUC can be high while F1 is low (ranking vs classification)
- The asymmetry between false positives and false negatives in RNA structure

## Quick Test

To verify the fix works, check after epoch 1:
- Val F1 should be > 0 (even if small like 0.10)
- Model should predict some probabilities > 0.5
- Loss should be higher initially (due to pos_weight) but converge better
