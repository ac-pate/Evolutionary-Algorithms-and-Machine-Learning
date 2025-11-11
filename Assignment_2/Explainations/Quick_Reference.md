# Quick Reference - RNA CNN Training

## Before Running

1. Ensure CSV files are in the same directory as notebook:
   - TR0.csv (training data)
   - VL0.csv (validation data)
   - TS0.csv (test data)

2. Check GPU availability:
   ```python
   import torch
   print(torch.cuda.is_available())
   ```

---

## Key Parameters (Easy to Tweak)

### In Dataset Section
```python
MAX_LEN = 128        # Change to 64 (faster) or 256 (more info)
BATCH_SIZE = 16      # Change to 8 (less memory) or 32 (smoother training)
```

### In Training Section
```python
num_epochs = 20      # Change to 10 (quick test) or 50 (full training)
lr = 0.001          # In optimizer: change to 0.0001 (stable) or 0.01 (fast)
threshold = 0.5     # In calculate_metrics: change to 0.3 (better F1)
```

### In DataLoader (if on Linux/Mac)
```python
num_workers = 0     # Change to 2 or 4 for faster data loading
```

---

## Expected Results

### Normal Performance
- Training loss: Decreases smoothly from ~0.5 to ~0.1-0.2
- Validation loss: Similar to training, slight gap
- AUC: 0.7-0.9 (high due to class imbalance)
- F1: 0.1-0.3 (low is NORMAL for this task)
- Training time: 2-5 minutes per epoch on GPU, 10-20 on CPU

### Signs of Problems
- Loss stays constant: Learning rate too low or model issue
- Loss explodes (NaN): Learning rate too high
- Train loss << Val loss: Overfitting
- Very slow training: Reduce MAX_LEN or BATCH_SIZE

---

## Common Adjustments

### If Out of Memory
1. `BATCH_SIZE = 8` (or even 4)
2. `MAX_LEN = 64`
3. Close other programs

### If Training Too Slow
1. `MAX_LEN = 64`
2. `BATCH_SIZE = 32`
3. `num_epochs = 10`
4. Check GPU is being used

### If F1 Score Too Low
1. `threshold = 0.3` in calculate_metrics
2. Use weighted loss (see Implementation Guide)
3. This is expected - F1 of 0.2-0.3 is acceptable

### If Overfitting
1. Reduce model size (see Implementation Guide)
2. Add dropout (see Implementation Guide)
3. Reduce epochs
4. Increase BATCH_SIZE

---

## Experiment Template

When trying new parameters, document:
```
Experiment: [Name]
Changes: [What you changed]
MAX_LEN: [value]
BATCH_SIZE: [value]
Learning Rate: [value]
Epochs: [value]
Other: [any other changes]

Results:
- Train Loss: [final value]
- Val Loss: [final value]
- Test F1: [value]
- Test AUC: [value]
- Training Time: [total time]

Notes: [observations]
```

---

## Files Generated

After running:
- `best_model.pth` - Your trained model
- `training_metrics.png` - Loss and metric curves
- `contact_map_comparison.png` - Example predictions

---

## Troubleshooting Quick Fixes

| Problem | Quick Fix |
|---------|-----------|
| CUDA out of memory | BATCH_SIZE = 8 |
| Training very slow | MAX_LEN = 64 |
| Loss is NaN | lr = 0.0001 |
| Not learning | Check data loading cell output |
| F1 very low | threshold = 0.3 (this is expected) |
| Overfitting | Reduce epochs or model size |

---

## Running the Notebook

1. Run cells in order from top to bottom
2. Wait for each cell to complete before running next
3. Watch for progress bars and printed output
4. Check final plots and metrics
5. Save your experiment results

---

## What to Report

For assignment:
1. Final test metrics (F1, AUC)
2. Training curve plots
3. Contact map visualization
4. Analysis of overfitting
5. Discussion of imbalanced data
6. Proposed improvements

---

## Next Steps After Baseline

1. Run baseline with default parameters
2. Document baseline results
3. Try threshold=0.3 for better F1
4. Experiment with MAX_LEN based on your data
5. Try different learning rates
6. Compare all results
7. Write analysis

---

## Important Notes

- F1 score of 0.1-0.3 is NORMAL for this task
- AUC will be much higher than F1 (this is expected)
- Training on CPU is possible but slow
- Each experiment takes ~5-20 minutes depending on hardware
- Save your plots after each run (they get overwritten)
