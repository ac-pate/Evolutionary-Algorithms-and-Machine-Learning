# Design Decisions and Rationale

## Overview

This document explains the key choices made in the implementation and provides context for why certain approaches were selected.

---

## Architecture Decisions

### Why CNN Instead of RNN or Transformer?

**Choice:** 2D Convolutional Neural Network

**Reasoning:**
1. Contact maps are 2D spatial data (like images)
2. CNNs excel at finding local patterns in spatial data
3. Base pair interactions have local structure that CNNs can learn
4. Simpler to implement and train than Transformers
5. Less data needed compared to Transformers

**Trade-offs:**
- Pro: Fast training, efficient, good for local patterns
- Con: Limited ability to capture very long-range dependencies
- Con: Fixed receptive field based on kernel sizes

**Alternatives considered:**
- RNN/LSTM: Better for sequential dependencies but slower, harder to train
- Transformer: Best for long-range but requires more data and compute
- Graph Neural Networks: Natural for structure but more complex

---

### Why This Specific Layer Configuration?

**Choice:** 8 -> 32 -> 64 -> 32 -> 1 channel progression

**Reasoning:**
1. Progressive expansion (8 -> 32 -> 64) allows learning hierarchical features
2. Reduction (64 -> 32 -> 1) prevents overfitting and compresses information
3. Moderate width (32, 64) balances capacity with training speed
4. Small enough to train quickly, large enough to learn patterns

**Why not deeper?**
- Deeper networks need more data
- Risk of vanishing gradients
- Diminishing returns for this relatively simple task
- Want to keep training fast for experimentation

**Why not wider?**
- More parameters increase overfitting risk with limited data
- Slower training and inference
- Higher memory requirements
- Current width sufficient for task complexity

---

### Why 3x3 Kernels?

**Choice:** kernel_size=3 for all convolutional layers except final

**Reasoning:**
1. Standard choice in computer vision (proven effective)
2. Captures local patterns without too many parameters
3. Good balance of receptive field and efficiency
4. Stacking multiple 3x3 layers gives larger effective receptive field

**Math:**
- Layer 1: sees 3x3 region
- Layer 2: sees 5x5 region (3 + 2*1)
- Layer 3: sees 7x7 region (5 + 2*1)
- Effective receptive field: 7x7 after 3 layers

**Why not 5x5 or 7x7?**
- More parameters with same effective receptive field
- Two 3x3 layers have fewer parameters than one 5x5
- 3x3 is more flexible (more non-linearities between operations)

---

### Why BatchNorm?

**Choice:** BatchNormalization after each convolutional layer

**Reasoning:**
1. Dramatically improves training stability
2. Allows higher learning rates (faster training)
3. Reduces sensitivity to weight initialization
4. Acts as mild regularization
5. Industry standard for CNNs

**When it helps most:**
- Deeper networks (3+ layers)
- Higher learning rates
- Small batch sizes (though less effective < 4)

**Alternative (not used):**
- LayerNorm: Normalizes per sample instead of per batch
- Good for very small batch sizes or RNNs
- Not standard for CNNs

---

## Data Processing Decisions

### Why MAX_LEN = 128?

**Choice:** Fixed length of 128 bases

**Analysis of dataset:**
- Most sequences 100-150 bases long
- Very few > 200 bases
- 128 is power of 2 (GPU efficient)

**Trade-offs:**
- Too small (64): Truncates many sequences, loses information
- Too large (256): Slower training, more memory, mostly empty padding
- 128: Good middle ground for this dataset

**Dataset-dependent:**
- Check your actual sequence lengths
- Choose MAX_LEN to cover 90-95% of sequences
- Prefer powers of 2 for GPU efficiency

---

### Why One-Hot Encoding?

**Choice:** One-hot vectors for nucleotides

**Reasoning:**
1. Treats all bases equally (no artificial ordering)
2. Standard in bioinformatics
3. Easy for neural networks to process
4. Clear interpretation

**Why not integer encoding (A=0, C=1, G=2, U=3)?**
- Implies false relationships (G is not "greater than" A)
- Neural networks might learn spurious patterns from numbers
- One-hot is more explicit

**Why not learned embeddings?**
- Only 4 bases (vocabulary too small)
- One-hot works well for this size
- Embeddings add unnecessary complexity

---

### Why Symmetry Enforcement?

**Choice:** Average output with its transpose

```python
x = (x + x.transpose(1, 2)) / 2
```

**Reasoning:**
1. Base pairs are inherently symmetric (if i pairs with j, j pairs with i)
2. Contact maps must be symmetric matrices
3. Model might not learn perfect symmetry naturally
4. Enforcing it ensures valid biological structure

**When applied:**
- After sigmoid (preserves probability interpretation)
- Before returning final output

**Alternative (not used):**
- Let model learn symmetry naturally
- Risk: Model predicts different values for (i,j) and (j,i)
- Result: Biologically invalid structures

---

## Training Decisions

### Why Adam Optimizer?

**Choice:** Adam with default parameters

**Reasoning:**
1. Adapts learning rate per parameter (handles different scales)
2. Combines momentum and RMSProp (best of both)
3. Works well out-of-the-box for most tasks
4. Less sensitive to learning rate choice than SGD
5. Industry standard for deep learning

**Why not SGD?**
- Requires more careful learning rate tuning
- Often needs learning rate schedules
- Can be slower to converge
- Adam is more forgiving

**Why not other optimizers (AdamW, RMSProp)?**
- AdamW adds weight decay (good for larger models)
- RMSProp is older, Adam improves on it
- Adam is safest default choice

---

### Why Learning Rate = 0.001?

**Choice:** lr=0.001

**Reasoning:**
1. Default for Adam optimizer
2. Proven effective across many tasks
3. Good starting point for experimentation
4. Not too fast (stable), not too slow

**How chosen:**
- 0.001 is standard Adam default
- Works for 80% of cases
- Adjust if problems arise

**When to change:**
- Loss explodes: Reduce to 0.0001
- Training too slow: Increase to 0.01
- Fine-tuning: Reduce to 0.0001

---

### Why BATCH_SIZE = 16?

**Choice:** 16 samples per batch

**Reasoning:**
1. Fits comfortably in most GPUs (4-8GB)
2. Good balance of memory and gradient quality
3. Batch norm works well with this size
4. Fast enough iteration speed

**Trade-offs:**
- Smaller (4, 8): Less memory, noisier gradients
- Larger (32, 64): Smoother gradients, more memory
- 16: Sweet spot for this problem

**Hardware-dependent:**
- Adjust based on your GPU memory
- Larger is generally better if you have memory
- Don't go below 4 (batch norm becomes unstable)

---

### Why 20 Epochs?

**Choice:** 20 training epochs

**Reasoning:**
1. Enough to see convergence trends
2. Not too long for experimentation
3. Can always train longer if needed
4. Typical convergence around 10-30 epochs

**Observation from practice:**
- Loss usually stabilizes by epoch 10-15
- More epochs may improve slightly
- Diminishing returns after 30-40 epochs

**Recommendation:**
- Start with 10 epochs for quick tests
- Use 20-30 for serious training
- Monitor validation loss to decide when to stop

---

## Loss Function Decisions

### Why BCELoss?

**Choice:** Binary Cross-Entropy Loss

**Reasoning:**
1. Standard for binary classification (paired vs not paired)
2. Each position is independent binary decision
3. Naturally handles probabilities (0-1 range)
4. Well-understood and stable

**Why not other losses?**
- BCEWithLogitsLoss: More numerically stable (combines sigmoid + BCE)
  - Current implementation is fine since we use sigmoid explicitly
- MSE: Not appropriate for binary classification
- Cross-Entropy: Same as BCE for binary case

**Could improve with:**
- Weighted BCE: Account for class imbalance
- Focal loss: Focus on hard examples
- Dice loss: Better for segmentation-like tasks

---

### Why Not Weighted Loss?

**Choice:** Unweighted BCE (all positions equal)

**Reasoning:**
1. Simplicity for baseline implementation
2. Easier to debug and understand
3. Weighted loss adds complexity

**Trade-offs:**
- Pro: Simpler, standard approach
- Con: Doesn't account for 90% negatives vs 10% positives
- Con: Model may bias toward predicting 0

**When to add weighting:**
- If F1 score is very low (< 0.1)
- If model predicts all zeros
- For improved performance

**How to implement:**
```python
pos_weight = negative_samples / positive_samples
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
```

---

## Evaluation Decisions

### Why F1 Score?

**Choice:** Primary metric is F1 score

**Reasoning:**
1. Balances precision and recall
2. Better than accuracy for imbalanced data
3. Single number for model comparison
4. Standard in bioinformatics

**Why accuracy is misleading:**
- Predicting all zeros gives 90% accuracy (useless model)
- Doesn't reflect performance on minority class
- F1 focuses on actual base pair predictions

**Why also report AUC:**
- Less sensitive to threshold choice
- Measures ranking ability
- Complementary to F1

---

### Why Threshold = 0.5?

**Choice:** Default classification threshold of 0.5

**Reasoning:**
1. Standard default for binary classification
2. Natural midpoint of probability range
3. Equal treatment of both classes

**Why it may not be optimal:**
- Data is imbalanced (more 0s than 1s)
- May want to favor recall over precision
- Optimal threshold often differs from 0.5

**How to find better threshold:**
1. Try 0.3, 0.4, 0.5, 0.6, 0.7
2. Plot F1 score vs threshold
3. Choose threshold with best F1

**Expected:**
- Lower threshold (0.3): Higher recall, lower precision, better F1
- Higher threshold (0.7): Lower recall, higher precision, worse F1

---

## Design Philosophy

### Prioritize Simplicity

**Principle:** Start simple, add complexity only when needed

**Applied to:**
1. Model architecture: Small, shallow network
2. Training: Standard Adam, no tricks
3. Data processing: Basic one-hot encoding
4. Loss: Unweighted BCE

**Rationale:**
- Easier to debug
- Faster experimentation
- Clear what works and what doesn't
- Can always add complexity later

---

### GPU-First Design

**Principle:** Make it work efficiently on GPU

**Applied to:**
1. Batch processing: Avoid loops, use vectorized ops
2. Data types: float32 instead of float64
3. MAX_LEN choice: Power of 2 for efficient memory
4. Batch size: Large enough to utilize GPU

**Rationale:**
- 10-50x speedup on GPU vs CPU
- Modern deep learning is GPU-centric
- Training should take minutes, not hours

---

### Experiment-Friendly

**Principle:** Easy to modify and test different ideas

**Applied to:**
1. Clear parameter definitions at top
2. Modular functions (easy to replace)
3. Comprehensive logging and visualization
4. Save best model automatically

**Rationale:**
- Machine learning is iterative
- Need to try many configurations
- Quick feedback loop is crucial

---

## What I Would Change With More Time/Resources

### 1. Attention Mechanisms

Add self-attention to capture long-range dependencies:
```python
class AttentionBlock(nn.Module):
    def __init__(self, channels):
        self.attention = nn.MultiheadAttention(channels, num_heads=4)
```

**Why:** Better at modeling distant base pairs

---

### 2. Data Augmentation

Implement sequence augmentation:
- Reverse complement
- Random cropping
- Noise injection

**Why:** Artificially increase training data

---

### 3. Ensemble Methods

Train multiple models and average predictions:
```python
predictions = (model1(x) + model2(x) + model3(x)) / 3
```

**Why:** Usually improves performance 2-5%

---

### 4. Hyperparameter Optimization

Use libraries like Optuna for automatic tuning:
- Learning rate
- Model architecture
- Batch size
- Regularization strength

**Why:** Find optimal configuration systematically

---

### 5. More Sophisticated Architecture

Implement U-Net or ResNet-style architecture:
- Skip connections
- Multiple scales
- Deeper network

**Why:** Better feature learning, proven in similar tasks

---

## Conclusion

The implementation prioritizes:
1. Simplicity and clarity
2. GPU efficiency
3. Easy experimentation
4. Solid baseline performance

This provides a strong foundation for understanding deep learning applied to bioinformatics while keeping complexity manageable for learning purposes.

Every design choice is a trade-off. The current choices optimize for learning, experimentation, and getting reasonable results quickly. Advanced techniques can be added once the baseline is working and understood.
