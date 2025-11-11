# Understanding the Dataset - RNA Sequences and Structures

## 🧬 What is the bpRNA Dataset?

The **bpRNA-1m** dataset contains RNA molecules with their sequences and known secondary structures.

### What's Included
1. **RNA Sequence**: The actual bases (nucleotides)
2. **Secondary Structure**: How the RNA folds (which bases pair)

---

## 📄 One Sample Explained

### Example RNA Molecule

#### **Sequence** (What you see in CSV):
```
AUGCGAUUCGAUUCGAUCCGAUUCGAU
```
- Length: 27 nucleotides
- Bases: A (Adenine), U (Uracil), G (Guanine), C (Cytosine)
- Linear string representation

#### **Structure** (Dot-bracket notation):
```
(((((.....)))))..(((...)))..
```
- Same length as sequence (27 characters)
- **`(`** = Opening of a pair
- **`)`** = Closing of a pair
- **`.`** = Unpaired base

#### **Pairing Interpretation**:
```
Position:  0 1 2 3 4 5 6 7 8 9 10 11 12 13 ...
Sequence:  A U G C G A U U C G A  U  U  C  ...
Structure: ( ( ( ( ( . . . . . )  )  )  )  ...
           ↑ ↑ ↑ ↑ ↑           ↑  ↑  ↑  ↑
           └─┴─┴─┴─┴───────────┘  │  │  │
             └─┴─┴───────────────┘  │  │
               └─────────────────────┘  │
                 └───────────────────────┘

Position 0 (A) pairs with Position 10 (A)
Position 1 (U) pairs with Position 9 (G)
Position 2 (G) pairs with Position 8 (C)
...
```

---

## 📊 Contact Map Representation

The **contact map** is a 2D matrix representation of the structure.

### Example Contact Map (27x27 for the sequence above):

```
        0  1  2  3  4  5  6  7  8  9 10 11 12 13 ...
     ┌──────────────────────────────────────────
  0  │  0  0  0  0  0  0  0  0  0  0  1  0  0  0  ...  ← Pos 0 pairs with pos 10
  1  │  0  0  0  0  0  0  0  0  0  1  0  0  0  0  ...  ← Pos 1 pairs with pos 9
  2  │  0  0  0  0  0  0  0  0  1  0  0  0  0  0  ...  ← Pos 2 pairs with pos 8
  3  │  0  0  0  0  0  0  0  1  0  0  0  0  0  0  ...
  4  │  0  0  0  0  0  0  1  0  0  0  0  0  0  0  ...
  5  │  0  0  0  0  0  0  0  0  0  0  0  0  0  0  ...  ← Unpaired (all 0s)
  6  │  0  0  0  0  1  0  0  0  0  0  0  0  0  0  ...
  7  │  0  0  0  1  0  0  0  0  0  0  0  0  0  0  ...
  8  │  0  0  1  0  0  0  0  0  0  0  0  0  0  0  ...
  9  │  0  1  0  0  0  0  0  0  0  0  0  0  0  0  ...
 10  │  1  0  0  0  0  0  0  0  0  0  0  0  0  0  ...
 ...
```

### Properties:
- **Size**: L × L (where L = sequence length)
- **Symmetric**: If M[i,j] = 1, then M[j,i] = 1
- **Sparse**: Mostly zeros (most bases are unpaired)
- **Binary**: Only 0s and 1s

---

## 📁 CSV File Structure

### Expected Format (TR0.csv, VL0.csv, TS0.csv):

```csv
id,sequence,structure
1,AUGCGAUUCGAUUCGAUCCGAUUCGAU,(((((.....)))))..(((...)))..
2,GCUAGCUAGCUAGCUA,(((...)))(((...)))
3,AAAAUUUUGGGGCCCC,....((((....))))
...
```

### Columns:
- **id**: Unique identifier for the RNA molecule
- **sequence**: RNA bases (A, U, G, C)
- **structure**: Dot-bracket notation

### Dataset Splits:
- **TR0.csv**: Training set (largest, ~60-80%)
- **VL0.csv**: Validation set (tuning, ~10-20%)
- **TS0.csv**: Test set (final evaluation, ~10-20%)

---

## 🔄 Data Transformation Pipeline

### Step-by-Step: From CSV to Model Input

#### 1. **Load CSV (Pandas)**
```python
df = pd.read_csv('TR0.csv')
```

#### 2. **Extract Sequences and Structures**
```python
sequence = "AUGCGAU"
structure = "((...))."
```

#### 3. **One-Hot Encode Sequence** (1D → 2D)
```python
# Mapping: A=0, U=1, G=2, C=3
sequence = "AUG"

# One-hot encoding:
[
  [1, 0, 0, 0],  # A
  [0, 1, 0, 0],  # U
  [0, 0, 1, 0]   # G
]
# Shape: (sequence_length, 4)
```

#### 4. **Create Contact Map from Structure**
```python
structure = "((.))"

# Use stack to find pairs:
# '(' at pos 0 → push to stack
# '(' at pos 1 → push to stack
# '.' at pos 2 → no pairing
# ')' at pos 3 → pop stack → pair pos 1 with pos 3
# ')' at pos 4 → pop stack → pair pos 0 with pos 4

# Resulting contact map:
[
  [0, 0, 0, 0, 1],  # Pos 0 pairs with pos 4
  [0, 0, 0, 1, 0],  # Pos 1 pairs with pos 3
  [0, 0, 0, 0, 0],  # Pos 2 unpaired
  [0, 1, 0, 0, 0],  # Pos 3 pairs with pos 1
  [1, 0, 0, 0, 0]   # Pos 4 pairs with pos 0
]
# Shape: (sequence_length, sequence_length)
```

#### 5. **Expand 1D to 2D for CNN**
```python
# One-hot encoded sequence: (L, 4)
# Need: (L, L, 8) for 2D CNN

# For each position pair (i, j):
# Concatenate features for base i and base j

# Example (simplified):
for i in range(L):
    for j in range(L):
        features[i, j] = concat(one_hot[i], one_hot[j])
        # Shape: (4,) + (4,) = (8,)

# Final shape: (L, L, 8)
```

---

## 🆚 Comparison: RNA Data vs Image Data

### Fundamental Differences

| Aspect | Image Data | RNA Data (This Assignment) |
|--------|-----------|----------------------------|
| **Input Type** | 2D grid of pixels | 1D sequence of bases |
| **Input Shape** | (Height, Width, 3) RGB | (Length, 4) One-hot |
| **Output Type** | Single label | 2D matrix |
| **Output Shape** | Scalar (0-999 classes) | (Length, Length) |
| **Task** | Classification | Structured Prediction |
| **Example Input** | 224×224×3 image | 100-base sequence |
| **Example Output** | "cat" (class 281) | 100×100 contact map |

### Visual Comparison

#### **Image Classification**:
```
Input: Photo of a cat (2D image)
       ┌────────────┐
       │   🐱      │  224×224×3
       │            │
       └────────────┘
           ↓ CNN
       "Cat" (label)
```

#### **RNA Structure Prediction**:
```
Input: RNA Sequence (1D string)
       AUGCGAUUCGAU... (length L)
           ↓ Encode to 2D
       ┌──────────────┐
       │   Features   │  L×L×8
       │     Grid     │
       └──────────────┘
           ↓ CNN
       Contact Map (2D matrix)
       ┌──────────────┐
       │ 0 0 0 0 1 0  │  L×L
       │ 0 0 0 1 0 0  │
       │ 0 0 0 0 0 0  │
       │ 0 1 0 0 0 0  │
       │ 1 0 0 0 0 0  │
       │ 0 0 0 0 0 0  │
       └──────────────┘
```

### Key Insight
**Images**: You're classifying **what's in** the data
**RNA**: You're predicting **relationships between positions** in the data

---

## 📊 Dataset Statistics (Typical)

### Sequence Properties:
- **Length**: Variable (20-500 nucleotides typically)
- **Composition**: ~25% each of A, U, G, C (varies)
- **Padding**: Sequences padded to `max_len` for batching

### Structure Properties:
- **Pairing Ratio**: ~10-30% of bases are paired
- **Imbalance**: Contact maps are ~90% zeros, ~10% ones
- **Symmetry**: Contact map is always symmetric

### Dataset Size:
- **Training**: Thousands of RNA molecules
- **Validation**: Hundreds of RNA molecules
- **Testing**: Hundreds of RNA molecules

---

## 🎯 Why This is Challenging

### 1. **Imbalanced Data**
```python
# In a 100×100 contact map:
Total positions: 100 × 100 = 10,000
Paired positions: ~300-500 (3-5%)
Unpaired positions: ~9,500-9,700 (95-97%)

# Model can achieve 95% accuracy by predicting all zeros!
# That's why we use F1 score instead of accuracy
```

### 2. **Long-Range Dependencies**
```python
# Bases far apart can still pair:
Position 5 can pair with Position 95
# 90 bases between them!

# CNNs have limited receptive field
# Hard to capture these long-range interactions
```

### 3. **Variable Length Sequences**
```python
# RNA sequences have different lengths:
Seq 1: 50 bases
Seq 2: 200 bases
Seq 3: 150 bases

# Solution: Pad to max_len
# Downside: Wasted computation on padding
```

---

## 💡 What Your Model Learns

### The Model's Job:
Given a sequence like `AUGCGAU`, predict which positions pair:
```
Input:  A U G C G A U
Output: Does A (pos 0) pair with U (pos 1)? → No
        Does A (pos 0) pair with G (pos 2)? → No
        Does A (pos 0) pair with U (pos 6)? → Maybe!
        Does U (pos 1) pair with A (pos 5)? → Maybe!
        ... (all L×L pairs)
```

### What Makes a Good Prediction:
- **Biological rules**: A-U and G-C pairs are common
- **Structural patterns**: Stems, loops, bulges
- **Distance constraints**: Nearby bases rarely pair
- **Symmetry**: If i pairs with j, then j pairs with i

---

## 🔬 Real-World Applications

### Why RNA Structure Prediction Matters:

1. **Drug Design**: RNA structures are drug targets
2. **Vaccine Development**: mRNA vaccines need stable structures
3. **Disease Understanding**: RNA misfolding causes diseases
4. **Synthetic Biology**: Design RNA with desired functions

### Example: COVID-19 mRNA Vaccines
- mRNA sequence encodes spike protein
- Secondary structure affects stability and translation
- Predicting structure helps optimize vaccine design

---

## 📚 Quick Reference

### Data Types You'll Work With:

```python
# String (from CSV)
sequence = "AUGCGAU"
structure = "((...))."

# NumPy arrays (after encoding)
one_hot = np.array([[1,0,0,0], [0,1,0,0], ...])  # Shape: (7, 4)
contact_map = np.array([[0,0,1], [0,0,0], ...])  # Shape: (7, 7)

# PyTorch tensors (for model)
seq_tensor = torch.tensor(one_hot).float()       # Shape: (7, 4)
map_tensor = torch.tensor(contact_map).float()   # Shape: (7, 7)

# Batched (in DataLoader)
batch_seqs = torch.randn(32, 7, 4)              # 32 sequences
batch_maps = torch.randn(32, 7, 7)              # 32 contact maps
```

### Common Operations:

```python
# Load CSV
df = pd.read_csv('TR0.csv')

# Get one sample
seq, struct = df.iloc[0]['sequence'], df.iloc[0]['structure']

# Encode
one_hot = one_hot_encode(seq, max_len=100)      # (100, 4)
contact = create_contact_map(struct, max_len=100)  # (100, 100)

# To tensor
seq_t = torch.from_numpy(one_hot).float()
contact_t = torch.from_numpy(contact).float()

# Model prediction
pred_contact = model(seq_t.unsqueeze(0))        # Add batch dim
```

This is your complete data pipeline!
